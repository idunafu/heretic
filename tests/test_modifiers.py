import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import torch
import tomli_w
from peft import LoraConfig, PeftModel
from peft.tuners.lora.layer import Linear
from pydantic import ValidationError
from pydantic_settings import TomlConfigSettingsSource
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from heretic.config import Settings
from heretic.model import AbliterationParameters, Model
from heretic.modifier import create_modifier
from heretic.modifiers.som import SOM, fit_directions
from heretic.utils import Prompt, generate_reproduce_json, get_trial_parameters


HAS_SOM = importlib.util.find_spec("minisom") is not None


def settings(**kwargs: Any) -> Settings:
    with patch("sys.argv", ["heretic"]):
        options: dict[str, Any] = {"model": "unused"}
        options.update(kwargs)
        return Settings(**options)


class ModifierConfigTests(unittest.TestCase):
    def test_saved_settings_override_current_modifier_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                'modifier = "som"\nsom_direction_scope = "auto"\nsom_grid_shape = [3, 5]\n'
            )
            with (
                patch("sys.argv", ["heretic", "--modifier", "som"]),
                patch.dict(os.environ, {"HERETIC_MODIFIER": "som"}),
                patch(
                    "heretic.config.TomlConfigSettingsSource",
                    side_effect=lambda cls, **kwargs: TomlConfigSettingsSource(
                        cls, toml_file=config_path
                    ),
                ),
            ):
                # Actual pre-modifier settings omit the field entirely.
                saved = {"model": "unused", "seed": 42}
                restored = Settings.from_saved(saved)
                self.assertEqual(restored.modifier, "abliteration")
                self.assertNotIn("modifier", saved)
                legacy_parameters = {"direction_index": 1.0, "parameters": {}}
                trial = optuna.trial.create_trial(value=0, user_attrs=legacy_parameters)
                modifier = create_modifier(restored, MagicMock(spec=Model))
                self.assertEqual(
                    modifier.parameters_from_trial(trial), legacy_parameters
                )

                # Older SOM settings must not inherit a new grid or auto scope.
                old_som = saved | {"modifier": "som", "som_grid_size": 4}
                restored = Settings.from_saved(old_som)
                self.assertEqual(restored.modifier, "som")
                self.assertEqual(restored.som_direction_scope, "global")
                self.assertIsNone(restored.som_grid_shape)
                self.assertNotIn("som_grid_shape", old_som)
                current_som = old_som | {
                    "som_direction_scope": "per layer",
                    "som_grid_shape": [2, 3],
                }
                restored = Settings.from_saved(current_som)
                self.assertEqual(restored.som_direction_scope, "per layer")
                self.assertEqual(restored.som_grid_shape, (2, 3))

    def test_cli_reports_root_and_field_validation_errors(self):
        from heretic.main import run

        cases = (
            (
                ["--som-direction-scope", "per layer", "--som-source-layer", "0"],
                "settings: Value error, som_source_layer is only used by global or auto SOM",
            ),
            (
                ["--som-grid-size", "1", "--som-directions", "4"],
                "settings: Value error, som_directions cannot exceed the SOM grid's neuron count",
            ),
            (
                ["--som-iterations", "0"],
                "som_iterations: Input should be greater than 0",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text("")
            for arguments, message in cases:
                with (
                    self.subTest(arguments=arguments),
                    patch(
                        "sys.argv",
                        [
                            "heretic",
                            "--model",
                            "unused",
                            "--modifier",
                            "som",
                            *arguments,
                        ],
                    ),
                    patch(
                        "heretic.config.TomlConfigSettingsSource",
                        side_effect=lambda cls, **kwargs: TomlConfigSettingsSource(
                            cls, toml_file=config_path
                        ),
                    ),
                    patch("heretic.main.Model") as model,
                    patch("sys.stdout", new_callable=io.StringIO) as output,
                ):
                    run()
                text = " ".join(output.getvalue().split())
                self.assertIn("Configuration contains 1 errors:", text)
                self.assertIn(message, text)
                self.assertIn("heretic --help", text)
                model.assert_not_called()

    def test_rejects_invalid_som_parameters(self):
        for options in (
            {"som_grid_size": 0},
            {"som_grid_size": 2, "som_directions": 5},
            {"som_iterations": 0},
            {"som_learning_rate": 0},
            {"som_sigma": -1},
            {"som_max_weight": float("inf")},
            {"som_direction_scope": "unknown"},
            {"som_direction_scope": "per layer", "som_source_layer": 0},
            {"som_grid_shape": [2, 0]},
            {"som_grid_shape": [2, 3], "som_directions": 7},
        ):
            with self.subTest(options=options), self.assertRaises(ValidationError):
                settings(modifier="som", **options)

    def test_saved_global_settings_keep_their_scope(self):
        with patch("sys.argv", ["heretic"]):
            self.assertEqual(settings(modifier="som").som_direction_scope, "auto")
            legacy = {"model": "unused", "modifier": "som", "som_source_layer": 0}
            self.assertEqual(Settings.from_saved(legacy).som_direction_scope, "global")
            self.assertNotIn("som_direction_scope", legacy)
            self.assertEqual(
                Settings.from_saved(
                    legacy | {"som_direction_scope": "auto"}
                ).som_direction_scope,
                "auto",
            )


@unittest.skipUnless(HAS_SOM, "Install the som extra")
class CandidateTests(unittest.TestCase):
    def test_ranks_by_samples_and_removes_duplicate_directions(self):
        class FakeSom:
            def random_weights_init(self, data):
                pass

            def train_random(self, data, iterations):
                pass

            def get_weights(self):
                return np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 2.0], [0.0, 0.0]]])

            def winner(self, x):
                if x[1] == 2:
                    return (1, 0)
                return (0, 1) if x[1] else (0, 0)

        config = settings(
            modifier="som", som_grid_size=2, orthogonalize_direction=False
        )
        samples = torch.tensor([[0.0, 1.0]] * 10 + [[0.0, 2.0]] * 4 + [[1.0, 0.0]])
        with patch("minisom.MiniSom", return_value=FakeSom()):
            directions, ids, counts = fit_directions(samples, torch.zeros(2), config, 0)
        self.assertEqual(ids, [1, 0])
        self.assertEqual(counts, [10, 1])
        torch.testing.assert_close(directions, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

    def test_real_som_seed_and_degenerate_inputs(self):
        config = settings(
            modifier="som",
            som_grid_size=2,
            som_iterations=40,
            orthogonalize_direction=False,
        )
        data = torch.randn(20, 8, generator=torch.Generator().manual_seed(1))
        a, ids, _ = fit_directions(data, torch.zeros(8), config, 5)
        b, other_ids, _ = fit_directions(data, torch.zeros(8), config, 5)
        self.assertEqual(ids, other_ids)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        zero, ids, _ = fit_directions(torch.zeros(8, 8), torch.zeros(8), config, 5)
        self.assertEqual(zero.shape, (0, 8))
        self.assertEqual(ids, [])

    def test_rectangular_grid_counts_and_ids(self):
        config = settings(modifier="som", som_grid_shape=(2, 3), som_directions=6)
        som = MagicMock()
        som.get_weights.return_value = np.eye(6).reshape(2, 3, 6)
        som.winner.side_effect = [(1, 2), (1, 2), (0, 1)]
        with patch("minisom.MiniSom", return_value=som) as constructor:
            directions, ids, counts = fit_directions(
                torch.eye(6)[:3], torch.zeros(6), config, 1
            )
        self.assertEqual(constructor.call_args.args[:2], (2, 3))
        self.assertEqual(ids, [5, 1])
        self.assertEqual(counts, [2, 1])
        torch.testing.assert_close(directions, torch.eye(6)[[5, 1]])

    def test_scopes_route_matching_layers_with_variable_candidate_counts(self):
        good = [Prompt(system="", user="good")]
        bad = [Prompt(system="", user="bad"), Prompt(system="", user="bad2")]
        good_residuals = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
        bad_residuals = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
        for scope, analysis in (
            ("global", False),
            ("per layer", False),
            ("auto", True),
        ):
            with self.subTest(scope=scope):
                config = settings(
                    modifier="som",
                    som_direction_scope=scope,
                    batch_size=2,
                    print_residual_geometry=analysis,
                    plot_residuals=analysis,
                    row_normalization="none",
                    seed=10,
                )
                model = MagicMock(spec=Model)
                model.get_layers.return_value = [None] * 3
                model.get_abliterable_components.return_value = ["attn.o_proj"]
                model.get_residuals_mean.return_value = good_residuals[0]
                model.get_residuals.return_value = bad_residuals
                model.get_residuals_batched.side_effect = [
                    good_residuals,
                    bad_residuals,
                ]
                banks = [torch.eye(4)[:2], torch.empty(0, 4), torch.eye(4)[2:3]]
                ids = [[3, 1], [], [0]]
                sources = [0, 1] if scope == "global" else [0, 1, 2]
                with (
                    patch(
                        "heretic.modifiers.som.fit_directions",
                        side_effect=[
                            (banks[i], ids[i], [1] * len(ids[i])) for i in sources
                        ],
                    ) as fit,
                    patch("heretic.modifiers.som.Analyzer") as analyzer,
                ):
                    modifier = SOM(config, model)
                    modifier.init(good, bad)
                for call, source in zip(fit.call_args_list, sources):
                    torch.testing.assert_close(
                        call.args[0], bad_residuals[:, source + 1]
                    )
                    torch.testing.assert_close(
                        call.args[1], good_residuals[0, source + 1]
                    )
                self.assertEqual(
                    analyzer.return_value.print_residual_geometry.call_count,
                    int(analysis),
                )
                self.assertEqual(
                    analyzer.return_value.plot_residuals.call_count, int(analysis)
                )
                model.apply_lora.assert_called_once_with(2)
                study = optuna.create_study(
                    sampler=optuna.samplers.RandomSampler(seed=3)
                )
                choices = ["global", "per layer"] if scope == "auto" else [scope]
                distributions = []
                for choice in choices:
                    if scope == "auto":
                        study.enqueue_trial({"som.direction_scope": choice})
                    trial = study.ask()
                    parameters = json.loads(
                        json.dumps(modifier.suggest_parameters(trial))
                    )
                    distributions.append(trial.distributions)
                    self.assertEqual(parameters["direction_scope"], choice)
                    modifier.modify_model(parameters)
                    actual_banks, actual_weights, _ = (
                        model.abliterate_multiple.call_args.args
                    )
                    if choice == "global":
                        self.assertEqual(parameters["source_layer"], 0)
                        torch.testing.assert_close(actual_banks, banks[0])
                        torch.testing.assert_close(
                            actual_weights.sum(), torch.tensor(1.0)
                        )
                        # Payloads written before scope support still restore.
                        parameters.pop("direction_scope")
                        modifier.modify_model(parameters)
                    else:
                        self.assertEqual(set(actual_banks), {0, 2})
                        torch.testing.assert_close(actual_banks[2], banks[2])
                        torch.testing.assert_close(
                            actual_weights[2], torch.tensor([1.0])
                        )
                        self.assertNotIn("source_layer", parameters)
                        parameters["layers"]["2"]["direction_weights"] = [float("nan")]
                        model.abliterate_multiple.reset_mock()
                        with self.assertRaisesRegex(ValueError, "mixture"):
                            modifier.modify_model(parameters)
                        model.abliterate_multiple.assert_not_called()
                if scope == "auto":
                    self.assertEqual(distributions[0], distributions[1])


@unittest.skipUnless(HAS_SOM, "Install the som extra")
class ModelLifecycleTests(unittest.TestCase):
    """Real local tiny transformer + PEFT, without Hub downloads or a GPU."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.base = cls.root / "base"
        vocab = {
            word: i
            for i, word in enumerate(
                [
                    "[PAD]",
                    "[UNK]",
                    "[EOS]",
                    "system",
                    "user",
                    "assistant",
                    "hello",
                    "world",
                    "blue",
                    "red",
                    "green",
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                ]
            )
        }
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()  # ty:ignore[invalid-assignment]
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="[PAD]",
            unk_token="[UNK]",
            eos_token="[EOS]",
        )
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
        torch.manual_seed(4)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(vocab),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                pad_token_id=0,
                eos_token_id=2,
                bos_token_id=None,
            )
        )
        model.save_pretrained(cls.base)
        tokenizer.save_pretrained(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_trial_reset_adapter_merge_and_restore(self):
        good = [
            Prompt(system="hello", user=x)
            for x in ["blue one", "green two", "blue three"]
        ]
        bad = [
            Prompt(system="hello", user=x)
            for x in ["red four", "red five", "world five", "world four"]
        ]
        for method, scope in (
            ("abliteration", "global"),
            ("som", "global"),
            ("som", "per layer"),
            ("som", "auto"),
        ):
            with self.subTest(method=method, scope=scope):
                config = settings(
                    model=str(self.base),
                    modifier=method,
                    dtypes=["float32"],
                    device_map="cpu",
                    batch_size=2,
                    seed=9,
                    row_normalization="none",
                    som_grid_size=2,
                    som_directions=2,
                    som_iterations=50,
                    som_direction_scope=scope,
                    som_source_layer=None if scope == "per layer" else 0,
                    orthogonalize_direction=False,
                )
                model = Model(config)
                modifier = create_modifier(config, model)
                baseline = model.get_logits_batched(good).detach().clone()
                modifier.init(good, bad)
                study = optuna.create_study(
                    sampler=optuna.samplers.RandomSampler(seed=2)
                )
                if scope == "auto":
                    study.enqueue_trial({"som.direction_scope": "global"})
                    study.enqueue_trial({"som.direction_scope": "per layer"})
                trial = study.ask()
                params = json.loads(json.dumps(modifier.suggest_parameters(trial)))
                if method == "abliteration":
                    legacy = optuna.trial.create_trial(
                        value=0, user_attrs={**params, "scores": []}
                    )
                    self.assertEqual(modifier.parameters_from_trial(legacy), params)
                    self.assertIn("direction_index", get_trial_parameters(legacy))
                    old_artifact = json.loads(
                        generate_reproduce_json(config, legacy, "test", {}, False)
                    )
                    self.assertEqual(old_artifact["version"], "3")
                trial.set_user_attr("modifier", method)
                trial.set_user_attr("modifier_parameters", params)
                trial.set_user_attr("scores", [])
                self.assertIn("modifier", get_trial_parameters(trial))
                artifact = json.loads(
                    generate_reproduce_json(config, trial, "test", {}, False)
                )
                self.assertEqual(artifact["version"], "4")
                self.assertEqual(artifact["parameters"]["parameters"], params)
                self.assertIn("minisom", artifact["environment"]["requirements"])
                modifier.modify_model(modifier.parameters_from_trial(trial))
                changed = model.get_logits_batched(good).detach().clone()
                self.assertGreater((changed - baseline).abs().max().item(), 1e-6)
                adapter_path = self.root / f"adapter-{method}-{scope}"
                model.model.save_pretrained(str(adapter_path))
                adapter_base = LlamaForCausalLM.from_pretrained(self.base)
                restored_adapter = PeftModel.from_pretrained(adapter_base, adapter_path)
                adapter_config = restored_adapter.peft_config["default"]
                assert isinstance(adapter_config, LoraConfig)
                self.assertEqual(adapter_config.r, model.peft_config.r)
                inputs = torch.tensor([[6, 8, 11]])
                model.model.eval()
                restored_adapter.eval()
                torch.testing.assert_close(
                    model.model(inputs).logits, restored_adapter(inputs).logits
                )
                modifier.reset_model()
                torch.testing.assert_close(model.get_logits_batched(good), baseline)
                modifier.modify_model(modifier.suggest_parameters(study.ask()))
                modifier.reset_model()
                modifier.modify_model(params)
                torch.testing.assert_close(model.get_logits_batched(good), changed)
                expected_rank = model.peft_config.r
                before_merge = model.model(inputs).logits.detach().clone()
                merged = model.get_merged_model()
                merged_path = self.root / f"merged-{method}-{scope}"
                merged.save_pretrained(merged_path)
                restored = LlamaForCausalLM.from_pretrained(merged_path)
                restored.eval()
                torch.testing.assert_close(
                    restored(inputs).logits, before_merge, rtol=1e-4, atol=1e-6
                )
                modifier.reset_model()
                self.assertEqual(model.peft_config.r, expected_rank)
                modifier.modify_model(params)
                torch.testing.assert_close(model.get_logits_batched(good), changed)

    def test_per_layer_adapter_updates_match_each_layers_own_directions(self):
        model = Model(
            settings(
                model=str(self.base),
                device_map="cpu",
                dtypes=["float32"],
                row_normalization="none",
            )
        )
        model.apply_lora(2)
        banks = {0: torch.eye(16)[:2], 1: torch.eye(16)[2:3]}
        mixtures = {0: torch.tensor([0.3, 0.7]), 1: torch.tensor([1.0])}
        components = {
            name: AbliterationParameters(
                max_weight=0.9,
                min_weight=0.4,
                max_weight_position=1.0,
                min_weight_distance=2.0,
            )
            for name in model.get_abliterable_components()
        }
        model.abliterate_multiple(banks, mixtures, components)
        for layer, directions in banks.items():
            strength = 0.65 if layer == 0 else 0.9
            for modules in model.get_layer_modules(layer).values():
                for module in modules:
                    assert isinstance(module, Linear)
                    W = cast(torch.Tensor, module.base_layer.weight)
                    expected = (
                        -strength * (directions.T * mixtures[layer]) @ (directions @ W)
                    )
                    actual = cast(torch.Tensor, module.lora_B["default"].weight) @ cast(
                        torch.Tensor, module.lora_A["default"].weight
                    )
                    torch.testing.assert_close(actual, expected)
        model.reset_model()
        # A layer with no usable candidates must remain unchanged after reset.
        model.abliterate_multiple({1: banks[1]}, {1: mixtures[1]}, components)
        for modules in model.get_layer_modules(0).values():
            for module in modules:
                assert isinstance(module, Linear)
                self.assertEqual(
                    torch.count_nonzero(
                        cast(torch.Tensor, module.lora_B["default"].weight)
                    ),
                    0,
                )
        with self.assertRaisesRegex(ValueError, "matching valid layers"):
            model.abliterate_multiple(banks, {0: mixtures[0]}, components)

    def test_cli_save_resume_and_standard_checkpoint_separation(self):
        run = self.root / "cli"
        run.mkdir()
        (run / "good.txt").write_text("blue one\ngreen two\nblue three\n")
        (run / "bad.txt").write_text("red four\nred five\nworld five\nworld four\n")
        config = {
            "model": str(self.base),
            "modifier": "som",
            "dtypes": ["float32"],
            "device_map": "cpu",
            "batch_size": 2,
            "seed": 9,
            "row_normalization": "full",
            "full_normalization_lora_rank": 4,
            "som_grid_size": 2,
            "som_directions": 2,
            "som_iterations": 50,
            "som_direction_scope": "global",
            "som_source_layer": 0,
            "orthogonalize_direction": False,
            "response_prefix": "",
            "max_response_length": 2,
            "n_trials": 2,
            "n_startup_trials": 1,
            "checkpoint_action": "continue",
            "trial_index": 0,
            "model_action": "save",
            "export_strategy": "merge",
            "save_directory": str(run / "output"),
            "good_prompts": {"dataset": str(run / "good.txt")},
            "bad_prompts": {"dataset": str(run / "bad.txt")},
            "scorers": [
                {
                    "plugin": "heretic.scorers.kl_divergence.KLDivergence",
                    "optimization": "minimize",
                }
            ],
            "scorer": {"KLDivergence": {"prompts": {"dataset": str(run / "good.txt")}}},
        }
        config_path = run / "config.toml"
        config_path.write_text(tomli_w.dumps(config))
        repo = Path(__file__).resolve().parents[1]
        env = os.environ | {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "2",
        }

        def invoke():
            result = subprocess.run(
                [sys.executable, "-c", "from heretic.main import main; main()"],
                cwd=run,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout

        # Use the installed entry point's callable; main.py has no module runner.
        env["PYTHONPATH"] = str(repo / "src")
        first = invoke()
        self.assertIn("Model saved", first)
        output = run / "output" / "model.safetensors"
        before = output.read_bytes()
        # Migrate a checkpoint from before som_direction_scope was introduced.
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend

        journal = next((run / "checkpoints").glob("*.jsonl"))
        storage = JournalStorage(JournalFileBackend(str(journal)))
        study = optuna.load_study(study_name=None, storage=storage)
        old_settings = json.loads(study.user_attrs["settings"])
        old_settings.pop("som_direction_scope")
        study.set_user_attr("settings", json.dumps(old_settings))
        second = invoke()
        self.assertIn("Resuming existing study", second)
        self.assertIn("(1/1)", second)
        self.assertEqual(before, output.read_bytes())
        config["modifier"] = "abliteration"
        config_path.write_text(tomli_w.dumps(config))
        self.assertIn("Model saved", invoke())
        journals = list((run / "checkpoints").glob("*.jsonl"))
        self.assertEqual(len(journals), 2)
        self.assertEqual(sum(p.stem.endswith("--som") for p in journals), 1)

        # Use an independent study to exercise per-layer CLI export and resume.
        config.update(
            modifier="som",
            som_direction_scope="per layer",
            study_checkpoint_dir=str(run / "per-layer-checkpoints"),
        )
        config.pop("som_source_layer")
        config_path.write_text(tomli_w.dumps(config))
        self.assertIn("Model saved", invoke())
        before = output.read_bytes()
        self.assertIn("Resuming existing study", invoke())
        self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
