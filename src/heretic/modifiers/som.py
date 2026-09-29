# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Philipp Emanuel Weidmann + contributors

from typing import Any

import numpy as np
import torch
from optuna import Trial
from torch import Tensor

from ..analyzer import Analyzer
from ..config import RowNormalization, Settings
from ..modifier import Modifier
from ..system import empty_cache
from ..utils import Prompt, batchify, print
from .common import decode_components, suggest_components


def fit_directions(
    data: Tensor, good_mean: Tensor, settings: Settings, seed: int
) -> tuple[Tensor, list[int], list[int]]:
    """Fit a SOM and rank its distinct, nonzero directions by sample support."""
    try:
        from minisom import MiniSom
    except ImportError:
        raise ImportError(
            "SOM requires MiniSom. Install with `uv sync --extra som` "
            'or `pip install "heretic-llm[som]"`.'
        ) from None
    samples = data.detach().float().cpu().numpy()
    if len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError("SOM training samples must be nonempty and finite")
    size_x, size_y = settings.som_grid_shape or (
        settings.som_grid_size,
        settings.som_grid_size,
    )
    som = MiniSom(
        size_x,
        size_y,
        samples.shape[1],
        sigma=settings.som_sigma,
        learning_rate=settings.som_learning_rate,
        topology="hexagonal",
        activation_distance="euclidean",
        random_seed=seed,
    )
    som.random_weights_init(samples)
    som.train_random(samples, settings.som_iterations)
    # Count real training samples, not votes from each prototype for itself.
    counts = np.zeros(size_x * size_y, dtype=np.int64)
    for sample in samples:
        x, y = som.winner(sample)
        counts[x * size_y + y] += 1
    neurons = som.get_weights().reshape(size_x * size_y, -1)
    good = good_mean.detach().double().cpu().numpy()
    if not np.isfinite(good).all():
        raise ValueError("Good residual mean must be finite")
    good_norm = np.linalg.norm(good)
    good_unit = good / good_norm if good_norm > 1e-12 else np.zeros_like(good)
    selected: list[np.ndarray] = []
    ids: list[int] = []
    support: list[int] = []
    for index in np.argsort(-counts, kind="stable"):
        if counts[index] == 0:
            continue
        direction = neurons[index] - good
        if settings.orthogonalize_direction:
            direction -= np.dot(direction, good_unit) * good_unit
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm < 1e-8:
            continue
        direction /= norm
        if any(abs(np.dot(direction, v)) > 1 - 1e-6 for v in selected):
            continue
        selected.append(direction)
        ids.append(int(index))
        support.append(int(counts[index]))
        if len(selected) == settings.som_directions:
            break
    if not selected:
        return torch.empty(0, samples.shape[1]), ids, support
    return torch.tensor(np.stack(selected), dtype=torch.float32), ids, support


class SOM(Modifier):
    """SOM-derived global or per-layer directions with bounded strength envelopes.

    Candidate training is performed once, on CPU. No interpolation is performed
    between unrelated neurons learned at different source layers.
    """

    def init(self, good_prompts: list[Prompt], bad_prompts: list[Prompt]) -> None:
        # Fail early, before collecting potentially large residual tensors.
        try:
            import minisom  # noqa: F401
        except ImportError:
            raise ImportError(
                "Install SOM support with `uv sync --extra som`."
            ) from None
        if not good_prompts or not bad_prompts:
            raise ValueError("SOM requires both good and bad prompts")
        last = len(self.model.get_layers()) - 1
        if self.settings.som_source_layer is not None:
            if self.settings.som_source_layer > last:
                raise ValueError(f"som_source_layer must be between 0 and {last}")
            global_layers = [self.settings.som_source_layer]
        else:
            global_layers = list(range(int(0.4 * last), int(0.9 * last) + 1))
        layers = (
            global_layers
            if self.settings.som_direction_scope == "global"
            else list(range(last + 1))
        )
        slots = [layer + 1 for layer in layers]  # Exclude the embedding slot.
        needs_analysis = (
            self.settings.print_residual_geometry or self.settings.plot_residuals
        )
        if needs_analysis:
            print("* Collecting full residuals for SOM and residual analysis...")
            good_residuals = self.model.get_residuals_batched(good_prompts).cpu()
            bad_residuals = self.model.get_residuals_batched(bad_prompts).cpu()
            analyzer = Analyzer(
                self.settings, self.model, good_residuals, bad_residuals
            )
            if self.settings.print_residual_geometry:
                analyzer.print_residual_geometry()
            if self.settings.plot_residuals:
                analyzer.plot_residuals()
            good_means = good_residuals.mean(dim=0, dtype=torch.float64).float()
            bad = bad_residuals[:, slots, :]
            del analyzer, good_residuals, bad_residuals
        else:
            print("* Obtaining good residual means for SOM...")
            good_means = self.model.get_residuals_mean(good_prompts).cpu()
            print("* Collecting bad residuals for SOM source layers...")
            chunks = []
            # Keep only the source layers and do not retain the full good dataset.
            for batch in batchify(bad_prompts, self.settings.batch_size):
                residuals = self.model.get_residuals(batch)
                chunks.append(residuals[:, slots, :].cpu())
            del residuals
            bad = torch.cat(chunks)
            del chunks
        self.banks: dict[int, Tensor] = {}
        self.neuron_ids: dict[int, list[int]] = {}
        for position, layer in enumerate(layers):
            print(
                f"* Training SOM at source layer {layer} ({position + 1}/{len(layers)})..."
            )
            directions, ids, counts = fit_directions(
                bad[:, position, :],
                good_means[layer + 1],
                self.settings,
                ((self.settings.seed or 0) + layer) % 2**32,
            )
            if len(ids) == 0:
                print("  * No usable directions; skipping this source layer")
                continue
            self.banks[layer] = directions
            self.neuron_ids[layer] = ids
            print(f"  * {len(ids)} directions, sample support: {counts}")
        if not self.banks:
            raise ValueError(
                "SOM found no nonzero directions; check the contrastive datasets"
            )
        self.global_layers = [layer for layer in global_layers if layer in self.banks]
        if not self.global_layers and self.settings.som_direction_scope == "global":
            raise ValueError("No usable SOM directions in the global source layers")
        if not self.global_layers and self.settings.som_direction_scope == "auto":
            print("* No usable global source layers; using per-layer trials only")
        self.direction_count = max(len(bank) for bank in self.banks.values())
        rank = self.direction_count
        if self.settings.row_normalization == RowNormalization.FULL:
            rank = max(rank, self.settings.full_normalization_lora_rank)
        self.model.apply_lora(rank)
        del bad, good_means
        empty_cache()

    def suggest_parameters(self, trial: Trial) -> dict[str, Any]:
        scope = self.settings.som_direction_scope
        if scope == "auto":
            scopes = ["global", "per layer"] if self.global_layers else ["per layer"]
            scope = trial.suggest_categorical("som.direction_scope", scopes)
        # Sample the source even in per-layer trials under auto, keeping the
        # parameter space fixed for multivariate TPE. It is unused in their payload.
        layer = None
        if self.settings.som_direction_scope != "per layer" and self.global_layers:
            layer = trial.suggest_categorical("som.source_layer", self.global_layers)
        # k-1 logits avoid a redundant common offset. Per-layer mixtures share
        # support-rank preferences, not neuron identities or interpolated vectors.
        logits = [
            trial.suggest_float(f"som.mix.{i}", -3.0, 3.0)
            for i in range(self.direction_count - 1)
        ] + [0.0]

        def bank_parameters(source: int) -> dict[str, Any]:
            count = len(self.banks[source])
            # Renormalize the same rank preferences over the available directions;
            # missing directions never receive weight or duplicate another vector.
            values = torch.tensor(logits[:count], dtype=torch.float64)
            return {
                "neuron_ids": self.neuron_ids[source],
                "direction_weights": torch.softmax(values, dim=0).tolist(),
            }

        parameters: dict[str, Any] = {
            "direction_scope": scope,
            "parameters": suggest_components(
                self.model, trial, self.settings.som_max_weight
            ),
        }
        if scope == "global":
            assert layer is not None
            parameters.update(source_layer=layer, **bank_parameters(layer))
        else:
            parameters["layers"] = {
                str(source): bank_parameters(source) for source in sorted(self.banks)
            }
        return parameters

    def _validated_weights(self, layer: int, parameters: dict[str, Any]) -> Tensor:
        if (
            layer not in self.banks
            or parameters["neuron_ids"] != self.neuron_ids[layer]
        ):
            raise ValueError(
                "SOM candidates changed; start a new study with these settings"
            )
        weights = torch.tensor(parameters["direction_weights"], dtype=torch.float32)
        if (
            weights.shape != (len(self.banks[layer]),)
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or not torch.isclose(weights.sum(), torch.tensor(1.0))
        ):
            raise ValueError("Invalid SOM mixture weights")
        return weights

    def modify_model(self, parameters: dict[str, Any]) -> None:
        # Original SOM checkpoints contain only a global bank and no scope key.
        scope = parameters.get("direction_scope", "global")
        components = decode_components(parameters["parameters"])
        if scope == "global":
            layer = parameters["source_layer"]
            weights = self._validated_weights(layer, parameters)
            self.model.abliterate_multiple(self.banks[layer], weights, components)
        elif scope == "per layer":
            layers = parameters["layers"]
            if set(layers) != {str(layer) for layer in self.banks}:
                raise ValueError("SOM candidate layers changed; start a new study")
            # Validate all banks before applying any updates to the model.
            weights_by_layer = {
                layer: self._validated_weights(layer, layers[str(layer)])
                for layer in self.banks
            }
            self.model.abliterate_multiple(self.banks, weights_by_layer, components)
        else:
            raise ValueError(f"Unknown SOM direction scope: {scope!r}")
