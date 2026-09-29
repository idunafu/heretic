# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Philipp Emanuel Weidmann + contributors

from typing import Any

import torch
import torch.nn.functional as F
from optuna import Trial

from ..analyzer import Analyzer
from ..modifier import Modifier
from ..system import empty_cache
from ..utils import Prompt, print
from .common import decode_components, suggest_components


class Abliteration(Modifier):
    """The original single-direction method, with its existing parameter space."""

    def init(self, good_prompts: list[Prompt], bad_prompts: list[Prompt]) -> None:
        print("Calculating per-layer residual directions...")

        needs_full_residuals = (
            self.settings.print_residual_geometry or self.settings.plot_residuals
        )

        if needs_full_residuals:
            print("* Obtaining residuals for good prompts...")
            good_residuals = self.model.get_residuals_batched(good_prompts)
            print("* Obtaining residuals for bad prompts...")
            bad_residuals = self.model.get_residuals_batched(bad_prompts)

            good_means = good_residuals.mean(dim=0)
            bad_means = bad_residuals.mean(dim=0)

            analyzer = Analyzer(
                self.settings, self.model, good_residuals, bad_residuals
            )

            if self.settings.print_residual_geometry:
                analyzer.print_residual_geometry()

            if self.settings.plot_residuals:
                analyzer.plot_residuals()

            # We don't need the full residuals after computing their means and analyzing geometry.
            del good_residuals, bad_residuals, analyzer
        else:
            print("* Obtaining residual mean for good prompts...")
            good_means = self.model.get_residuals_mean(good_prompts)
            print("* Obtaining residual mean for bad prompts...")
            bad_means = self.model.get_residuals_mean(bad_prompts)

        self.residual_directions = F.normalize(bad_means - good_means, p=2, dim=1)

        if self.settings.orthogonalize_direction:
            # Implements https://huggingface.co/blog/grimjim/projected-abliteration
            # Adjust the residual directions so that only the component that is
            # orthogonal to the good direction is subtracted during abliteration.
            good_directions = F.normalize(good_means, p=2, dim=1)
            projection_vector = torch.sum(
                self.residual_directions * good_directions, dim=1
            )
            self.residual_directions = (
                self.residual_directions
                - projection_vector.unsqueeze(1) * good_directions
            )
            self.residual_directions = F.normalize(self.residual_directions, p=2, dim=1)
            del good_directions, projection_vector

        del good_means, bad_means

        # Clear cache before starting the optimization study.
        # This should free up memory from the objects released with the del statements above.
        empty_cache()

    def suggest_parameters(self, trial: Trial) -> dict[str, Any]:
        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        last_layer_index = len(self.model.get_layers()) - 1

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )

        if direction_scope == "per layer":
            direction_index = None

        return {
            "direction_index": direction_index,
            "parameters": suggest_components(self.model, trial),
        }

    def modify_model(self, parameters: dict[str, Any]) -> None:
        self.model.abliterate(
            self.residual_directions,
            parameters["direction_index"],
            decode_components(parameters["parameters"]),
        )
