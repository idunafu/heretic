# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Philipp Emanuel Weidmann + contributors

from dataclasses import asdict

from optuna import Trial

from ..model import AbliterationParameters, Model


def suggest_components(
    model: Model, trial: Trial, max_strength: float = 1.5
) -> dict[str, dict[str, float]]:
    last_layer_index = len(model.get_layers()) - 1
    parameters = {}

    for component in model.get_abliterable_components():
        # The parameter ranges are based on experiments with various models
        # and much wider ranges. They are not set in stone and might have to be
        # adjusted for future models.
        #
        # The MLP gets a negative lower bound that is then clamped to 0, so the
        # optimizer can fully disable its ablation. The clamp puts a positive
        # probability mass on exactly 0 (the continuous sampler would otherwise
        # reach 0 with probability zero). Ablating the MLP is often unnecessary for
        # removing refusals and tends to damage model intelligence more than
        # ablating the attention output, so on many models the optimum is to leave
        # it (mostly) untouched. See issue #202.
        max_weight_lower_bound = -0.25 if component == "mlp.down_proj" else 0.8
        max_weight = max(
            0.0,
            trial.suggest_float(
                f"{component}.max_weight",
                max_weight_lower_bound,
                max_strength,
            ),
        )
        max_weight_position = trial.suggest_float(
            f"{component}.max_weight_position",
            0.6 * last_layer_index,
            1.0 * last_layer_index,
        )
        # For sampling purposes, min_weight is expressed as a fraction of max_weight,
        # again because multivariate TPE doesn't support variable-range parameters.
        # The value is transformed into the actual min_weight value below.
        min_weight = trial.suggest_float(
            f"{component}.min_weight",
            0.0,
            1.0,
        )
        min_weight_distance = trial.suggest_float(
            f"{component}.min_weight_distance",
            1.0,
            max(0.6 * last_layer_index, 1.0),
        )

        parameters[component] = AbliterationParameters(
            max_weight=max_weight,
            max_weight_position=max_weight_position,
            min_weight=(min_weight * max_weight),
            min_weight_distance=min_weight_distance,
        )

    return {name: asdict(value) for name, value in parameters.items()}


def decode_components(
    data: dict[str, dict[str, float]],
) -> dict[str, AbliterationParameters]:
    return {name: AbliterationParameters(**value) for name, value in data.items()}
