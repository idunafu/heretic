# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Philipp Emanuel Weidmann + contributors

from abc import ABC, abstractmethod
from typing import Any

from optuna import Trial
from optuna.trial import FrozenTrial

from .config import Settings
from .model import Model
from .utils import Prompt


class Modifier(ABC):
    """Small internal API for one active model modification method.

    Parameters must be JSON-serializable. Keep expensive preparation in init;
    modify_model is also called when restoring a selected or saved trial.
    LoRA-based methods can use the default reset, including after model export.
    """

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

    @abstractmethod
    def init(self, good_prompts: list[Prompt], bad_prompts: list[Prompt]) -> None: ...

    @abstractmethod
    def suggest_parameters(self, trial: Trial) -> dict[str, Any]: ...

    @abstractmethod
    def modify_model(self, parameters: dict[str, Any]) -> None: ...

    def reset_model(self) -> None:
        self.model.reset_model()

    def parameters_from_trial(self, trial: Trial | FrozenTrial) -> dict[str, Any]:
        name = trial.user_attrs.get("modifier", "abliteration")
        if name != self.settings.modifier:
            raise ValueError(
                f"Trial uses modifier {name!r}, not {self.settings.modifier!r}"
            )
        if "modifier_parameters" in trial.user_attrs:
            return trial.user_attrs["modifier_parameters"]
        # Existing version-3 checkpoints remain usable by the original method.
        if name != "abliteration":
            raise ValueError("Missing modifier parameters in trial")
        return {
            "direction_index": trial.user_attrs["direction_index"],
            "parameters": trial.user_attrs["parameters"],
        }


def create_modifier(settings: Settings, model: Model) -> Modifier:
    # Imports are lazy: the standard method does not require SOM dependencies.
    from .modifiers.abliteration import Abliteration
    from .modifiers.som import SOM

    registry: dict[str, type[Modifier]] = {
        "abliteration": Abliteration,
        "som": SOM,
    }
    try:
        cls = registry[settings.modifier]
    except KeyError:
        raise ValueError(
            f"Unknown modifier {settings.modifier!r}; choose from {', '.join(registry)}"
        ) from None
    return cls(settings, model)
