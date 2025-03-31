import re
from typing import Dict, Type, Union

import torch
from torch import nn

from peft.tuners.lycoris_utils import LycorisTuner
from .layer import CompoundLayer, Linear, Conv2d

class CompoundModel(LycorisTuner):
    prefix: str = "compound"
    layers_mapping: Dict[Type[torch.nn.Module], Type[CompoundLayer]] = {
        torch.nn.Conv2d: Conv2d,
        torch.nn.Linear: Linear,
    }

    def _create_and_replace(
        self,
        config,
        adapter_name: str,
        target: Union[CompoundLayer, nn.Module],
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        """
        A private method to create and replace the target module with the adapter module.
        """

        kwargs = config.to_dict()
        if isinstance(target, CompoundLayer):
            print('is instance CompoundLayer'
            )
            target.update_layer(adapter_name, **kwargs)
        else:
            # print(f' adapter_name = {adapter_name} | parent = {parent} ')
            new_module = self._create_new_module(config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

