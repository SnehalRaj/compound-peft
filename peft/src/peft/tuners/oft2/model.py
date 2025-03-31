# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, soft2ware
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict, Type, Union

import torch
from torch import nn

from peft.tuners.lycoris_utils import LycorisConfig, LycorisTuner

from .layer import Conv2d, Linear, OFT2Layer


class OFT2Model(LycorisTuner):
    """
    Creates Orthogonal Finetuning model from a pretrained model. The method is described in
    https://arxiv.org/abs/2306.07280

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`OFT2Config`]): The configuration of the OFT2 model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The OFT2 model.

    Example:
        ```py
        >>> from diffusers import StableDiffusionPipeline
        >>> from peft import OFT2Model, OFT2Config

        >>> config_te = OFT2Config(
        ...     r=8,
        ...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ... )
        >>> config_unet = OFT2Config(
        ...     r=8,
        ...     target_modules=[
        ...         "proj_in",
        ...         "proj_out",
        ...         "to_k",
        ...         "to_q",
        ...         "to_v",
        ...         "to_out.0",
        ...         "ff.net.0.proj",
        ...         "ff.net.2",
        ...     ],
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ... )

        >>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> model.text_encoder = OFT2Model(model.text_encoder, config_te, "default")
        >>> model.unet = OFT2Model(model.unet, config_unet, "default")
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`OFT2Config`]): The configuration of the OFT2 model.
    """

    prefix: str = "oft2_"
    layers_mapping: Dict[Type[torch.nn.Module], Type[OFT2Layer]] = {
        torch.nn.Conv2d: Conv2d,
        torch.nn.Linear: Linear,
    }

    def _create_and_replace(
        self,
        config: LycorisConfig,
        adapter_name: str,
        target: Union[OFT2Layer, nn.Module],
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        """
        A private method to create and replace the target module with the adapter module.
        """

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(config.rank_pattern.keys())
        target_name_key = next(filter(lambda key: re.match(rf"(.*\.)?{key}$", current_key), pattern_keys), target_name)

        kwargs = config.to_dict()
        kwargs["r"] = config.rank_pattern.get(target_name_key, config.r)

        if isinstance(target, OFT2Layer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)
