# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.tuners.lycoris_utils import LycorisConfig
from peft.utils import PeftType


@dataclass
class OFT2Config(LycorisConfig):
    """
    This is the configuration class to store the configuration of a [`OFT2Model`].

    Args:
        r (`int`): OFT2 rank.
        module_dropout (`int`): The dropout probability for disabling OFT2 modules during training.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear modules are chosen, excluding
            the output layer. If this is not specified, modules will be chosen according to the model architecture. If
            the architecture is not known, an error will be raised -- in this case, you should specify the target
            modules manually.
        init_weights (`bool`):
            Whether to perform initialization of OFT2 weights.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        coft (`bool`):
            Whether to use the constrained variant of OFT2 or not, off by default.
        eps (`float`):
            The control strength of COFT2. The freedom of rotation. Only has an effect if `coft` is set to True.
        block_share (`bool`):
            Whether to share the OFT2 parameters between blocks or not. This is `False` by default.
    """

    r: int = field(default=8, metadata={"help": "OFT2 rank"})
    module_dropout: float = field(
        default=0.0, metadata={"help": "The dropout probability for disabling OFT2 modules during training"}
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with OFT2."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
            "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the OFT2 layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from OFT2 layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    coft: bool = field(
        default=False,
        metadata={"help": "Whether to use the constrained variant of OFT2 or not."},
    )
    eps: float = field(
        default=6e-5,
        metadata={
            "help": "The control strength of COFT2. The freedom of rotation. Only has an effect if `coft` is set to True."
        },
    )
    block_share: bool = field(
        default=False,
        metadata={"help": "Whether to share the OFT2 parameters between blocks or not."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.OFT2
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
