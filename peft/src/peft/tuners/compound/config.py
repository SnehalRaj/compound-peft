from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class CompoundConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`CompoundModel`].

    Args:
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to.
        init_weights (`bool`):
            Whether to initialize weights orthogonally.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
    """
    r: int = field(default=8, metadata={"help": "Compound rank"})
    module_dropout: float = field(default=0.0, metadata={"help": "Dropout probability for compound modules"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to replace with Compound Adapters."},
    )
    init_weights: bool = field(default=True, metadata={"help": "Whether to initialize weights orthogonally"})
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={"help": "The layer indices to transform"},
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={"help": "The layer pattern name"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of modules apart from compound adapter layers to be set as trainable and saved"},
    )
    compound_pattern: Optional[List[str]] = field(default=None, metadata={"help": "Compound orders to be use"})
    compound_type: Optional[str] = field(default='comp', metadata={"help": "Type of operation to be performed: comp/max/avg"})
    block_share: bool = field(
        default=False,
        metadata={"help": "Whether to share the compound parameters between blocks or not."},
    )
    use_orthogonal: bool = field(
        default=True,
        metadata={"help": "Whether to initialise and keep the base matrix Orthogonal."},
    )
    
    num_adapters: int = field(
        default=True,
        metadata={"help": "Number of adapters to multiply together."},
    )
    
    adapter_multiplicative: bool = field(
        default = True,
        metadata = { "help": "Whether to multiply adapters or add them before combining with pretrained weights."},
    )




    def __post_init__(self):
        self.peft_type = PeftType.COMPOUND
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )  