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

import math
import warnings
from typing import Any, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from peft.tuners.lycoris_utils import LycorisLayer, check_adapters_to_merge


class OFT2Layer(nn.Module, LycorisLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("oft2_r",)
    # other_param_names is defined on parent class

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        LycorisLayer.__init__(self, base_layer)

        # OFT2 info
        self.oft2_r = nn.ParameterDict({})
        self.coft2 = {}
        self.eps = {}
        self.block_share = {}

    @property
    def _available_adapters(self) -> Set[str]:
        return {*self.oft2_r}

    # def create_adapter_parameters(self, adapter_name: str, r: int, shape: Tuple[int, ...], block_share: bool):
    #     if block_share:
    #         self.oft2_r[adapter_name] = nn.Parameter(torch.empty(1, math.ceil(shape[0] / r), math.ceil(shape[0] / r)))
    #     else:
    #         self.oft2_r[adapter_name] = nn.Parameter(torch.empty(r, math.ceil(shape[0] / r), math.ceil(shape[0] / r)))
    @staticmethod
    def find_closest_binomial(target):
        n = 2
        prev_binom = 1
        while True:
            binom_value = math.comb(n, 1)
            if binom_value > target:
                    return n - 1, prev_binom
            prev_binom = binom_value
            n += 1
    def create_adapter_parameters(self, adapter_name: str, r: int, shape: Tuple[int, ...], block_share: bool):
        self.shape = shape
        self.compound_n, self.binom_value = self.find_closest_binomial(int(shape[0]/r))
        
        if block_share:
            self.oft2_r[adapter_name] = nn.Parameter(torch.empty(1, self.compound_n, self.compound_n))
        else:
            self.oft2_r[adapter_name] = nn.Parameter(torch.empty(r, self.compound_n, self.compound_n))


    def reset_adapter_parameters(self, adapter_name: str):
        nn.init.zeros_(self.oft2_r[adapter_name])

    def reset_adapter_parameters_random(self, adapter_name: str):
        nn.init.kaiming_uniform_(self.oft2_r[adapter_name], a=math.sqrt(5))

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        module_dropout: float,
        init_weights: bool,
        coft2: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        **kwargs,
    ) -> None:
        """Internal function to create oft2 adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize weights.
            coft2 (`bool`): Whether to use the constrained variant of OFT2 or not.
            eps (`float`):
                The control strength of COFT2. The freedom of rotation. Only has an effect if `coft2` is set to True.
            block_share (`bool`): Whether to share the OFT2 parameters between blocks or not.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.module_dropout[adapter_name] = module_dropout
        self.coft2[adapter_name] = coft2
        self.block_share[adapter_name] = block_share

        # Determine shape of OFT2 weights
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            shape = tuple(base_layer.weight.shape)
        elif isinstance(base_layer, nn.Conv2d):
            shape = (
                base_layer.out_channels,
                base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
            )
        else:
            raise TypeError(f"OFT2 is not implemented for base layers of type {type(base_layer).__name__}")

        self.eps[adapter_name] = eps * math.ceil(shape[0] / r) * math.ceil(shape[0] / r)

        # Create weights with provided shape
        self.create_adapter_parameters(adapter_name, r, shape, block_share)

        # Initialize weights
        if init_weights:
            self.reset_adapter_parameters(adapter_name)
        else:
            self.reset_adapter_parameters_random(adapter_name)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def unscale_layer(self, scale=None) -> None:
        # scale is not used
        pass

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()

                orig_weights = base_layer.weight.data
                if isinstance(base_layer, nn.Linear):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    orig_weights = orig_weights.view(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
                        ]
                    )
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                delta_weight = self.get_delta_weight(active_adapter)
                if orig_weights.shape[1] != delta_weight.shape[1]:
                    # when in channels is not divisible by r
                    delta_weight = delta_weight[: orig_weights.shape[1], : orig_weights.shape[1]]
                new_weights = torch.mm(orig_weights, delta_weight)
                if isinstance(base_layer, nn.Linear):
                    new_weights = torch.transpose(new_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    new_weights = torch.transpose(new_weights, 0, 1)
                    new_weights = new_weights.view(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels,
                            base_layer.kernel_size[0],
                            base_layer.kernel_size[1],
                        ]
                    )

                if safe_merge and not torch.isfinite(new_weights).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                base_layer.weight.data = new_weights.contiguous()
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()
                new_weights = base_layer.weight.data
                if isinstance(base_layer, nn.Linear):
                    new_weights = torch.transpose(new_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    new_weights = new_weights.view(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
                        ]
                    )
                    new_weights = torch.transpose(new_weights, 0, 1)
                delta_weight = self.get_delta_weight(active_adapter)
                if new_weights.shape[1] != delta_weight.shape[1]:
                    # when in channels is not divisible by r
                    delta_weight = delta_weight[: new_weights.shape[1], : new_weights.shape[1]]
                delta_inv = torch.inverse(delta_weight)
                orig_weights = torch.mm(new_weights, delta_inv)

                if isinstance(base_layer, nn.Linear):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights.reshape(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels,
                            base_layer.kernel_size[0],
                            base_layer.kernel_size[1],
                        ]
                    )
                base_layer.weight.data = orig_weights.contiguous()

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        rank = self.r[adapter_name]
        coft2 = self.coft2[adapter_name]
        eps = self.eps[adapter_name]
        opt_r = self.oft2_r[adapter_name]

        if coft2:
            with torch.no_grad():
                opt_r.copy_(self._project_batch(opt_r, eps=eps))

        orth_rotate = self._cayley_batch(opt_r)
        weight = self._block_diagonal(orth_rotate, rank)

        return weight

    # Copied from https://github.com/Zeju1997/oft2/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft2-control/oft2.py#L144
    def _cayley_batch(self, data: torch.Tensor) -> torch.Tensor:
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)  # noqa: E741

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    # Copied from https://github.com/Zeju1997/oft2/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft2-control/oft2.py#L155
    # def _block_diagonal(self, oft2_r: torch.Tensor, rank: int) -> torch.Tensor:
    #     if oft2_r.shape[0] == 1:
    #         # block share
    #         blocks = [oft2_r[0, ...] for i in range(rank)]
    #     else:
    #         blocks = [oft2_r[i, ...] for i in range(rank)]

    #     # Use torch.block_diag to create the block diagonal matrix
    #     A = torch.block_diag(*blocks)

    #     return A
    def _compute_circuit_compound_(self, unary, order: int = 1):
        num_qubits = unary.shape[-1]
        # unary = unary.flip(dims=(0, 1))
        if (order < 0) or (order > num_qubits):
            raise ValueError("Invalid order.")
        elif (order == 0) or (order == num_qubits):
            compound = torch.ones((1, 1))
        elif order == 1:
            compound = unary
        else:
            subsets = list(itertools.combinations(range(num_qubits), order))
            subsets_tensor = torch.tensor(
                subsets, dtype=torch.long, device=unary.device
            )
            submatrices = (
                unary[subsets_tensor][:, :, subsets_tensor]
                .transpose(1, 2)
                .transpose(2, 3)
            )
            compound = torch.linalg.det(submatrices)
        return compound
    def _block_diagonal(self, oft2_r: torch.Tensor, rank: int) -> torch.Tensor:
        blocks = [] 
        for i in range( oft2_r.shape[0]):
            block = self._compute_circuit_compound_(oft2_r[i,...], 1)
            # block = orth_rotate[i,...]
            block_size = int(self.shape[0] / rank)
            if block_size != block.shape[0]: 
                c = block_size
                r = block.shape[0]
                pad_size = c - r
                padded_block = torch.zeros(c, c, device=orth_rotate.device)
                padded_block[:r, :r] = block
                padded_block[r:, r:] = torch.eye(pad_size, device=orth_rotate.device)
                block = padded_block
            blocks.append(block)
        A = torch.block_diag(*blocks).to(orth_rotate.device)
        return A

    # Copied from https://github.com/Zeju1997/oft2/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft2-control/oft2.py#L52
    def _project_batch(self, oft2_r, eps=1e-5):
        # scaling factor for each of the smaller block matrix
        eps = eps * 1 / torch.sqrt(torch.tensor(oft2_r.shape[0]))
        I = (  # noqa: E741
            torch.zeros((oft2_r.size(1), oft2_r.size(1)), device=oft2_r.device, dtype=oft2_r.dtype)
            .unsqueeze(0)
            .expand_as(oft2_r)
        )
        diff = oft2_r - I
        norm_diff = torch.norm(oft2_r - I, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= eps).bool()
        out = torch.where(mask, oft2_r, I + eps * (diff / norm_diff))
        return out

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            if len(result.shape) == 4:
                result = result.permute(0, 2, 3, 1)

            base_layer = self.get_base_layer()
            base_bias = base_layer.bias
            if base_bias is not None:
                # Bias should be added after OFT2 forward
                result = result - base_bias.data

            # Execute all the adapters
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (self.training and torch.rand(1) > module_dropout):
                    result = self._get_delta_activations(active_adapter, result, *args, **kwargs)

            if base_bias is not None:
                result = result + base_bias.data
            if len(result.shape) == 4:
                result = result.permute(0, 3, 1, 2)

        result = result.to(previous_dtype)
        return result


class Linear(OFT2Layer):
    """OFT2 implemented in Linear layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, module_dropout, init_weights, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)

        base_layer = self.get_base_layer()
        base_weight = base_layer.weight.data
        delta_weight = delta_weight[: base_weight.shape[0], : base_weight.shape[0]]

        # don't add bias here, because the bias will be added after OFT2 forward
        return torch.matmul(input, delta_weight)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft2." + rep


class Conv2d(OFT2Layer):
    """OFT2 implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, module_dropout, init_weights, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)

        base_layer = self.get_base_layer()
        base_weight = base_layer.weight.data
        delta_weight = delta_weight[: base_weight.shape[0], : base_weight.shape[0]]

        # don't add bias here, because the bias will be added after OFT2 forward
        return torch.matmul(input, delta_weight)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft2." + rep
