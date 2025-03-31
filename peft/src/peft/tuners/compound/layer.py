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

import math
import warnings
from typing import Any, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import numpy as np
import itertools
from peft.tuners.lycoris_utils import LycorisLayer, check_adapters_to_merge

class CompoundLayer(nn.Module, LycorisLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("compound",)
    # other_param_names is defined on parent class

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        LycorisLayer.__init__(self, base_layer)

        # compound info
        self.compound = nn.ParameterDict({})
        self.block_share = {}
        self.use_orthogonal = {}
        self.compound_pattern = {}
        self.compound_type = {}
        
        self.num_adapters = {}
        self.adapter_multiplicative = {}

    @property
    def _available_adapters(self) -> Set[str]:
        return {*self.compound}
    

    def SON(self, x, n):
        # Create a complex tensor for X
        X = torch.zeros((n, n), dtype=torch.complex64, device=x.device)
        triu_indices = torch.triu_indices(n, n, offset=1)
        
        # Convert x to complex if it's not already
        x_complex = x.to(torch.complex64) if x.dtype != torch.complex64 else x
        
        # Assign values to the upper triangular part
        X[triu_indices[0], triu_indices[1]] = x_complex
        X = X - X.conj().T  # Use conj().T for complex conjugate transpose
        
        # Compute matrix exponential
        V = torch.matrix_exp(X).real.to(torch.float32)
        
        return V

    @staticmethod
    def find_closest_binomial(target,compound_pattern):
        binom_map = {
        'comp_1': lambda x: x,
        'comp_2': lambda x: math.comb(x, 2),
        'comp_3': lambda x: math.comb(x, 3)
        }
        n = 2
        prev_binom = 1
        while True:
            binom_value = sum(binom_map[pattern](n) for pattern in compound_pattern if pattern in binom_map)
            if binom_value > target:
                    return n - 1, prev_binom
            prev_binom = binom_value
            n += 1

    def create_adapter_parameters(self, adapter_name: str, r:int,  shape: Tuple[int, ...], compound_pattern:Optional[List[str]], block_share:bool, num_adapters:int):
        self.shape = shape
        self.compound_n, self.compound_k = self.find_closest_binomial(target=int(shape[0]/r), compound_pattern=compound_pattern)
        if block_share:
            self.compound[adapter_name] = nn.Parameter(torch.empty(num_adapters, 1,  self.compound_n, self.compound_n ))
        else:
            self.compound[adapter_name] = nn.Parameter(torch.empty(num_adapters, r, self.compound_n, self.compound_n ))
    
    def reset_adapter_parameters(self, adapter_name: str):
        nn.init.zeros_(self.compound[adapter_name])

    # def reset_adapter_parameters_random(self, adapter_name: str):
    #     nn.init.kaiming_uniform_(self.compound[adapter_name], a=math.sqrt(5))
    def reset_adapter_parameters_random(self, adapter_name: str):
        nn.init.xavier_uniform_(self.compound[adapter_name], gain=nn.init.calculate_gain('linear'))


    def update_layer(
        self,
        r:int,
        adapter_name: str,
        module_dropout: float,
        init_weights: bool,
        compound_pattern: Optional[List[str]],
        compound_type: Optional[str],
        block_share: bool = False,
        use_orthogonal: bool = True,
        num_adapters: int = 1,
        adapter_multiplicative: bool = True,
        **kwargs,
    ) -> None:
        """Internal function to create compound adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize weights.
        """
        self.module_dropout[adapter_name] = module_dropout
        self.block_share[adapter_name] = block_share
        self.use_orthogonal[adapter_name] = use_orthogonal
        self.compound_pattern[adapter_name] = compound_pattern
        self.compound_type[adapter_name] = compound_type
        self.num_adapters[adapter_name] = num_adapters
        self.adapter_multiplicative[adapter_name] = adapter_multiplicative
        self.r = r
        # Determine shape of compound weights
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            shape = tuple(base_layer.weight.shape)
        elif isinstance(base_layer, nn.Conv2d):
            shape = (
                base_layer.out_channels,
                base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
            )
        else:
            raise TypeError(f"Compound is not implemented for base layers of type {type(base_layer).__name__}")


        # Create weights with provided shape
        self.create_adapter_parameters(adapter_name, r,  shape,compound_pattern, block_share, num_adapters)

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
                if orig_weights.shape[-1] != delta_weight.shape[0]:  # Compare c with r
                    c = orig_weights.shape[-1]
                    r = delta_weight.shape[0]
                    pad_size = c - r
                    padded_delta_weight = torch.zeros(c, c, device=delta_weight.device)
                    padded_delta_weight[:r, :r] = delta_weight
                    padded_delta_weight[r:, r:] = torch.eye(pad_size, device=delta_weight.device)
                    delta_weight = padded_delta_weight
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
                if new_weights.shape[-1] != delta_weight.shape[0]:  # Compare c with r
                    c = new_weights.shape[-1]
                    r = delta_weight.shape[0]
                    pad_size = c - r
                    padded_delta_weight = torch.zeros(c, c, device=delta_weight.device)
                    padded_delta_weight[:r, :r] = delta_weight
                    padded_delta_weight[r:, r:] = torch.eye(pad_size, device=delta_weight.device)
                    delta_weight = padded_delta_weight
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
        compound = self.compound[adapter_name]
        block_share = self.block_share[adapter_name]
        use_orthogonal = self.use_orthogonal[adapter_name]
        compound_pattern = self.compound_pattern[adapter_name]
        compound_type = self.compound_type[adapter_name]
        num_adapters = self.num_adapters[adapter_name]
        adapter_multiplicative = self.adapter_multiplicative[adapter_name]

        weight = None
        
        for i in range(num_adapters):
            if block_share:
                comp = compound[i] # [1, compound_n, compound_n]
            else:
                comp = compound[i] # [r, compound_n, compound_n]
                
            if use_orthogonal:
                orth_rotate = self._cayley_batch(comp)
            else:
                orth_rotate = comp
                
            weight_i = self._block_diagonal(
                orth_rotate, compound_pattern, compound_type, block_share
            )
            
            if weight is None:
                weight = weight_i
            else:
                if adapter_multiplicative:
                    weight = torch.matmul(weight, weight_i)
                else:
                    weight = weight + weight_i
        return weight
                
                
                
        # if use_orthogonal:
        #     orth_rotate = self._cayley_batch(compound)
        # else:
        #     orth_rotate = compound
        # weight = self._block_diagonal(orth_rotate, compound_pattern,compound_type, block_share)
        # return weight

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L144
    def _cayley_batch(self, data: torch.Tensor) -> torch.Tensor:
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)  # noqa: E741

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q
                                                       
    def _compute_circuit_compound_(self, unary, order: int = 1, compound_type: str= 'comp'):
        def permanent(matrix):
            n = matrix.size(0)
            total = 0
            for perm in itertools.permutations(range(n)):
                product = torch.tensor(1.0, device=matrix.device)
                for i, j in enumerate(perm):
                    product *= matrix[i, j]
                total += product
            return total

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
            if compound_type == 'comp':
                compound = torch.linalg.det(submatrices)
            elif compound_type == 'max':
                compound = torch.amax(submatrices, dim=(-2, -1))
            elif compound_type == 'avg':
                compound = torch.mean(submatrices, dim=(-2, -1))
            elif compound_type == 'perm':
                n=num_qubits
                compound = torch.empty(n, n, device=submatrices.device)
                for i in range(n):
                    for j in range(n):
                        compound[i, j] = permanent(submatrices[i, j])
            else:
                raise ValueError(f"Unsupported compound_type: {compound_type} | Please choose one of comp/max/avg/perm")

        return compound

    def _reconstruct_matrix(self, unary):
        n = 2**(unary.shape[0])
        size = unary.shape[0]
        reconstructed = torch.zeros((n, n), device=unary.device)
        start_index = 0
        for order in range(0, size + 1):
            compound = self._compute_circuit_compound_(unary, order)
            end_index = start_index + compound.shape[0]
            reconstructed[start_index:end_index, start_index:end_index] = compound
            start_index = end_index
        return reconstructed                                              

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L155
    def _block_diagonal(self, orth_rotate: torch.Tensor, compound_pattern: Optional[List[str]],compound_type: Optional[str], block_share: bool) -> torch.Tensor:
        def get_comp(tensor, k):
            return self._compute_circuit_compound_(tensor, k, compound_type)
    
        comp_map = {
            'comp_1': lambda tensor: tensor,
            'comp_2': lambda tensor: get_comp(tensor, 2),
            'comp_3': lambda tensor: get_comp(tensor, 3)
        }
    
    
        # Use torch.block_diag to create the block diagonal matrix
        blocks = []
        for i in range(self.r):
            sub_blocks = []
            for pattern in compound_pattern:
                if pattern in comp_map:
                    if orth_rotate.shape[0] == 1:
                        sub_blocks.append(comp_map[pattern](orth_rotate[0,...]))
                    else:
                        sub_blocks.append(comp_map[pattern](orth_rotate[i,...]))
            block = torch.block_diag(*sub_blocks)
            block_size = int(self.shape[0] / self.r)
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
                # Bias should be added after Compound forward
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




class Linear(CompoundLayer):
    """Compound Adapter implemented in Linear layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        r: int,
        adapter_name: str = "default",
        module_dropout: float = 0.0,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(r,adapter_name, module_dropout,init_weights, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)
        if input.shape[-1] != delta_weight.shape[0]:  # Compare c with r
            c = input.shape[-1]
            r = delta_weight.shape[0]
            pad_size = c - r
            padded_delta_weight = torch.zeros(c, c, device=delta_weight.device)
            padded_delta_weight[:r, :r] = delta_weight
            padded_delta_weight[r:, r:] = torch.eye(pad_size, device=delta_weight.device)
            delta_weight = padded_delta_weight

        return torch.matmul(input, delta_weight)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "compound." + rep


class Conv2d(CompoundLayer):
    """Compound Adapter implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        r: int,
        adapter_name: str = "default",
        module_dropout: float = 0.0,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(r,adapter_name,module_dropout, init_weights, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)
        if input.shape[-1] != delta_weight.shape[0]:  # Compare c with r
            c = input.shape[-1]
            r = delta_weight.shape[0]
            pad_size = c - r
            padded_delta_weight = torch.zeros(c, c, device=delta_weight.device)
            padded_delta_weight[:r, :r] = delta_weight
            padded_delta_weight[r:, r:] = torch.eye(pad_size, device=delta_weight.device)
            delta_weight = padded_delta_weight
        # don't add bias here, because the bias will be added after Compound forward
        return torch.matmul(input, delta_weight)


    def __repr__(self) -> str:
        rep = super().__repr__()
        return "compound." + rep