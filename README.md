# QuIC Adapters

**QuIC Adapters** is a fork of [PEFT](https://github.com/huggingface/peft) that injects Quantum-Inspired **Compound Adapters** into your pretrained models for parameter-efficient fine-tuning.

## Installation

```bash
pip install -e .
````

## Quickstart

```python
from peft import get_peft_model, CompoundConfig

# 1. Define your CompoundConfig
peft_config = CompoundConfig(
    r=4,
    compound_pattern=["comp_1","comp_2"],
    compound_type="comp",
    block_share=False,
    use_orthogonal=True,
    num_adapters=1,
    adapter_multiplicative=True,
    task_type="CAUSAL_LM",
    target_modules=["q_proj","v_proj"],  # e.g. your model’s projection layers
)


