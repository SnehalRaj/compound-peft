# Compound Adapters for GLUE and VTAB1K Benchmarks

This repository contains implementations of Compound Adapters for natural language understanding and computer vision tasks. It includes code for running experiments on GLUE and VTAB1K datasets.

## Overview

Compound Adapters implement a flexible architecture for parameter-efficient fine-tuning that combines multiple adapter components in different ways. This method allows for efficient transfer learning across various tasks while maintaining a small parameter footprint.

## Requirements

```
torch
transformers
datasets
peft
scikit-learn
scipy
numpy
pandas
matplotlib
seaborn
```

## Hyperparameters

### GLUE Benchmark

For GLUE experiments, we use the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| Base Model | DeBERTaV3-base |
| Batch Size | 32-64 (varies by configuration) |
| Learning Rate | ~7e-4 (task-specific, ranges from 2e-5 to 9e-4) |
| Warmup Ratio | 0.06 |
| Weight Decay | 0.01 |
| Number of Epochs | 5-20 (varies by task and debug mode) |
| Max Sequence Length | Task-specific (64-512) |

GLUE tasks use different learning rates for optimal performance:
- CoLA: ~4e-4
- MRPC: ~8e-4 
- RTE: ~3e-4
- SST-2: ~2e-4
- QNLI: ~2e-4
- QQP: ~3e-4
- STS-B: ~7e-4
- MNLI: ~8e-5

### VTAB1K Benchmark

For VTAB1K experiments, we use the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| Base Model | DINOv2-large |
| Batch Size | 16 |
| Learning Rate | Task-specific (ranges from 2e-6 to 3e-3) |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.02 |
| Number of Epochs | 15-60 (varies by task) |
| Dropout Rate | 0.1 |

VTAB tasks use different learning rates for optimal performance:
- CIFAR100: 8e-4
- Caltech101: 5e-4 - 9e-4
- DTD: 6e-4 - 1e-3
- Flowers102: 1e-3 - 2e-3
- Pets: 3e-4 - 8e-4
- SVHN: 3e-3
- EuroSAT: 6e-4 - 9e-4
- Resisc45: 5e-4 - 1e-3
- PatchCamelyon: 9e-4 - 4e-3

## GLUE Benchmark Experiments

The GLUE benchmark is a collection of tasks for evaluating natural language understanding systems.

### Running GLUE Experiments

```bash
python compound_glue.py
```

By default, the script runs experiments on CoLA, MRPC, and RTE tasks. You can modify the `tasks` list in the script to include other GLUE tasks.

Key features:
- Uses DeBERTa-v3-base as the backbone model
- Implements compound adapters using PEFT
- Runs with 3 different random seeds and averages results for reliable performance evaluation
- Supports various compound pattern configurations

### Configuration Options

You can modify the following parameters in the script:
- `tasks`: List of GLUE tasks to evaluate
- `configurations_for_experiment`: Different compound adapter configurations to test
- `debug`: Set to True for quick testing with smaller datasets

## VTAB1K Benchmark Experiments

The VTAB1K benchmark evaluates visual transfer learning across diverse computer vision tasks.

### Running VTAB1K Experiments

```bash
python compound_vtab1k.py
```

By default, the script runs experiments on selected VTAB tasks including CIFAR-100. You can modify the `vtab_dataset_configs` dictionary to include other VTAB tasks.

Key features:
- Uses DINOv2-large as the backbone model
- Implements compound adapters for vision models
- Runs with 3 different random seeds and averages results for reliable performance evaluation
- Evaluates both validation and test performance

### Configuration Options

You can modify the following parameters in the script:
- `vtab_dataset_configs`: Dictionary of VTAB datasets and their configurations
- `configurations_for_experiment`: Different compound adapter configurations to test
- `debug`: Set to True for quick testing with smaller datasets

## Experiment Methodology

For both benchmarks, we run experiments with **3 different random seeds** and report the averaged performance metrics. This approach provides more reliable estimates of model performance and reduces variance due to random initialization.

All experiments for GLUE tasks can be run on a single NVIDIA A100-SXM4-80GB GPU. Similarly, the VTAB1K experiments are designed to run on a single NVIDIA A100-SXM4-80GB GPU.

Results include:
- Mean and standard deviation of accuracy metrics
- Detailed per-run performance statistics
- Parameter counts for each configuration

## Results Analysis

After running experiments, the scripts provide detailed analysis of the results, including:
- Overall performance across different configurations
- Comparisons between different compound adapter types
- Efficiency analysis (performance vs. parameter count)

Results are saved in CSV format for further analysis.

