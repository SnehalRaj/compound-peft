import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from peft import get_peft_model, CompoundConfig, TaskType, PeftModel
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import numpy as np
import tempfile
import logging
import random
from datasets import Dataset, DatasetDict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import importlib
import gc
import copy
import time
import os

def generate_unique_seed():
    return int(time.time() * 1000) & 0xffffffff  # Use current time in milliseconds, masked to 32 bits

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

def analyze_model_parameters(model):
    layer_averages = {}
    for name, param in model.named_parameters():
        # if param.requires_grad:
        layer_name = '.'.join(name.split('.')[:5])  # Get the top-level layer name
        if layer_name not in layer_averages:
            layer_averages[layer_name] = []
        layer_averages[layer_name].append(param.data.mean().item())
    
    for layer_name, values in layer_averages.items():
        layer_averages[layer_name] = np.mean(values)
    
    return layer_averages

def reload_peft():
    import peft
    importlib.reload(peft)

def load_glue_dataset(task, debug=False, debug_sample_size=16):
    dataset = load_dataset("glue", task)
    if debug:
        for split in dataset.keys():
            dataset[split] = dataset[split].select(range(min(len(dataset[split]), debug_sample_size)))
    return dataset

def compute_metrics(eval_pred, task):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)

    if task == "cola":
        return {"matthews_correlation": matthews_corrcoef(labels, predictions)}
    elif task == "stsb":
        return {"pearson": pearsonr(labels, predictions)[0], "spearmanr": spearmanr(labels, predictions)[0]}
    elif task in ["mrpc", "qqp"]:
        return {"accuracy": accuracy_score(labels, predictions), "f1": f1_score(labels, predictions)}
    else:
        return {"accuracy": accuracy_score(labels, predictions)}

def tokenize_function(examples, tokenizer, task, max_length):
    if task in ["mnli", "qnli", "qqp", "rte", "mrpc", "stsb"]:
        result = tokenizer(examples["premise" if task == "mnli" else "sentence1"], 
                           examples["hypothesis" if task == "mnli" else "sentence2"], 
                           padding="max_length", truncation=True, max_length=max_length)
    elif task in ["sst2", "cola"]:
        result = tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)
    else:
        raise ValueError(f"Unsupported task: {task}")

    if "label" in examples:
        result["labels"] = examples["label"]
    return result

def get_training_args(task, debug=False, tmp_dir=None, seed=None):
    common_args = {
        "output_dir": tmp_dir,
        "evaluation_strategy": "steps",
        "save_strategy": "no",
        "eval_steps": 100 if debug else 500,
        "fp16": False,
        "dataloader_num_workers": 0,
        "group_by_length": True,
        "disable_tqdm": False,
        "report_to": "none",
        "logging_steps": 10,
        "lr_scheduler_type": "cosine",
    }

    if debug:
        common_args.update({
            "num_train_epochs": 20,
            "per_device_train_batch_size": 64,
            "per_device_eval_batch_size": 128,
            "warmup_ratio": 0.06,
            "weight_decay": 0.01,
            "learning_rate": 8e-4,
        })
    else:
        common_args.update({
            "num_train_epochs": 5,
            "warmup_ratio": 0.06,
            "weight_decay": 0.01,
            "per_device_train_batch_size": 64,
            "per_device_eval_batch_size": 128,
            "learning_rate": 7e-4,
        })
    common_args.update({
        "seed":seed,
    })
    return TrainingArguments(**common_args)

def get_compound_config(block_share, compound_pattern, compound_type):
    return CompoundConfig(
        r=8,
        compound_pattern=compound_pattern,
        compound_type=compound_type,
        block_share=block_share,
        task_type=TaskType.SEQ_CLS,
        target_modules=["query", "key", "value", "output.dense"],
    )

def save_results(df, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"compound_adapter_results_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def load_results(filename):
    return pd.read_csv(filename)




def initialize_model(model_name, task, seed):
    set_seed(seed)
    num_labels = 3 if task == "mnli" else 1 if task == "stsb" else 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Add small random noise to model parameters
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * 1e-5  # Small noise
            param.add_(noise)
    
    return model

def run_experiment(base_model, tokenizer, dataset, task, block_share, compound_pattern, compound_type, debug=False, seed=None):
    if seed is not None:
        set_seed(seed)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        max_length = 128

        tokenized_datasets = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, task, max_length),
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Create a deep copy of the model for this experiment
        model = copy.deepcopy(base_model)

        peft_config = get_compound_config(block_share, compound_pattern, compound_type)
        model = get_peft_model(model, peft_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        training_args = get_training_args(task, debug, tmp_dir, seed)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation_matched" if task == "mnli" else "validation"],
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, task),
            data_collator=data_collator,
        )

        # # Analyze parameters before training
        # initial_params = analyze_model_parameters(model)
        # logger.info("Initial average parameter values by layer:")
        # for layer, avg in initial_params.items():
        #     logger.info(f"{layer}: {avg:.6f}")

        logger.info(f"Starting training for Compound adapter with block_share={block_share}, compound_pattern={compound_pattern}, compound_type={compound_type}")
        trainer.train()
        logger.info("Completed training")

        # # Analyze parameters after training
        # final_params = analyze_model_parameters(model)
        # logger.info("Final average parameter values by layer:")
        # for layer, avg in final_params.items():
        #     logger.info(f"{layer}: {avg:.6f}")

        logger.info("Starting evaluation")
        eval_results = trainer.evaluate()
        logger.info("Completed evaluation")

        # Clean up
        del model, trainer
        free_memory()

        return eval_results, trainable_params

def run_experiments_for_task(base_model, tokenizer, dataset, task, configurations, num_runs=2, debug=False):
    results = []
    for block_share, compound_pattern, compound_type in configurations:
        config_name = f"bs_{block_share}_cp_{'-'.join(compound_pattern)}_ct_{compound_type}"
        print(f"Running experiments for configuration: {config_name}")
        
        config_results = []
        for run in range(num_runs):
            print(f' Reloading peft...')
            reload_peft()

            print(f' Free memory...')
            free_memory()
            print(f"  Run {run + 1}/{num_runs}")

            seed = generate_unique_seed()
            print(f"  Using seed: {seed}")
            eval_results, params = run_experiment(
                base_model, tokenizer, dataset, task, block_share, compound_pattern, compound_type, debug=debug, seed=seed
            )
            if eval_results is not None:
                if task == "cola":
                    eval_acc = eval_results.get("eval_matthews_correlation", 0.0) * 100
                elif task == "stsb":
                    pearson = eval_results.get("eval_pearson", 0.0)
                    spearman = eval_results.get("eval_spearmanr", 0.0)
                    eval_acc = (pearson + spearman) / 2 * 100
                else:
                    eval_acc = eval_results.get("eval_accuracy", 0.0) * 100
            else:
                eval_acc = 0.0
            
            config_results.append({
                "task": task,
                "run": run + 1,
                "accuracy": eval_acc,
                "params": params,
                "seed": seed,
            })
        
        avg_accuracy = np.mean([r["accuracy"] for r in config_results])
        std_accuracy = np.std([r["accuracy"] for r in config_results])
        
        results.append({
            "task": task,
            "block_share": block_share,
            "compound_pattern": '-'.join(compound_pattern),
            "compound_type": compound_type,
            "avg_accuracy": avg_accuracy,
            "std_accuracy": std_accuracy,
            "params": config_results[0]["params"], 
            "individual_runs": config_results
        })
    
    return results


def analyze_results(df):
    print("\nOverall Results:")
    print(df[["block_share", "compound_pattern", "compound_type", "avg_accuracy", "std_accuracy", "params"]])

    print("\nAverage Accuracy by Compound Type:")
    print(df.groupby("compound_type")["avg_accuracy"].mean())

    print("\nAverage Accuracy by Block Share:")
    print(df.groupby("block_share")["avg_accuracy"].mean())

    print("\nTop 5 Configurations by Average Accuracy:")
    print(df.sort_values("avg_accuracy", ascending=False).head())

    # Visualizations
    plt.figure(figsize=(12, 6))
    sns.barplot(x="compound_type", y="avg_accuracy", data=df)
    plt.errorbar(x=df.index, y=df["avg_accuracy"], yerr=df["std_accuracy"], fmt="none", c="k", capsize=5)
    plt.title("Average Accuracy by Compound Type")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x="params", y="avg_accuracy", hue="compound_type", data=df)
    plt.errorbar(x=df["params"], y=df["avg_accuracy"], yerr=df["std_accuracy"], fmt="none", c="k", alpha=0.3)
    plt.title("Average Accuracy vs Number of Parameters")
    plt.show()

    # Additional analysis: Effect of number of components
    df['num_components'] = df['compound_pattern'].apply(lambda x: len(x.split('-')))
    print("\nAverage Accuracy by Number of Components:")
    print(df.groupby("num_components")["avg_accuracy"].mean())

def run_multi_task_experiments(tasks, configurations, debug=False, debug_sample_size=16, num_runs=3):
    model_name = "microsoft/deberta-v3-base"
    all_results = []

    for task in tasks:
        print(f"\nRunning experiments for task: {task}")
        base_seed = generate_unique_seed()
        print(f"Using base seed: {base_seed}")
        base_model = initialize_model(model_name, task, base_seed)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = load_glue_dataset(task, debug, debug_sample_size)
        
        task_results = run_experiments_for_task(base_model, tokenizer, dataset, task, configurations, num_runs=num_runs, debug=debug)
        all_results.extend(task_results)

        # Save intermediate results
        intermediate_df = pd.DataFrame(all_results)
        save_results(intermediate_df, f"intermediate_results_{task}.csv")

    return pd.DataFrame(all_results)



def main(tasks, configurations, debug=False, debug_sample_size=16, load_from_file=None, num_runs=3):
    if load_from_file:
        results_df = load_results(load_from_file)
        print(f"Loaded results from {load_from_file}")
    else:
        results_df = run_multi_task_experiments(tasks, configurations, debug, debug_sample_size, num_runs)
        save_results(results_df, "multi_task_results.csv")
    
    analyze_results(results_df)

    # Example of custom analysis
    print("\nCustom Analysis: Comparing 'avg' and 'max' compound types across tasks")
    avg_max_df = results_df[results_df['compound_type'].isin(['avg', 'max'])]
    print(avg_max_df.groupby(['task', 'compound_type', 'block_share'])['avg_accuracy'].mean().unstack().unstack())

import itertools

# Define the basic parameters
block_share_options = [False]
compound_components = ['comp_1','comp_2','comp_3']
compound_types = [ 'comp']
max_components = 3

# Generate all possible compound patterns
compound_patterns = [list(combo) for i in range(1, max_components + 1) 
                     for combo in itertools.combinations(compound_components, i)]

# Generate all configurations
all_configurations = list(itertools.product(block_share_options, compound_patterns, compound_types))

# Example: Filter configurations (e.g., exclude single component patterns)
filtered_configurations = [config for config in all_configurations if len(config[1]) > 1]

# Example: Generate specific configurations
specific_configurations = [
    config for config in all_configurations
    if (config[0] and 'comp_1' in config[1] and config[2] == 'avg') or
       (not config[0] and 'comp_2' in config[1] and 'comp_3' in config[1])
]

# Print some examples
print("All configurations:", len(all_configurations))
print("Filtered configurations:", len(filtered_configurations))
print("Specific configurations:", len(specific_configurations))



# Use in main script
configurations_for_experiment = all_configurations # or specific_configurations

print("\n Configurations for experiment:")
for config in configurations_for_experiment:
    print(config)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_SAMPLE_SIZE = 16  # debug mode downsampling
tasks = ["cola", "mrpc", "rte"] 
# To run new experiments and save results:
# main(tasks=tasks, configurations=configurations_for_experiment, debug=False, debug_sample_size=DEBUG_SAMPLE_SIZE)

# To load previous results and analyze:
main(tasks=tasks, configurations=configurations_for_experiment,load_from_file="multi_task_results.csv")