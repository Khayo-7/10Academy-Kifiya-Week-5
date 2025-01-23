import os
import sys
import json
import pandas as pd
from datetime import datetime
from datasets import DatasetDict
from typing import List, Dict, Tuple
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import load_yml
from scripts.modeling.metrics import compute_metrics
from scripts.modeling.model_utils import (
                                            initialize_tokenizer, 
                                            initialize_model,  
                                            initialize_data_collator, 
                                            save_metadata,
                                            load_tokenized_datasets, 
                                            load_metadata, 
                                            load_and_tokenize_dataset, 
                                            save_tokenized_dataset
                                        )

logger = setup_logger("training")

# Load configuration from YAML file
# config_file = "config.yaml"
# config = load_yml(config_file)

# # Constants from config
# NER_TAGS = config["ner_tags"]
# TRAINING_CONFIG = config["training"]
# MODEL_CHECKPOINT = config["model_checkpoint"]
# MODEL_OUTPUT_DIR = config["model_output_dir"]

# Training Constants
# LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
# EPOCHS = TRAINING_CONFIG["epochs"]
# BATCH_SIZE = TRAINING_CONFIG["batch_size"]
LEARNING_RATE = 5e-5
EPOCHS = 5
BATCH_SIZE = 8

# Model and Data Constants
MODEL_CHECKPOINT = "xlm-roberta-base"
NER_TAGS = ["O", "B-Product", "I-Product", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE"]
MODEL_OUTPUT_DIR = os.path.join("..", "resources", "models", "checkpoints", "ner_model")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join('..', 'resources', 'data')
TOKENIZED_DIR = os.path.join(DATA_PATH, "tokenized_dataset")
os.makedirs(TOKENIZED_DIR, exist_ok=True)
METADATA_FILEPATH = os.path.join(TOKENIZED_DIR, "metadata.json")

def ner_model_trainer(
    model,
    datasets: DatasetDict,
    tokenizer,
    data_collator: DataCollatorForTokenClassification,
    output_dir: str,
    ner_tags: List = NER_TAGS,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> Trainer:
    """
    Initialize and configure the Trainer for NER model fine-tuning.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save at the end of each epoch
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=3,
        push_to_hub=False,
        load_best_model_at_end=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        report_to="wandb",  # Experiment tracking
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, ner_tags),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], 
    )

    return trainer

def apply_peft_to_model(model, task_type: TaskType = TaskType.TOKEN_CLS):
    """
    Apply PEFT (LoRA) to the model for parameter-efficient fine-tuning.
    """
    lora_config = LoraConfig(
        task_type=task_type,
        r=8,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,  # Dropout rate
        target_modules=["q_proj", "v_proj"],  # Target modules to apply LoRA
    )
    return get_peft_model(model, lora_config)


def initialize_and_apply_peft(model_name: str, num_labels: int, use_peft: bool = False):
    """
    Initialize the model and apply PEFT if needed.

    Args:
        model_name (str): Name of the pre-trained model.
        num_labels (int): Number of labels for the model.
        use_peft (bool): Whether to use PEFT for fine-tuning.

    Returns:
        Model: Initialized model.
    """
    model = initialize_model(model_name, num_labels=num_labels)

    if use_peft:
        model = apply_peft_to_model(model, TaskType.TOKEN_CLS)
        logger.info("[INFO] PEFT (LoRA) applied to the model for efficient fine-tuning.")

    return model

def train_and_save_model(model, tokenized_datasets, tokenizer, data_collator, output_dir, ner_tags, learning_rate, epochs, batch_size):
    """
    Train the model and save it to disk.

    Args:
        model: Model to train.
        tokenized_datasets (DatasetDict): Tokenized dataset.
        tokenizer: Tokenizer used.
        data_collator: Data collator.
        output_dir (str): Directory to save the model.
        ner_tags (List): List of NER tags.
        learning_rate (float): Learning rate for training.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    trainer = ner_model_trainer(
        model, tokenized_datasets, tokenizer, data_collator, output_dir, ner_tags, learning_rate, epochs, batch_size
    )

    # Train the model
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model fine-tuned and saved to {output_dir} with tokenizer.")

    return trainer

def train_ner_model(
    data_path: str,
    model_name: str = MODEL_CHECKPOINT,
    output_dir: str = MODEL_OUTPUT_DIR,
    ner_tags: List = NER_TAGS,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    use_peft: bool = False,
    use_hf: bool = False,
    metadata_filepath: str = METADATA_FILEPATH,
    tokenized_dir: str = TOKENIZED_DIR,
) -> None:
    """
    Train a Named Entity Recognition model with optional PEFT (LoRA) fine-tuning.

    Args:
        data_path (str): Path to the dataset.
        model_name (str): Name of the pre-trained model.
        output_dir (str): Directory to save the model.
        ner_tags (List): List of NER tags.
        learning_rate (float): Learning rate for training.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        use_peft (bool): Whether to use PEFT for fine-tuning.
        use_hf (bool): Whether to use Hugging Face for loading dataset.
        metadata_filepath (str): Path to save/load metadata.
        tokenized_dir (str): Directory to save/load tokenized dataset.
    """
    try:
        # Check if metadata and tokenized dataset exist
        if os.path.exists(metadata_filepath) and os.path.exists(tokenized_dir):
            tokenized_datasets = load_tokenized_datasets(tokenized_dir)
            metadata = load_metadata(metadata_filepath)
            max_length = metadata["max_length"]
            ner_tags = metadata["ner_tags"]
            tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"])
        else:
            max_length = 128
            tokenizer = initialize_tokenizer(model_name)
            tokenized_datasets = load_and_tokenize_dataset(data_path, tokenizer, ner_tags, max_length, use_hf)
            
            metadata = {
                "tokenizer": tokenizer.name_or_path,
                "max_length": max_length,
                "ner_tags": ner_tags,
                "preprocessing_date": datetime.now().isoformat(),
                "version": "1.0",
            }
            save_tokenized_dataset(tokenized_datasets, tokenized_dir)
            save_metadata(metadata, metadata_filepath)

        # Initialize model and apply PEFT if needed
        model = initialize_and_apply_peft(model_name, len(ner_tags), use_peft)

        # Define data collator
        data_collator = initialize_data_collator(tokenizer)

        # Train and save the model
        trainer = train_and_save_model(model, tokenized_datasets, tokenizer, data_collator, output_dir, ner_tags, learning_rate, epochs, batch_size)

        return trainer

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def fine_tune_multiple_models(
    models: List[str],
    dataset_dir: str,
    base_output_dir: str,
    params: Dict,
    use_peft: bool = False,
    use_hf: bool = False,
) -> pd.DataFrame:
    """
    Fine-tune multiple models with optional PEFT and compare their performance.

    Args:
        models (List[str]): List of model names to fine-tune.
        dataset_dir (str): Path to the dataset directory.
        base_output_dir (str): Base directory to save fine-tuned models.
        params (Dict): Training parameters (learning_rate, epochs, batch_size).
        use_peft (bool): Whether to use PEFT for fine-tuning.
        use_hf (bool): Whether to use Hagging Face for loading dataset.

    Returns:
        pd.DataFrame: DataFrame containing evaluation results for all models.
    """
    trainers = []
    results = []

    for model_name in models:
        logger.info(f"\n[INFO] Fine-tuning model: {model_name}")
        output_dir = os.path.join(base_output_dir, model_name.replace("/", "_"))

        # Fine-tune the model
        trainer, _ = train_ner_model(
            model_name=model_name,
            data_path=dataset_dir,
            output_dir=output_dir,
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            use_peft=use_peft,
            use_hf=use_hf,
        )
        trainers.append(trainer)

        # Load validation results
        eval_results_path = os.path.join(output_dir, "trainer_state.json")
        if os.path.exists(eval_results_path):
            with open(eval_results_path, "r") as f:
                eval_results = json.load(f)
            eval_metrics = eval_results.get("metrics", {}).get("eval", {})
            eval_metrics["model_name"] = model_name
            results.append(eval_metrics)

    # Create a DataFrame for comparison
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="eval_f1", ascending=False)
    results_df.to_csv(os.path.join(base_output_dir, "model_comparison_results.csv"), index=False)

    logger.info("\n[INFO] Model comparison complete. Results saved.")
    return trainers, results_df