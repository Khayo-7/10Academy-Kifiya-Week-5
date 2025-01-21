import os
import sys
from datasets import DatasetDict
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import load_yml
from scripts.modeling.metrics import compute_metrics
from scripts.modeling.trainer import ner_model_trainer
from scripts.modeling.model_utils import initialize_tokenizer, initialize_model, initialize_data_collator, \
                                            load_conll_data, tokenize_and_align_labels, map_and_format_datasets

logger = setup_logger("training")

# Load configuration from YAML file
config_file = "config.yaml"
config = load_yml(config_file)

# # Constants from config
# LABEL_LIST = config["label_list"]
# TRAINING_CONFIG = config["training"]
# MODEL_CHECKPOINT = config["model_checkpoint"]
# MODEL_OUTPUT_DIR = config["model_output_dir"]

# Model and Data Constants
MODEL_CHECKPOINT = "xlm-roberta-base"
LABEL_LIST = ["O", "B-Product", "I-Product", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE"]
label_mapping = map_and_format_datasets(LABEL_LIST)
# Training Constants
# LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
# EPOCHS = TRAINING_CONFIG["epochs"]
# BATCH_SIZE = TRAINING_CONFIG["batch_size"]
LEARNING_RATE = 5e-5
EPOCHS = 5
BATCH_SIZE = 8

MODEL_OUTPUT_DIR = os.path.join("resources", "models", "checkpoints")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def ner_model_trainer(
    model,
    datasets: DatasetDict,
    tokenizer,
    data_collator: DataCollatorForTokenClassification,
    output_dir: str,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> Trainer:
    """
    Fine-tune a pre-trained NER model.

    Args:
        model: Pre-trained model.
        datasets (DatasetDict): Dataset containing train, val, and test sets.
        tokenizer: Tokenizer for the model.
        data_collator (DataCollatorForTokenClassification): Data collator for token classification.
        output_dir (str): Directory to save the model.
        learning_rate (float): Learning rate for training.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.

    Returns:
        Trainer: Hugging Face Trainer instance.
    """

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=3,
        push_to_hub=False,
        load_best_model_at_end=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        early_stopping_patience=3,  # Early stopping
        report_to="wandb", # Experiment tracking
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, LABEL_LIST),
    )

    return trainer

def train_ner_model(data_path: str, output_dir: str = MODEL_OUTPUT_DIR, model_name: str = MODEL_CHECKPOINT) -> None:
    """
    Train a Named Entity Recognition model.

    Args:
        data_path (str): Path to the dataset.
        output_dir (str): Directory to save the model.
        model_name (str): Name of the pre-trained model.
    """
    try:
        # Load tokenizer and model
        tokenizer = initialize_tokenizer(model_name)
        model = initialize_model(model_name, num_labels=len(LABEL_LIST))

        # Define data collator
        data_collator = initialize_data_collator(tokenizer)

        # Load datasets
        dataset = load_conll_data(data_path, use_hf_load_dataset=False)
        # label_list = dataset["train"].features["ner_tags"].feature.names

        # Tokenize datasets
        tokenized_datasets = dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, num_proc=4 # 4 Parallel processing
        )

        # Initialize Trainer
        trainer = ner_model_trainer(model, tokenized_datasets, tokenizer, data_collator, output_dir)

        # Train the model
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model fine-tuned and saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise