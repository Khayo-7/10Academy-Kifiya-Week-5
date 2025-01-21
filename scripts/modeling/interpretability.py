import shap
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from datasets import load_dataset


def explain_predictions(model_name: str, dataset_path: str, example_idx: int, label_map: dict):
    """
    Explain model predictions using SHAP.

    Args:
    - model_name (str): Path to the fine-tuned model or Hugging Face model hub name.
    - dataset_path (str): Path to the dataset for analysis.
    - example_idx (int): Index of the example to interpret.
    - label_map (dict): Mapping of label IDs to entity names.

    Returns:
    - None: Visualizes and prints explanations.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

    # Load dataset and example
    dataset = load_dataset("text", data_files={"data": dataset_path})["data"]
    example_text = dataset[example_idx]["text"]

    # Generate predictions
    predictions = ner_pipeline(example_text)

    # Tokenize text
    tokens = tokenizer(example_text, truncation=True, return_tensors="pt")

    # Set up SHAP explainer
    def model_predict(inputs):
        outputs = model(**inputs).logits
        return torch.nn.functional.softmax(outputs, dim=-1).detach().numpy()

    explainer = shap.Explainer(model_predict, tokenizer)
    shap_values = explainer(tokens)

    # Visualization
    shap.summary_plot(shap_values, tokens.tokens, class_names=list(label_map.values()))
