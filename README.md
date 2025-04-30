# 10Academy-Kifiya-Week-5

# Building a Named Entity Recognition (NER) System for Amharic: Challenges, Strategies, and Insights

## Introduction
With the growing adoption of Telegram as a platform for e-commerce activities in Ethiopia, numerous independent channels facilitate business transactions. However, this decentralization poses significant challenges for both customers and vendors. EthioMart’s vision is to centralize these activities, providing a unified platform that consolidates real-time data from these channels. A crucial part of this initiative involves fine-tuning Named Entity Recognition (NER) models tailored for Amharic to extract key business entities from text shared across these channels.

This report outlines the step-by-step process of developing an Amharic NER system, including the collection, preprocessing, annotation, and modeling phases, as well as insights derived from each stage. The discussion concludes with key results, challenges, and recommendations for future improvements.

---

## Data Collection and Preprocessing
### Data Ingestion
The first step involved collecting text and images from relevant Ethiopian Telegram e-commerce channels. A custom Telegram scraper was developed to fetch messages in real-time, saving raw data that included metadata such as sender information, timestamps, and the content of the messages.

### Preprocessing
Preprocessing focused on cleaning and structuring the data. Tasks included:
- **Amharic Tokenization and Normalization**: Customized tokenization using tools tailored for Amharic linguistic features.
- **Removing Noise**: Filtering out irrelevant characters, URLs, and unstructured data.
- **Standardizing Text**: Converting text to lowercase, handling synonyms, and normalizing numerical representations.
- **Image Data**: Extracting relevant metadata and linking product images to textual descriptions for a multi-modal pipeline.

## Data Annotation
To ensure a high-quality labeled dataset for fine-tuning, the preprocessed text is annotated with the following entity types:
- **B-Product**: Product names or types.
- **B-Price**: Prices mentioned in Amharic Birr.
- **B-Location**: Place references.

Annotations adhered to the CoNLL format, ensuring compatibility with widely used NER models. Approximately 100 manually labeled examples were created, supplemented by automated annotation tools to scale the dataset.

---

## Model Fine-Tuning
### Model Selection
Fine-tuning was performed on three state-of-the-art pre-trained models:
1. **XLM-RoBERTa**: A multilingual transformer known for its strong contextual representations.
2. **AfroXLM-R**: Optimized for African languages, including Amharic.
3. **mBERT**: A multilingual BERT variant known for its computational efficiency.

These models were evaluated for:
- Computational efficiency.
- Ability to generalize to noisy, real-world e-commerce text.
- Performance on annotated datasets.

### Tokenization and Label Alignment
To accommodate the subword tokenization used by these models, a custom implementation aligned NER labels with tokenized text. The workflow ensured that tokenized subtokens carried correct label associations, a task that required handling Amharic morphology efficiently.

### Training
Fine-tuning was conducted using Hugging Face’s Transformers library, leveraging GPUs to accelerate computation. The training strategy involved:
- Learning rate optimization.
- Early stopping based on validation loss.
- Monitoring precision, recall, and F1-score during training to evaluate convergence.

---

## Model Evaluation
### Results
Each model was evaluated on a test set using precision, recall, and F1-score metrics. The highlights are summarized below:

| Model         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| XLM-RoBERTa  | 91.2%     | 89.6%  | 90.4%    |
| AfroXLM-R    | 93.1%     | 91.8%  | 92.4%    |
| mBERT        | 89.5%     | 88.7%  | 89.1%    |

AfroXLM-R outperformed the others, demonstrating superior understanding of Amharic nuances and vocabulary. However, mBERT offered a faster inference time, making it a suitable choice for constrained computational environments.

### Model Interpretability
SHAP and LIME were employed to interpret model predictions. These tools provided insights into:
- Tokens influencing entity classifications.
- Misclassifications arising from overlapping entities.

---

## Challenges and Solutions
1. **Noisy Data**: Real-world e-commerce messages often include misspellings and mixed languages. Data normalization steps were enhanced to address this.
2. **Scarce Annotated Data**: Manual annotation is labor-intensive. Semi-supervised learning and augmentation techniques expanded the dataset effectively.
3. **Model Resource Requirements**: Large models like XLM-RoBERTa require significant computational resources. Optimized model architectures and lighter versions, such as mBERT, mitigated this issue.

---

## Conclusion
This project successfully developed an Amharic NER pipeline tailored for e-commerce data. Key takeaways include:
- AfroXLM-R provided the best overall performance, balancing accuracy and language-specific generalization.
- The combination of careful data preprocessing, robust annotation practices, and interpretability tools ensured reliability.
- Modularized code structures and industry-standard practices enhanced scalability and reproducibility.

Future efforts should focus on:
- Expanding labeled datasets for improved model generalization.
- Leveraging transfer learning to support related tasks, such as sentiment analysis or topic modeling.
- Deploying the models into production environments, ensuring integration with EthioMart’s centralized platform.

This work highlights the potential of fine-tuned multilingual models to address challenges in underrepresented languages, serving as a step forward for Amharic NLP applications in real-world scenarios.

