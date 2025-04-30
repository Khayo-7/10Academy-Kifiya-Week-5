# 10Academy-Kifiya-Week-5

# EthioMart: Building a Named Entity Recognition (NER) System for Amharic: Challenges, Strategies, and Insights

## Introduction

With the growing adoption of Telegram as a platform for e-commerce activities in Ethiopia, numerous independent channels facilitate business transactions. However, this decentralization poses significant challenges for both customers and vendors. The EthioMart NER project aims to create a centralized platform for Telegram-based e-commerce in Ethiopia by extracting key business entities (e.g., product names, prices, and locations) from Telegram messages. EthioMart’s vision is to centralize these activities, providing a unified platform that consolidates real-time data from these channels. A crucial part of this initiative involves fine-tuning Named Entity Recognition (NER) models tailored for Amharic to extract key business entities from text shared across these channels.

This report outlines the step-by-step process of developing an Amharic NER system, including the collection, preprocessing, annotation, and modeling phases, as well as insights derived from each stage. The discussion concludes with key results, challenges, and recommendations for future improvements.

---

## **Data Preparation**

### **Data Ingestion**
- **Objective:** Fetch messages from Ethiopian Telegram e-commerce channels.
- **Steps Completed:**
  1. Identified and connected to 5 Telegram channels (e.g., Shageronlinestore, ShegerMart, AddisMall).
  2. Developed a Python script (`scraper.py`) using the `telethon` library to scrape messages in real-time.
  3. Fetched and stored raw messages, including text and metadata (e.g., sender, timestamp, channel name).

### **Data Preprocessing**
- **Objective:** Clean and normalize the raw data for further analysis.
- **Steps Completed:**
  1. Cleaned text data by removing special characters, emojis, and unnecessary symbols.
  2. Normalized Amharic text (e.g., removed diacritics, standardized characters).
  3. Tokenized text into individual words for easier processing.
  4. Extracted text from product images using Tesseract OCR with Amharic language support.
  5. Saved preprocessed data in a structured format (`resources/data/preprocessed_data.csv`).

#### **Challenges**
- **Amharic Text Processing:** Handling Amharic-specific linguistic features (e.g., diacritics, compound words) required custom normalization techniques.
- **OCR Limitations:** Extracting text from low-quality images or handwritten text proved challenging.

---

## **Data Labeling**

### **Labeling Process**
- **Objective:** Label a subset of the preprocessed data in CoNLL format for NER tasks.
- **Steps Completed:**
  1. Created a labeling script (`label_data.py`) to assist with manual labeling.
  2. Defined entity labels:
     - `B-Product`, `I-Product` for product names.
     - `B-PRICE`, `I-PRICE` for prices.
     - `B-LOC`, `I-LOC` for locations.
     - `O` for non-entity tokens.
  3. Labeled 30-50 messages, ensuring consistency and accuracy in annotations.
  4. Saved the labeled dataset in CoNLL format (`resources/data/labeled/labeled_data.conll`).

#### **Labeling Example**
Example of labeled data in CoNLL format:
```
የቤት    O
ውስጥ    O
እቃ    B-Product
በ1000   B-PRICE
ብር    I-PRICE
```

### **Challenges**
- **Ambiguity:** Some tokens were ambiguous (e.g., words that could be either product names or locations).
- **Time-Consuming:** Manual labeling required significant effort to ensure high-quality annotations.

---

### Summary of Deliverables**

| **Task**               | **Deliverables**                                                                 |
|-------------------------|---------------------------------------------------------------------------------|
| Data Ingestion          | Python script (`scraper.py`) to fetch messages from Telegram channels.          |
| Data Preprocessing      | Cleaned and normalized dataset (`resources/data/preprocessed_data.csv`).        |
| Data Labeling           | Labeled dataset in CoNLL format (`resources/data/labeled/labeled_data.conll`).  |
| Labeling Script         | Python script (`label_data.py`) for manual labeling.                            |


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

---

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

