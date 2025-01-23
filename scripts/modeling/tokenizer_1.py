import logging
from typing import Dict, List, Optional, Union
import numpy as np
from datasets import ClassLabel
from transformers import AutoTokenizer
from amseg.amharicSegmenter import AmharicSegmenter

# Setup logger
logger = logging.getLogger("tokenizer")


class Tokenizer:
    """
    A highly efficient and modular tokenizer class for tokenizing and aligning labels with tokens.
    Supports both Hugging Face tokenizers and custom tokenizers (e.g., AmharicSegmenter).
    """

    def __init__(
        self,
        tokenizer: Union[AutoTokenizer, "AmharicSegmenter"],
        max_length: int = 128,
        ner_tags: Optional[List[str]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Initializes the Tokenizer.

        Args:
            tokenizer (Union[AutoTokenizer, AmharicSegmenter]): Tokenizer instance.
            max_length (int): Maximum sequence length for tokenization.
            ner_tags (Optional[List[str]]): List of NER tags.
            special_tokens (Optional[List[str]]): List of special tokens.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ner_tags = ner_tags or ["O"]
        self.special_tokens = special_tokens or ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]
        self.label_id_mapping = self._create_label_id_mapping(self.ner_tags + self.special_tokens)
        self.token_cache = {}  # Cache for tokenized outputs

        logger.info(f"Tokenizer initialized with max_length={self.max_length}")

    def _create_label_id_mapping(self, labels: List[str]) -> Dict[str, int]:
        """
        Creates a mapping from labels to IDs using ClassLabel.

        Args:
            labels (List[str]): List of labels.

        Returns:
            Dict[str, int]: Mapping from labels to IDs.
        """
        try:
            class_label = ClassLabel(names=labels)
            return {label: class_label.str2int(label) for label in labels}
        except Exception as e:
            logger.error(f"Error creating label mapping: {e}")
            raise

    def _tokenize_with_hf(self, tokens: List[str]) -> Dict:
        """
        Tokenizes input tokens using Hugging Face's tokenizer.

        Args:
            tokens (List[str]): List of tokens to tokenize.

        Returns:
            Dict: Tokenized output (input_ids, attention_mask).
        """
        return self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors="pt" if self.tokenizer.is_fast else "np",
        )

    def _align_labels_with_hf(self, tokens: List[str], labels: List[str]) -> Dict:
        """
        Aligns labels with tokenized tokens using Hugging Face's tokenizer.

        Args:
            tokens (List[str]): List of input tokens.
            labels (List[str]): List of labels corresponding to the input tokens.

        Returns:
            Dict: Tokenized tokens with aligned labels.
        """
        tokenized_inputs = self._tokenize_with_hf(tokens)
        word_ids = tokenized_inputs.word_ids() if self.tokenizer.is_fast else self._get_word_ids_from_slow_tokenizer(tokenized_inputs)

        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:  # Special tokens ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
            else:
                if word_id != previous_word_id:  # First token of a word
                    aligned_labels.append(self.label_id_mapping.get(labels[word_id], -100))
                    previous_word_id = word_id
                else:  # Subsequent tokens of the same word
                    aligned_labels.append(-100)

        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    def _get_word_ids_from_slow_tokenizer(self, tokenized_inputs) -> List[Optional[int]]:
        """
        Mimics the behavior of `word_ids()` for slow tokenizers.

        Args:
            tokenized_inputs: Tokenized output from the tokenizer.

        Returns:
            List[Optional[int]]: List of word IDs aligned with tokenized tokens.
        """
        word_ids = []
        current_word_id = 0
        for token in tokenized_inputs["input_ids"][0]:
            token = self.tokenizer.convert_ids_to_tokens(token.item())
            if token.startswith("##"):  # Subword token
                word_ids.append(current_word_id)
            elif token in self.tokenizer.all_special_tokens:  # Special token
                word_ids.append(None)
            else:  # New word
                word_ids.append(current_word_id)
                current_word_id += 1
        return word_ids

    def _tokenize_with_custom(self, tokens: List[str], labels: List[str]) -> Dict:
        """
        Tokenizes input tokens using a custom tokenizer (e.g., AmharicSegmenter).

        Args:
            tokens (List[str]): List of input tokens.
            labels (List[str]): List of labels corresponding to the input tokens.

        Returns:
            Dict: Tokenized tokens with aligned labels.
        """
        input_ids = [self.label_id_mapping["[CLS]"]]  # Start with [CLS]
        aligned_labels = [-100]  # Ignore [CLS] during loss calculation

        for word, label in zip(tokens, labels):
            if word in self.token_cache:  # Use cached tokenization if available
                subtokens = self.token_cache[word]
            else:
                subtokens = self.tokenizer.tokenize(word) if hasattr(self.tokenizer, "tokenize") else [word]
                self.token_cache[word] = subtokens  # Cache the tokenized output

            input_ids.extend([self.label_id_mapping.get(token, self.label_id_mapping["[UNK]"]) for token in subtokens])
            aligned_labels.extend([self.label_id_mapping.get(label, -100)] + [-100] * (len(subtokens) - 1))

        input_ids.append(self.label_id_mapping["[SEP]"])  # End with [SEP]
        aligned_labels.append(-100)  # Ignore [SEP] during loss calculation

        # Truncate and pad to max_length
        input_ids = input_ids[:self.max_length]
        aligned_labels = aligned_labels[:self.max_length]
        attention_mask = [1] * len(input_ids)

        padding_length = self.max_length - len(input_ids)
        input_ids.extend([self.label_id_mapping["[PAD]"]] * padding_length)
        aligned_labels.extend([-100] * padding_length)
        attention_mask.extend([0] * padding_length)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": aligned_labels,
        }
    
    def tokenize_and_align_labels(
        self,
        examples: Dict[str, Union[List, List[List]]],
        use_hf: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """
        Tokenizes and aligns labels for examples, handling both batched and non-batched inputs.

        Args:
            examples (Dict[str, Union[List, List[List]]]): A dictionary containing "tokens" and "labels".
            use_hf (bool): Whether to use Hugging Face tokenization logic.

        Returns:
            Dict[str, List[List[int]]]: Tokenized tokens and aligned labels, ready for model input.
        """
        try:
            # Check if input is batched
            is_batched = isinstance(examples["tokens"][0], list)

            # Prepare tokens and labels
            if is_batched:
                batch_tokens = examples["tokens"]
                batch_labels = examples["labels"]
            else:
                batch_tokens = [examples["tokens"]]
                batch_labels = [examples["labels"]]

            # Tokenize and align for each example
            tokenized_batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for tokens, labels in zip(batch_tokens, batch_labels):
                if use_hf:
                    result = self._align_labels_with_hf(tokens, labels)
                else:
                    result = self._tokenize_with_custom(tokens, labels)
                
                # Append results to the batch
                tokenized_batch["input_ids"].append(result["input_ids"])
                tokenized_batch["attention_mask"].append(result["attention_mask"])
                tokenized_batch["labels"].append(result["labels"])

            # If not batched, return single lists instead of lists of lists
            if not is_batched:
                tokenized_batch = {key: value[0] for key, value in tokenized_batch.items()}

            return tokenized_batch

        except Exception as e:
            logger.error(f"Error during tokenization and label alignment: {e}")
            raise
