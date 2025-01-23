import os
import sys
import numpy as np
from functools import lru_cache
from typing import Dict, List, Optional, Union, Generator

import sentencepiece as spm
from datasets import ClassLabel
from transformers import AutoTokenizer
from amseg.amharicSegmenter import AmharicSegmenter

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("tokenizer")

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

    @lru_cache(maxsize=10000)  # Cache for frequently occurring words
    def _tokenize_with_cache(self, word: str) -> List[str]:
        """Tokenize with cache to optimize performance for frequently occurring words."""
        if hasattr(self.tokenizer, "batch_tokenize"):
            return self.tokenizer.batch_tokenize(word)
        if hasattr(self.tokenizer, "amharic_tokenizer"):
            return self.tokenizer.amharic_tokenizer(word)
        elif hasattr(self.tokenizer, "tokenize"):
            return self.tokenizer.tokenize(word)
        return [word]

    def _process_single_instance(self, tokens: List[str], labels: List[str]) -> Dict: #Generator[Dict[str, List[int]], None, None]:
        """
        Processes a single input token-label pair using a custom tokenizer.

        Args:
            tokens (List[str]): List of input tokens.
            labels (List[str]): List of labels corresponding to the tokens.

        Returns:
            Dict: Tokenized tokens with aligned labels.
        """
        input_ids = [self.label_id_mapping["[CLS]"]]  # Start with [CLS]
        aligned_labels = [-100]  # Ignore [CLS] during loss calculation

        for token, label in zip(tokens, labels):
            subtokens = self._tokenize_with_cache(token)
            input_ids.extend(
                [self.label_id_mapping.get(subtoken, self.label_id_mapping["[UNK]"]) for subtoken in subtokens]
            )
            aligned_labels.extend([self.label_id_mapping.get(label, -100)] + [-100] * (len(subtokens) - 1))

        input_ids.append(self.label_id_mapping["[SEP]"])  # End with [SEP]
        aligned_labels.append(-100)  # Ignore [SEP] during loss calculation

        # Truncate and pad to max_length
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)

        if padding_length > 0:
            input_ids.extend([self.label_id_mapping["[PAD]"]] * padding_length)
            aligned_labels.extend([-100] * padding_length)
            attention_mask.extend([0] * padding_length)
            
            # input_ids = np.pad(input_ids, (0, padding_length), constant_values=self.label_id_mapping.get("[PAD]", 0)).tolist()
            # attention_mask = np.pad(attention_mask, (0, padding_length), constant_values=0).tolist()
            # aligned_labels = np.pad(aligned_labels, (0, padding_length), constant_values=-100).tolist()

        else:
            input_ids = input_ids[:self.max_length]
            aligned_labels = aligned_labels[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

            # input_ids = np.array(input_ids[:self.max_length], dtype=np.int32).tolist()
            # attention_mask = np.array(attention_mask[:self.max_length], dtype=np.int32).tolist()
            # aligned_labels = np.array(aligned_labels[:self.max_length], dtype=np.int32).tolist()

        # yield
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": aligned_labels,
        }

    def _tokenize_with_custom(self, tokens: List[List[str]], labels: List[List[str]]) -> Dict: # Generator[Dict[str, List[int]], None, None]:
        """
        Tokenizes input tokens and aligns labels using a custom tokenizer for batched inputs.

        Args:
            tokens (List[List[str]]): Batched list of input tokens.
            labels (List[List[str]]): Batched list of labels.

        Returns:
            Dict[str, List[List[int]]]: Tokenized tokens with aligned labels for each batch.
        """
        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for token_list, label_list in zip(tokens, labels):
            processed = self._process_single_instance(token_list, label_list)
            input_ids_batch.append(processed["input_ids"])
            attention_mask_batch.append(processed["attention_mask"])
            labels_batch.append(processed["labels"])
        
        # yield
        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }

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

    def _align_labels_with_hf(self, tokens: List[List[str]], labels: List[List[str]]) -> Dict:
        """
        Aligns labels with tokenized tokens using Hugging Face's tokenizer for batched inputs.

        Args:
            tokens (List[List[str]]): Batched list of input tokens.
            labels (List[List[str]]): Batched list of labels corresponding to the input tokens.

        Returns:
            Dict: Tokenized tokens with aligned labels for each batch.
        """
        tokenized_batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for token_list, label_list in zip(tokens, labels):
            tokenized_inputs = self._tokenize_with_hf(token_list)
            word_ids = tokenized_inputs.word_ids() if self.tokenizer.is_fast else self._get_word_ids_from_slow_tokenizer(tokenized_inputs)

            aligned_labels = np.full(len(word_ids), -100)
            previous_word_id = None

            for i, word_id in enumerate(word_ids):
                if word_id is not None and word_id != previous_word_id:
                    aligned_labels[i] = self.label_id_mapping.get(label_list[word_id], -100)
                    previous_word_id = word_id

            tokenized_inputs["labels"] = aligned_labels.tolist()
            tokenized_batch["input_ids"].append(tokenized_inputs["input_ids"])
            tokenized_batch["attention_mask"].append(tokenized_inputs["attention_mask"])
            tokenized_batch["labels"].append(tokenized_inputs["labels"])

        return tokenized_batch

    def tokenize_and_align_labels(
        self,
        examples: Dict[str, Union[List, List[List]]],
        use_hf: bool = False,
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

            if use_hf:
                tokenized_batch = self._align_labels_with_hf(batch_tokens, batch_labels)
            else:
                tokenized_batch = self._tokenize_with_custom(batch_tokens, batch_labels)

            # If not batched, return single lists instead of lists of lists
            if not is_batched:
                tokenized_batch = {key: value[0] for key, value in tokenized_batch.items()}

            return tokenized_batch

        except Exception as e:
            logger.error(f"Error during tokenization and label alignment: {e}")
            raise