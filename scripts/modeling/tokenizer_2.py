import os
import sys
import json
import joblib
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Generator

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
    Handles tokenization, alignment, and tokenizer initialization.
    Supports Hugging Face and custom tokenization methods.
    """

    def __init__(
        self,
        tokenizer: Union[AutoTokenizer, "AmharicSegmenter"],
        max_length: int = 128,
        ner_tags: Optional[List[str]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Initializes the Tokenizer with a tokenizer instance, max sequence length, and label mapping.

        Args:
            tokenizer (Union[AutoTokenizer, AmharicSegmenter]): Tokenizer instance.
            max_length (int): Maximum sequence length for tokenization.
            ner_tags (Optional[List[str]]): List of string tags.
            special_tokens (Optional[List[str]]): List of special tokens.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ner_tags = ner_tags or ["O"]
        self.special_tokens = special_tokens or ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]

        # Combine labels and special tokens into a single ClassLabel instance        
        all_labels = self.ner_tags + self.special_tokens
        label_id_mapping = self._create_label_id_mapping(all_labels)
        self.label_id_mapping = {label: label_id_mapping.str2int(label) for label in label_id_mapping.names}

        logger.info(f"Tokenizer initialized with max length: {self.max_length}")

    def _validate_inputs(self, tokens: List[str], labels: Optional[List[str]] = None) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Validates input tokens and labels.

        Args:
            tokens (List[str]): List of tokens.
            labels (Optional[List[str]]): NER labels corresponding to tokens.

        Returns:
            Tuple[List[List[str]], List[List[str]]]: Batched tokens and labels.

        Raises:
            ValueError: If tokens or labels are invalid.
        """
        if not tokens:
            raise ValueError("Tokens must not be empty.")
        if labels is not None:
            if not labels:
                raise ValueError("Labels must not be empty.")
        if isinstance(tokens[0], list) and isinstance(labels[0], list):  # Batched inputs
            if len(tokens) != len(labels):
                raise ValueError("Mismatched number of token and label sequences.")
        elif isinstance(tokens[0], str) and isinstance(labels[0], str):  # Single sequence
            tokens, labels = [tokens], [labels]
        else:
            raise ValueError("Tokens and labels must both be either lists or lists of lists.")

        return tokens, labels

    def _create_label_id_mapping(self, labels: List[str]) -> ClassLabel:
        """
        Creates a ClassLabel instance from the provided label mapping.

        Args:
            labels (List[str]): List of string labels.

        Returns:
            ClassLabel: A ClassLabel instance for label-to-ID mapping.

        Raises:
            ValueError: If the label list is empty.
        """
        if not labels:
            raise ValueError("Label list cannot be empty.")
        try:
            return ClassLabel(names=labels)
        except Exception as e:
            logger.error(f"Error when creating label mapping: {e}")
            raise

    def _set_label_id_mapping(self, labels: List[str]) -> ClassLabel:
        """
        Sets a ClassLabel instance to tokenizer instance from the provided label mapping.
        """
        if not labels:
            raise ValueError("Label list cannot be empty.")
        try:
            self.label_id_mapping = self._create_label_id_mapping(labels)
            return self.label_id_mapping
        except Exception as e:
            logger.error(f"Error when creating label mapping: {e}")
            raise

    def _tokenize_input(self, tokens: List[str]) -> Dict:
        """
        Tokenizes input tokens using Hugging Face's tokenizer.

        Args:
            tokens (List[str]): List of tokens to tokenize.

        Returns:
            Dict: Tokenized output from Hugging Face tokenizer.
        """
        return self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors="pt" if self.tokenizer.is_fast else "np",  # NumPy fallback
        )

    def _get_word_ids_from_slow_tokenizer(self, tokenized_inputs) -> List[Optional[int]]:
        """
        Mimics the behavior of `word_ids()` for slow tokenizers.
        This method aligns word indices with tokenized inputs for non-fast tokenizers.

        Args:
            tokenized_inputs: Tokenized output from the tokenizer.

        Returns:
            List[Optional[int]]: List of word IDs aligned with tokenized tokens.
        """
        word_ids = []
        current_word_id = 0
        for token in tokenized_inputs["input_ids"][0]:
            token = self.tokenizer.convert_ids_to_tokens(token.item())  # Convert token ID to string
            if token.startswith("##"):  # Subword token
                word_ids.append(current_word_id)
            elif token in self.tokenizer.all_special_tokens:  # Special token
                word_ids.append(None)
            else:  # New word
                word_ids.append(current_word_id)
                current_word_id += 1
        return word_ids

    def align_labels_with_tokens_hf(self, tokens: List[str], labels: List[str]) -> Dict:
        """
        Aligns labels with tokenized tokens and returns aligned tokens, labels, and Hugging Face labels.

        Args:
            tokens (List[str]): List of input tokens.
            labels (List[str]): List of labels corresponding to the input tokens.

        Returns:
            Dict: Tokenized tokens with aligned labels.
        """
        # Tokenize the input tokens
        tokenized_inputs = self._tokenize_input(tokens)

        # Get word IDs for each token
        word_ids = tokenized_inputs.word_ids() if self.tokenizer.is_fast else self._get_word_ids_from_slow_tokenizer(tokenized_inputs)

        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:  # Special tokens ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)  # Ignore special tokens during loss calculation
            else:
                if word_id != previous_word_id:  # First token of a word
                    # Map label to ID or use -100
                    aligned_labels.append(self.label_id_mapping.get(labels[word_id], -100))
                    previous_word_id = word_id
                else:  # Subsequent tokens of the same word
                    aligned_labels.append(-100)  # Ignore during loss calculation

        # Add Hugging Face-compatible labels to tokenized_tokens
        tokenized_inputs["labels"] = aligned_labels

        return tokenized_inputs

    def _custom_tokenize_and_align_labels(
        self, tokens: List[List[str]], labels: List[List[str]], max_length: int
    ) -> Generator[Dict[str, List[int]], None, None]:
        """
        Tokenizes sentences with a custom segmenter and aligns labels for NER task.
        Converts tokens to input IDs using the label mapping.

        Args:
            tokens (List[List[str]]): Nested list of sentences tokenized as words.
            labels (List[List[str]]): Nested list of corresponding labels for sentences.
            max_length (int): Maximum sequence length for the model.

        Yields:
            Dict[str, List[int]]: A dictionary with input_ids, attention_mask, and labels.
        """
        # Cache for tokenized words to avoid redundant tokenization
        token_cache = {}

        # Process sentences in batches
        for sentence_tokens, sentence_labels in zip(tokens, labels):
            segmented_tokens = []
            aligned_labels = []

            # Tokenize all words in the sentence (batch tokenization if supported)
            if hasattr(self.tokenizer, "batch_tokenize"):
                subtoken_batches = self.tokenizer.batch_tokenize(sentence_tokens)
            else:
                subtoken_batches = []
                for word in sentence_tokens:
                    if word in token_cache:
                        subtokens = token_cache[word]
                    else:
                        subtokens = (
                            self.tokenizer.amharic_tokenizer(word) if hasattr(self.tokenizer, "amharic_tokenizer")
                            else self.tokenizer.tokenize(word)
                        )
                        token_cache[word] = subtokens
                    subtoken_batches.append(subtokens)

            # Align labels and tokens
            for word, label, subtokens in zip(sentence_tokens, sentence_labels, subtoken_batches):
                segmented_tokens.extend(subtokens)
                aligned_labels.extend([label] + ["O"] * (len(subtokens) - 1))
            
            # Convert tokens to input IDs
            input_ids = [self.label_id_mapping.get("[CLS]", 0)] + \
                        [self.label_id_mapping.get(token, self.label_id_mapping.get("[UNK]", 0)) for token in segmented_tokens] + \
                        [self.label_id_mapping.get("[SEP]", 0)]

            # Align labels and convert to IDs
            aligned_labels = [-100] + [self.label_id_mapping.get(label, -100) for label in aligned_labels] + [-100]

            # Generate attention mask
            attention_mask = [1] * len(input_ids)

            # Truncate and pad to max_length using NumPy for vectorized operations
            input_ids = np.array(input_ids[:max_length], dtype=np.int32)
            attention_mask = np.array(attention_mask[:max_length], dtype=np.int32)
            aligned_labels = np.array(aligned_labels[:max_length], dtype=np.int32)

            padding_length = max_length - len(input_ids)
            input_ids = np.pad(input_ids, (0, padding_length), constant_values=self.label_id_mapping.get("[PAD]", 0))
            attention_mask = np.pad(attention_mask, (0, padding_length), constant_values=0)
            aligned_labels = np.pad(aligned_labels, (0, padding_length), constant_values=-100)

            # Yield processed data as a dictionary
            yield {
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist(),
                "labels": aligned_labels.tolist()
            }

    def tokenize_and_align_labels(
        self,
        tokens: Union[List[str], List[List[str]]],
        labels: Union[List[str], List[List[str]]],
        use_hf: bool = False,
    ) -> Dict[str, List[List[int]]]:
        """
        Unified method to tokenize and align labels based on the tokenizer type.

        Args:
            tokens (Union[List[str], List[List[str]]]): Input tokens or tokenized sentences.
            labels (Union[List[str], List[List[str]]]): NER labels corresponding to tokens.
            use_hf (bool): Whether to use Hugging Face tokenization logic.

        Returns:
            Dict[str, List[List[int]]]: Tokenized tokens and aligned labels, ready for model input.
        """
        try:
            # Validate inputs
            tokens, labels = self._validate_inputs(tokens, labels)

            if use_hf:
                aligned_data = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": []
                }
                for sentence_tokens, sentence_labels in zip(tokens, labels):
                    tokenized_inputs = self.align_labels_with_tokens_hf(sentence_tokens, sentence_labels)
                    aligned_data["input_ids"].append(tokenized_inputs["input_ids"][0].tolist())
                    aligned_data["attention_mask"].append(tokenized_inputs["attention_mask"][0].tolist())
                    aligned_data["labels"].append(tokenized_inputs["labels"])
                return aligned_data
            else:
                # Collect generator output into a single dictionary
                aligned_data = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": []
                }
                for processed_sentence in self._custom_tokenize_and_align_labels(tokens, labels, self.max_length):
                    aligned_data["input_ids"].append(processed_sentence["input_ids"])
                    aligned_data["attention_mask"].append(processed_sentence["attention_mask"])
                    aligned_data["labels"].append(processed_sentence["labels"])
                return aligned_data

        except Exception as e:
            logger.error(f"Error during tokenization and label alignment: {e}")
            raise
