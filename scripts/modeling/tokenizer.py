import os
import sys
from typing import List, Tuple, Union, Dict, Optional

import sentencepiece as spm
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

    def __init__(self, tokenizer: Union[AutoTokenizer, "AmharicSegmenter"], max_length: int = 128, label_id_mapping: Optional[Dict[str, int]] = None):
        """
        Initializes the Tokenizer with a tokenizer instance, max sequence length, and label mapping.

        Args:
            tokenizer (Union[AutoTokenizer, AmharicSegmenter]): Tokenizer instance.
            max_length (int): Maximum sequence length for tokenization.
            label_id_mapping (Optional[Dict[str, int]]): Mapping of string labels to integer IDs.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_id_mapping = label_id_mapping or {"O": 0}  # Default mapping for "O"
        logger.info(f"Tokenizer initialized with max length: {self.max_length}")

    def _validate_inputs(self, tokens: List[str], labels: Optional[List[str]] = None) -> None:
        """
        Validates input tokens and labels.

        Args:
            tokens (List[str]): List of tokens.
            labels (Optional[List[str]): NER labels corresponding to tokens.

        Raises:
            ValueError: If tokens or labels are invalid.
        """
        if not tokens:
            raise ValueError("Tokens must not be empty.")
        if labels is not None:
            if not labels:
                raise ValueError("Labels must not be empty.")
            if len(tokens) != len(labels):  # Check against the first list of labels
                raise ValueError("Tokens and labels must have the same length.")

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

    def _convert_labels_to_ids(self, labels: List[str]) -> List[int]:
        """
        Converts string labels to integer IDs using the label_id_mapping.

        Args:
            labels (List[str]): List of string labels.

        Returns:
            List[int]: List of integer label IDs.
        """
        return [self.label_id_mapping.get(label, -100) for label in labels]

    def align_labels_with_tokens_hf(self, tokens: List[str], labels: List[str]) -> Tuple[List[str], List[str], List[int]]:
        """
        Aligns labels with tokenized tokens and returns aligned tokens, labels, and Hugging Face labels.

        Args:
            tokens (List[str]): List of input tokens.
            labels (List[str]): List of labels corresponding to the input tokens.

        Returns:
            Tuple[List[str], List[str], List[int]]: Aligned tokens, aligned labels, and Hugging Face labels.
        """
        # Tokenize the input tokens
        tokenized_inputs = self._tokenize_input(tokens)
        
        # Get word IDs for each token
        word_ids = tokenized_inputs.word_ids() if self.tokenizer.is_fast else self._get_word_ids_from_slow_tokenizer(tokenized_inputs)

        aligned_tokens, aligned_labels, hf_labels = [], [], []
        previous_word_idx = None
        input_ids = tokenized_inputs["input_ids"][0]  # Access the first sequence (batch size = 1)

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:  # Special tokens ([CLS], [SEP], [PAD])
                aligned_tokens.append(self.tokenizer.convert_ids_to_tokens(input_ids[idx].item()))  # Convert token ID to token
                # aligned_tokens.append(self.tokenizer.convert_ids_to_tokens(input_ids[idx].row()))  # Convert token ID to token
                aligned_labels.append("O")  # Use "O" for special tokens
                hf_labels.append(-100)  # Ignore special tokens during loss calculation
            
            else:
                if word_idx != previous_word_idx:  # First token of a word
                    aligned_tokens.append(tokens[word_idx])
                    aligned_labels.append(labels[word_idx])
                    hf_labels.append(self.label_id_mapping.get(labels[word_idx], -100))  # Map label to ID or use -100

                else:  # Subsequent tokens of the same word
                    aligned_tokens.append(self.tokenizer.convert_ids_to_tokens(input_ids[idx].item()))
                    # aligned_tokens.append(self.tokenizer.convert_ids_to_tokens(input_ids[idx].row()))
                    aligned_labels.append(labels[word_idx])  # Inherit the label from the first token
                    hf_labels.append(-100)  # Ignore during loss calculation

            previous_word_idx = word_idx

        # Add Hugging Face-compatible labels to tokenized_tokens
        tokenized_inputs["labels"] = hf_labels

        return aligned_tokens, aligned_labels, tokenized_inputs

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
        current_word_idx = 0
        for token in tokenized_inputs["input_ids"][0]:
            token = self.tokenizer.convert_ids_to_tokens(token.item())  # Convert token ID to string
            if token.startswith("##"):  # Subword token
                word_ids.append(current_word_idx)
            elif token in self.tokenizer.all_special_tokens:  # Special token
                word_ids.append(None)
            else:  # New word
                word_ids.append(current_word_idx)
                current_word_idx += 1
        return word_ids


    def align_labels_with_tokens_generic(self, tokens: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """
        Aligns the original labels with the tokenized tokens.

        Args:
            tokens (List[str]): Original tokens.
            labels (List[str]): Labels corresponding to the original tokens.

        Returns:
            Tuple[List[str], List[str]]: Aligned tokens and aligned labels.

        Raises:
            Exception: If label alignment fails.
        """
        try:
            aligned_labels, aligned_tokens = [], []
            token_idx = 0

            for token, label in zip(tokens, labels):
                # Tokenize the current token to get its subtokens
                if hasattr(self.tokenizer, "amharic_tokenizer"):
                    subtokens = self.tokenizer.amharic_tokenizer(token)
                else:
                    subtokens = self.tokenizer.tokenize(token)

                # Add subtokens to aligned_tokens
                aligned_tokens.extend(subtokens)

                # Assign the label to the first subtoken and "O" to subsequent subtokens
                aligned_labels.append(label)
                aligned_labels.extend(["O"] * (len(subtokens) - 1))

                # Update token_idx to point to the next token
                token_idx += len(subtokens)

            logger.info("Tokens and labels aligned successfully.")
            return aligned_tokens, aligned_labels

        except Exception as e:
            logger.error(f"Error during label alignment: {e}")
            raise

    def tokenize_and_align_labels(self, tokens: List[str], labels: List[str], use_hf: bool = False) -> Union[Tuple[List[str], List[str]], Dict]:
        """
        Unified method to tokenize and align labels based on the selected approach.

        Args:
            tokens (List[str]): List of tokens.
            labels (List[str]): NER labels corresponding to tokens.
            use_hf (bool): Whether to use the Hugging Face approach.

        Returns:
            Union[Tuple[List[str], List[str]], Dict]: Tokenized tokens and aligned labels or Hugging Face-compatible inputs.

        Raises:
            Exception: If tokenization or alignment fails.
        """
        try:
            # Validate inputs
            self._validate_inputs(tokens, labels)

            if use_hf:
                return self.align_labels_with_tokens_hf(tokens, labels)
            else:
                aligned_tokens, aligned_labels = self.align_labels_with_tokens_generic(tokens, labels)
                return aligned_tokens, aligned_labels, aligned_tokens  # Return tokenized tokens for consistency
        except Exception as e:
            logger.error(f"Error during tokenization and label alignment: {e}")
            raise
