import os
import sys
import unittest
import numpy as np
from transformers import AutoTokenizer

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.modeling.tokenizer import Tokenizer

logger = setup_logger("test")

class TestTokenizer(unittest.TestCase):
    """
    Unit tests for the Tokenizer class.
    """

    def setUp(self):
        """
        Sets up the test environment by initializing a Tokenizer instance.
        """
        self.tokenizer = Tokenizer(
            tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
            max_length=128,
            ner_tags=["O", "B-PER", "I-PER"],
            special_tokens=["[CLS]", "[SEP]", "[PAD]", "[UNK]"],
        )

    def test_tokenize_and_align_labels_hf(self):
        """
        Tests the tokenize_and_align_labels method with use_hf set to True.
        """
        tokens = ["Hello", "world"]
        labels = ["O", "O"]

        try:
            tokenized_data = self.tokenizer.tokenize_and_align_labels(tokens, labels, use_hf=True)
            self.assertIn("input_ids", tokenized_data)
            self.assertIn("attention_mask", tokenized_data)
            self.assertIn("labels", tokenized_data)
            self.assertEqual(len(tokenized_data["input_ids"]), 1)  # Single sentence
            self.assertEqual(len(tokenized_data["input_ids"][0]), self.tokenizer.max_length)  # Padded to max_length
        except Exception as e:
            logger.error(f"Failed to tokenize and align labels: {e}")
            self.fail("Tokenization and label alignment failed")

    def test_tokenize_and_align_labels_non_hf(self):
        """
        Tests the tokenize_and_align_labels method with use_hf set to False.
        """
        tokens = ["Hello", "world"]
        labels = ["O", "O"]

        try:
            tokenized_data = self.tokenizer.tokenize_and_align_labels(tokens, labels, use_hf=False)
            self.assertIn("input_ids", tokenized_data)
            self.assertIn("attention_mask", tokenized_data)
            self.assertIn("labels", tokenized_data)
            self.assertEqual(len(tokenized_data["input_ids"]), 1)  # Single sentence
            self.assertEqual(len(tokenized_data["input_ids"][0]), self.tokenizer.max_length)  # Padded to max_length
        except Exception as e:
            logger.error(f"Failed to tokenize and align labels: {e}")
            self.fail("Tokenization and label alignment failed")

    def test_validate_inputs(self):
        """
        Tests the _validate_inputs method by attempting to validate invalid inputs.
        """
        # Test empty tokens
        with self.assertRaises(ValueError):
            self.tokenizer._validate_inputs([], ["O"])

        # Test empty labels
        with self.assertRaises(ValueError):
            self.tokenizer._validate_inputs(["Hello", "world"], [])

        # Test mismatched token and label sequences
        with self.assertRaises(ValueError):
            self.tokenizer._validate_inputs([["Hello", "world"]], [["O"]])

        # Test valid inputs
        tokens = ["Hello", "world"]
        labels = ["O", "O"]
        validated_tokens, validated_labels = self.tokenizer._validate_inputs(tokens, labels)
        self.assertEqual(validated_tokens, [tokens])
        self.assertEqual(validated_labels, [labels])

    def test_get_id(self):
        """
        Tests the _get_id method for label-to-ID conversion.
        """
        # Test valid labels
        self.assertEqual(self.tokenizer.label_id_mapping.get("O"), 0)
        self.assertEqual(self.tokenizer.label_id_mapping.get("B-PER"), 1)
        self.assertEqual(self.tokenizer.label_id_mapping.get("I-PER"), 2)

        # Test unknown label
        with self.assertRaises(ValueError):
            self.tokenizer.label_id_mapping.get("UNKNOWN")

    def test_align_labels_with_tokens_hf(self):
        """
        Tests the align_labels_with_tokens_hf method.
        """
        tokens = ["Hello", "world"]
        labels = ["O", "O"]

        tokenized_data = self.tokenizer.align_labels_with_tokens_hf(tokens, labels)
        self.assertIn("input_ids", tokenized_data)
        self.assertIn("attention_mask", tokenized_data)
        self.assertIn("labels", tokenized_data)

        # Check that special tokens are ignored (label = -100)
        self.assertEqual(tokenized_data["labels"][0], -100)  # [CLS]
        self.assertEqual(tokenized_data["labels"][-1], -100)  # [SEP]

    def test_custom_tokenize_and_align_labels(self):
        """
        Tests the _custom_tokenize_and_align_labels method.
        """
        tokens = [["Hello", "world"], ["I", "am", "Alice"]]
        labels = [["O", "O"], ["O", "O", "B-PER"]]

        # Process data using the custom tokenizer
        processed_data = list(self.tokenizer._custom_tokenize_and_align_labels(tokens, labels, self.tokenizer.max_length))

        # Check the output format
        self.assertEqual(len(processed_data), 2)  # Two sentences
        for data in processed_data:
            self.assertIn("input_ids", data)
            self.assertIn("attention_mask", data)
            self.assertIn("labels", data)
            self.assertEqual(len(data["input_ids"]), self.tokenizer.max_length)  # Padded to max_length

    def test_save_and_load_preprocessed_data(self):
        """
        Tests the save_preprocessed_data and load_preprocessed_data methods.
        """
        tokens = [["Hello", "world"], ["I", "am", "Alice"]]
        labels = [["O", "O"], ["O", "O", "B-PER"]]

        # Tokenize and align labels
        processed_data = self.tokenizer.tokenize_and_align_labels(tokens, labels, use_hf=False)

        # Save data to a file
        filepath = "test_data.json"
        self.tokenizer.save_preprocessed_data(processed_data, filepath, format="json")

        # Load data from the file
        loaded_data = self.tokenizer.load_preprocessed_data(filepath, format="json")

        # Check that the loaded data matches the original data
        self.assertEqual(processed_data, loaded_data)

        # Clean up the test file
        os.remove(filepath)

    def test_edge_cases(self):
        """
        Tests edge cases such as empty inputs, long sequences, and unknown labels.
        """
        # Test empty inputs
        with self.assertRaises(ValueError):
            self.tokenizer.tokenize_and_align_labels([], [])

        # Test long sequences (exceeding max_length)
        long_tokens = ["word"] * 200
        long_labels = ["O"] * 200
        processed_data = self.tokenizer.tokenize_and_align_labels(long_tokens, long_labels, use_hf=True)
        self.assertEqual(len(processed_data["input_ids"][0]), self.tokenizer.max_length)  # Truncated to max_length

        # Test unknown labels
        tokens = ["Hello", "world"]
        labels = ["O", "UNKNOWN"]
        processed_data = self.tokenizer.tokenize_and_align_labels(tokens, labels, use_hf=False)
        self.assertEqual(processed_data["labels"][0][1], -100)  # Unknown label mapped to -100

# if __name__ == "__main__":
#     unittest.main()