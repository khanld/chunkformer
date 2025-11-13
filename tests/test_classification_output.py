#!/usr/bin/env python3
"""
Test classification output format for ChunkFormer classification models.
"""

from pathlib import Path

import pytest

from chunkformer import ChunkFormerModel


class TestClassificationOutput:
    """Test cases for classification output format validation."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.model_name = "khanhld/chunkformer-gender-emotion-dialect-age-classification"
        cls.model = ChunkFormerModel.from_pretrained(cls.model_name)
        cls.data_dir = Path(__file__).parent.parent

        # Use sample audio from ASR tests if available
        cls.sample_audio_dir = cls.data_dir / "samples" / "audios"

        # Find first available audio file
        cls.test_audio_path = None
        if cls.sample_audio_dir.exists():
            for audio_file in cls.sample_audio_dir.glob("*.wav"):
                if audio_file.exists():
                    cls.test_audio_path = str(audio_file)
                    break

        # If not found, try samples directory directly
        if cls.test_audio_path is None:
            sample_dir_alt = cls.data_dir / "samples"
            if sample_dir_alt.exists():
                for audio_file in sample_dir_alt.glob("*.wav"):
                    if audio_file.exists():
                        cls.test_audio_path = str(audio_file)
                        break

        # Assert that we found an audio file
        assert (
            cls.test_audio_path is not None
        ), f"No test audio file found in {cls.sample_audio_dir} or {cls.data_dir / 'samples'}"

        print(f"Using test audio: {cls.test_audio_path}")

    def test_model_is_classification(self):
        """Test that the model is correctly identified as a classification model."""
        assert self.model.is_classification, "Model should be identified as classification model"

    def test_model_has_tasks(self):
        """Test that the model has classification tasks defined."""
        tasks = self.model.get_tasks()
        assert tasks is not None, "Model should have tasks defined"
        assert len(tasks) > 0, "Model should have at least one task"

        # Expected tasks for this model
        expected_tasks = ["gender", "emotion", "dialect", "age"]
        for task in expected_tasks:
            assert task in tasks, f"Task '{task}' should be in model tasks"

        print(f"Model tasks: {list(tasks.keys())}")
        for task_name, num_classes in tasks.items():
            print(f"  {task_name}: {num_classes} classes")

    def test_model_has_label_mapping(self):
        """Test that the model has label mapping loaded."""
        assert self.model.label_mapping is not None, "Model should have label_mapping loaded"

        # Check that label mapping has all tasks
        tasks = self.model.get_tasks()
        for task_name in tasks.keys():
            assert (
                task_name in self.model.label_mapping
            ), f"Task '{task_name}' should be in label_mapping"

            # Check that label mapping uses id:label format
            task_mapping = self.model.label_mapping[task_name]
            assert isinstance(task_mapping, dict), f"Label mapping for {task_name} should be a dict"

            # Check that keys are string IDs
            for key in task_mapping.keys():
                assert isinstance(
                    key, str
                ), f"Label mapping keys should be strings, got {type(key)}"
                assert key.isdigit(), f"Label mapping keys should be numeric strings, got '{key}'"

        print("Label mapping loaded successfully")
        for task_name, mapping in self.model.label_mapping.items():
            print(f"  {task_name}: {len(mapping)} labels")

    def test_classification_output_format(self):
        """Test that classification output has the correct format."""
        print(f"Testing with audio: {self.test_audio_path}")

        # Perform classification
        result = self.model.classify_audio(
            audio_path=self.test_audio_path,
            chunk_size=-1,  # Full attention
            left_context_size=-1,
            right_context_size=-1,
        )

        # Verify result is a dictionary
        assert isinstance(result, dict), f"Result should be a dict, got {type(result)}"

        # Verify each task is in the result
        tasks = self.model.get_tasks()
        for task_name in tasks.keys():
            assert task_name in result, f"Task '{task_name}' should be in result"

            task_result = result[task_name]

            # Verify task result structure
            assert isinstance(
                task_result, dict
            ), f"Result for {task_name} should be a dict, got {type(task_result)}"

            # Verify required keys
            required_keys = ["label", "label_id", "prob"]
            for key in required_keys:
                assert key in task_result, f"Key '{key}' should be in result for task '{task_name}'"

            # Verify types
            assert isinstance(
                task_result["label"], str
            ), f"label should be str, got {type(task_result['label'])}"
            assert isinstance(
                task_result["label_id"], int
            ), f"label_id should be int, got {type(task_result['label_id'])}"
            assert isinstance(
                task_result["prob"], float
            ), f"prob should be float, got {type(task_result['prob'])}"

            # Verify probability is in valid range
            assert (
                0.0 <= task_result["prob"] <= 1.0
            ), f"prob should be in [0, 1], got {task_result['prob']}"

            # Verify label_id matches label_mapping
            label_id_str = str(task_result["label_id"])
            assert (
                label_id_str in self.model.label_mapping[task_name]
            ), f"label_id {label_id_str} not found in label_mapping for {task_name}"

            expected_label = self.model.label_mapping[task_name][label_id_str]
            assert (
                task_result["label"] == expected_label
            ), f"label mismatch: got '{task_result['label']}', expected '{expected_label}'"

        # Print results
        print("\nClassification Results:")
        print("=" * 70)
        for task_name, task_result in result.items():
            print(f"{task_name.capitalize()}:")
            print(f"  Label: {task_result['label']}")
            print(f"  Label ID: {task_result['label_id']}")
            print(f"  Probability: {task_result['prob']:.4f}")

    def test_classification_with_chunking(self):
        """Test classification with chunking enabled."""
        # Perform classification with chunking
        result = self.model.classify_audio(
            audio_path=self.test_audio_path,
            chunk_size=64,
            left_context_size=128,
            right_context_size=128,
        )

        # Verify output format is still correct
        assert isinstance(result, dict), "Result should be a dict"

        tasks = self.model.get_tasks()
        for task_name in tasks.keys():
            assert task_name in result, f"Task '{task_name}' should be in result"
            task_result = result[task_name]

            # Verify structure
            assert "label" in task_result
            assert "label_id" in task_result
            assert "prob" in task_result

            # Verify types and ranges
            assert isinstance(task_result["label"], str)
            assert isinstance(task_result["label_id"], int)
            assert isinstance(task_result["prob"], float)
            assert 0.0 <= task_result["prob"] <= 1.0

        print("\nClassification with chunking completed successfully")

    def test_probabilities_always_returned(self):
        """Test that probabilities are always returned (no parameter needed)."""
        # Call classify_audio without any probability parameter
        result = self.model.classify_audio(
            audio_path=self.test_audio_path,
            chunk_size=-1,
        )

        # Verify probabilities are present
        for task_name, task_result in result.items():
            assert (
                "prob" in task_result
            ), f"Probability should always be present for task '{task_name}'"
            assert (
                task_result["prob"] > 0.0
            ), f"Probability should be > 0 for predicted class in task '{task_name}'"

        print("\nProbabilities are always returned by default")

    def test_multiple_audio_consistency(self):
        """Test that classification is consistent across multiple calls."""
        # Classify the same audio twice
        result1 = self.model.classify_audio(
            audio_path=self.test_audio_path,
            chunk_size=-1,
        )

        result2 = self.model.classify_audio(
            audio_path=self.test_audio_path,
            chunk_size=-1,
        )

        # Results should be identical
        assert result1.keys() == result2.keys(), "Tasks should be identical"

        for task_name in result1.keys():
            assert (
                result1[task_name]["label_id"] == result2[task_name]["label_id"]
            ), f"Predicted label_id should be consistent for task '{task_name}'"

            # Probabilities should be very close (allowing for minor floating point differences)
            prob_diff = abs(result1[task_name]["prob"] - result2[task_name]["prob"])
            assert prob_diff < 1e-6, f"Probabilities should be consistent for task '{task_name}'"

        print("\nClassification is consistent across multiple calls")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
