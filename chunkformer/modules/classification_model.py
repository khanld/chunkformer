# Copyright (c) 2024 ChunkFormer Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Speech Classification Model using ChunkFormer encoder."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from chunkformer.modules.encoder import ChunkFormerEncoder


class ClassificationHead(torch.nn.Module):
    """Simple linear classification head with optional dropout."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            input_size: Input feature dimension
            num_classes: Number of output classes
            dropout_rate: Dropout rate before classification layer
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear = torch.nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_size)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.dropout(x)
        return self.linear(x)


class SpeechClassificationModel(torch.nn.Module):
    """Speech Classification Model using ChunkFormer encoder.

    Supports both single-task and multi-task classification.
    Uses average pooling over encoder outputs for feature aggregation.
    """

    def __init__(
        self,
        encoder: ChunkFormerEncoder,
        tasks: Dict[str, int],
        dropout_rate: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            encoder: ChunkFormer encoder
            tasks: Dictionary mapping task names to number of classes
                   e.g., {'gender': 2, 'emotion': 7, 'region': 5}
                   For single task: {'gender': 2}
            dropout_rate: Dropout rate for classification heads
            label_smoothing: Label smoothing factor (0.0 = no smoothing, typically 0.1)
        """
        super().__init__()

        if not tasks:
            raise ValueError("At least one classification task must be defined")

        self.encoder = encoder
        self.tasks = tasks
        self.task_names = list(tasks.keys())
        self.num_tasks = len(tasks)
        self.label_smoothing = label_smoothing

        # Create classification head for each task
        encoder_output_size = encoder.output_size()
        self.classification_heads = torch.nn.ModuleDict(
            {
                task_name: ClassificationHead(
                    input_size=encoder_output_size,
                    num_classes=num_classes,
                    dropout_rate=dropout_rate,
                )
                for task_name, num_classes in tasks.items()
            }
        )

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            batch: Dictionary containing:
                - feats: (batch, time, feat_dim)
                - feats_lengths: (batch,)
                - {task_name}_label: (batch,) for each task
            device: Device to run on

        Returns:
            Dictionary containing:
                - loss: Total loss (averaged across tasks)
                - loss_{task_name}: Loss for each task
                - acc_{task_name}: Accuracy for each task
                - logits_{task_name}: Logits for each task
        """
        speech = batch["feats"].to(device)
        speech_lengths = batch["feats_lengths"].to(device)

        # 1. Encoder forward
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        # encoder_out: (batch, time, encoder_dim)
        # encoder_mask: (batch, 1, time)

        # 2. Average pooling over time dimension
        pooled_features = self._average_pooling(encoder_out, encoder_mask)
        # pooled_features: (batch, encoder_dim)

        # 3. Classification for each task
        outputs = {}
        total_loss = 0.0
        num_valid_tasks = 0

        for task_name in self.task_names:
            label_key = f"{task_name}_label"

            # Skip task if labels not provided (useful for multi-task with partial labels)
            if label_key not in batch:
                continue

            labels = batch[label_key].to(device)

            # Get logits from classification head
            logits = self.classification_heads[task_name](pooled_features)

            # Compute cross-entropy loss with label smoothing
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean()

            # Store outputs
            outputs[f"loss_{task_name}"] = loss
            outputs[f"acc_{task_name}"] = accuracy

            total_loss += loss
            num_valid_tasks += 1

        # Average loss across all tasks
        if num_valid_tasks > 0:
            outputs["loss"] = total_loss / num_valid_tasks
        else:
            outputs["loss"] = torch.tensor(0.0, device=device)

        return outputs

    def _average_pooling(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Average pooling over time dimension, considering padding mask.

        Args:
            encoder_out: (batch, time, dim)
            encoder_mask: (batch, 1, time) - True for valid frames, False for padding

        Returns:
            pooled: (batch, dim)
        """
        # encoder_mask: (batch, 1, time) -> (batch, time, 1)
        mask = encoder_mask.transpose(1, 2).float()

        # Sum over valid frames
        masked_sum = (encoder_out * mask).sum(dim=1)  # (batch, dim)

        # Count valid frames
        valid_counts = mask.sum(dim=1)  # (batch, 1)

        # Average
        pooled = masked_sum / (valid_counts + 1e-10)  # (batch, dim)

        return pooled

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        chunk_size: int = -1,
        left_context_size: int = -1,
        right_context_size: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio to features.

        Args:
            speech: (batch, time, feat_dim)
            speech_lengths: (batch,)
            chunk_size: Chunk size for chunked processing
            left_context_size: Left context size
            right_context_size: Right context size

        Returns:
            encoder_out: (batch, time, encoder_dim)
            encoder_mask: (batch, 1, time)
        """
        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
        )
        return encoder_out, encoder_mask

    def classify(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        chunk_size: int = -1,
        left_context_size: int = -1,
        right_context_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """Inference: classify audio samples.

        Args:
            speech: (batch, time, feat_dim)
            speech_lengths: (batch,)
            chunk_size: Chunk size for chunked processing
            left_context_size: Left context size
            right_context_size: Right context size

        Returns:
            Dictionary containing for each task:
                - {task_name}_prediction: (batch,) predicted class indices
                - {task_name}_probability: (batch,) probability of predicted class
        """
        # Encode
        encoder_out, encoder_mask = self.encode(
            speech,
            speech_lengths,
            chunk_size,
            left_context_size,
            right_context_size,
        )

        # Pool
        pooled_features = self._average_pooling(encoder_out, encoder_mask)

        # Classify for each task
        results = {}
        for task_name in self.task_names:
            logits = self.classification_heads[task_name](pooled_features)

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            results[f"{task_name}_prediction"] = predictions

            # Get probabilities and extract only the predicted class probability
            probabilities = F.softmax(logits, dim=-1)
            # Get probability of the predicted class for each sample in batch
            predicted_probs = probabilities.gather(1, predictions.unsqueeze(1)).squeeze(1)
            results[f"{task_name}_probability"] = predicted_probs

        return results

    def get_num_classes(self, task_name: str) -> int:
        """Get number of classes for a specific task."""
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not found. Available tasks: {self.task_names}")
        return self.tasks[task_name]

    def is_multi_task(self) -> bool:
        """Check if model is multi-task."""
        return self.num_tasks > 1
