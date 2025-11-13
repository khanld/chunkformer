# Speech Classification with ChunkFormer

Complete implementation for speech classification tasks using ChunkFormer encoder. Supports both **single-task** and **multi-task** classification (gender, emotion, region, accent, age, etc.).

```
Audio → Fbank → ChunkFormer Encoder → Average Pooling → Classification Heads
```

## Quick Start

### 1. Installation

```bash
cd /path/to/chunkformer
pip install -e .
```

### 2. Data Format

Create `data.tsv` files:

```tsv
key	wav	gender_label
utt001	/path/to/audio1.wav	0
utt002	/path/to/audio2.wav	1
```

For multi-task, add more label columns:
```tsv
key	wav	gender_label	emotion_label	region_label
utt001	/path/to/audio1.wav	0	1	2
utt002	/path/to/audio2.wav	1	3	0
```

**Note**:
- Labels must be integers starting from 0
- `key` column is optional (auto-generated from wav path if missing)

### 3. Directory Structure

```
examples/classification/
├── data/
│   ├── train/data.tsv
│   ├── dev/data.tsv
│   └── test/data.tsv
├── conf/
│   ├── single_task.yaml
│   └── multi_task.yaml
└── run.sh
```

### 4. Configuration

**Single-task** (`conf/single_task.yaml`):
```yaml
model: classification
model_conf:
    tasks:
        gender: 2          # 2 classes
    dropout_rate: 0.1
    label_smoothing: 0.1   # Optional

dataset: classification
dataset_conf:
    tasks: ['gender']
    batch_conf:
        batch_type: static
        batch_size: 8
```

**Multi-task** (`conf/multi_task.yaml`):
```yaml
model_conf:
    tasks:
        gender: 2
        emotion: 7
        region: 5
dataset_conf:
    tasks: ['gender', 'emotion', 'region']
```

### 5. Training Pipeline

```bash
cd examples/classification

# Full pipeline (data prep + training)
./run.sh --stage 0 --stop-stage 3

# Or run individual stages:
./run.sh --stage 0 --stop-stage 0  # Convert TSV to list format
./run.sh --stage 1 --stop-stage 1  # Compute CMVN
./run.sh --stage 2 --stop-stage 2  # Analyze labels
./run.sh --stage 3 --stop-stage 3  # Train model
```

**Multi-GPU training**:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./run.sh --stage 3 --stop-stage 3
```

**Resume training**:
```bash
./run.sh --stage 3 --stop-stage 3 --checkpoint exp/classification_v1/10.pt
```

**Transfer learning from ASR**:
```bash
# Load encoder weights from pre-trained ASR model
./run.sh --stage 3 --stop-stage 3 --checkpoint /path/to/asr_model.pt
```
Note: Only encoder weights are loaded; classification heads are randomly initialized.

**Monitor training**:
```bash
tensorboard --logdir tensorboard/classification_v1 --port 6006
```

### 6. Evaluation

```bash
# Inference on test set
./run.sh --stage 4 --stop-stage 4
```

Results in `exp/classification_v1/test/metrics.txt`:
```
======================================================================
Classification Metrics
======================================================================

GENDER:
  Samples: 4457

  Classification Report:
                  precision    recall  f1-score   support

               0       0.98      0.99      0.98      2243
               1       0.99      0.98      0.98      2214

        accuracy                           0.98      4457
       macro avg       0.98      0.98      0.98      4457
    weighted avg       0.98      0.98      0.98      4457

  Confusion Matrix:
    0: [2213, 30]
    1: [41, 2173]
```

## Tools

### Convert Text Labels to Integers

```bash
# Basic usage - auto-detects tasks from columns ending with '_label'
python tools/convert_text_labels_to_int.py \
    --input data.tsv

# Or specify tasks explicitly
python tools/convert_text_labels_to_int.py \
    --input data.tsv \
    --tasks gender emotion region
```

### Split Train/Test

```bash
python tools/split_classification_data.py \
    --input data/train/data.tsv \
    --train_output data/train/data.tsv \
    --test_output data/test/data.tsv \
    --test_ratio 0.2 \
    --seed 42
```
