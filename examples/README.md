# ChunkFormer Training Guide

This guide will walk you through the process of training ChunkFormer models from scratch.

## Prerequisites

### 1. Clone the Repository
```bash
git clone https://github.com/khanld/chunkformer.git
cd chunkformer
```

### 2. Install Conda
Please see https://docs.conda.io/en/latest/miniconda.html

### 3. Create Conda Environment
```bash
conda create -n chunkformer python=3.11
conda activate chunkformer
conda install conda-forge::sox
```

### 4. Install PyTorch & torchaudio
It's recommended to use PyTorch 2.5.1 with CUDA 12.1, though newer versions work fine.
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 5. Install ChunkFormer in Development Mode
```bash
pip install -e .
```
This installs ChunkFormer with all required dependencies for training and development.

### 6. Training RNN-T with K2 loss
For training RNN-T models with the k2 pruned loss, refer to this [PAGE](https://k2-fsa.github.io/k2/installation/pre-compiled-cuda-wheels-linux/index.html) to find the compatible k2 version.

## Data Preparation

### 1. Create Data Directory Structure

First, create a `data/` directory in your training example folder (e.g., `examples/asr/ctc/` or `examples/asr/rnnt/`):

```bash
cd examples/asr/ctc  # or examples/asr/rnnt
mkdir -p data
```

### 2. Required Folder Structure

Your training directory must follow this structure:

```
examples/asr/ctc/  # or rnnt
├── data/                       # MUST CREATE THIS FOLDER YOURSELF
│   ├── train_set_name/         # Your training set folder
│   │   └── data.tsv            # REQUIRED: Training data file
│   ├── dev_set_name/           # Your validation set folder
│   │   └── data.tsv            # REQUIRED: Validation data file
│   └── test_set_name/          # Your test set folder
│       └── data.tsv            # REQUIRED: Test data file
├── conf/
│   └── your_config.yaml        # Training configuration
├── exp/                        # Experiment outputs (auto-created)
├── tensorboard/                # Tensorboard logs (auto-created)
├── tools/                      # training tools
├── run.sh                      # Training script
└── path.sh                     # Environment setup
```

### 3. Data.tsv Format

Each `data.tsv` file must contain **exactly 3 columns** with **tab-separated values**:

```tsv
key	wav	txt
utterance_001	/path/to/audio1.wav	transcription text here
utterance_002	/path/to/audio2.wav	another transcription
utterance_003	/path/to/audio3.wav	more transcription text
```

**Column Specifications:**
- `key`: Unique identifier for each utterance
- `wav`: **Absolute path** to the audio file (.wav, .flac, .mp3, etc.)
- `txt`: Ground truth transcription text


## Training Configuration

### 1. Update run.sh Variables

Edit the `run.sh` script and modify these key variables:

```bash
...
# For multi-GPU training, set `CUDA_VISIBLE_DEVICES`:
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Use 4 GPUs

# To resume training from a checkpoint, set checkpoint path
checkpoint=/path/to/your/checkpoint.pt

# Set your dataset names (must match folder names in data/)
train_set=train_set_name           # Your training folder name
dev_set=dev_set_name           # Your validation folder name
recog_set=test_set_name        # Your test folder name

# Training configuration
train_config=conf/v0.yaml   # Your model config file
dir=exp/v0                 # Experiment output directory

# To enable Mixed Precision training (default)
chunkformer/bin/train.py \
    --use_amp \
    ...
...
```

### 2. Model Configuration

For model configuration, refer to the example configuration file in `conf/v0.yaml`. This file contains all the necessary parameters for model architecture, training settings, and data processing configurations.

## Training Process

### Stage-by-Stage Training

ChunkFormer training follows a 7-stage process:

#### Stage 0: Data Format Conversion
```bash
./run.sh --stage 0 --stop-stage 0
```
- Converts `data.tsv` files to required `.list`, `text`, and `wav.scp` formats
- **Automatic**: No manual intervention needed

#### Stage 1: Feature Generation
```bash
./run.sh --stage 1 --stop-stage 1
```
- Computes CMVN (Cepstral Mean and Variance Normalization) statistics
- Generates `global_cmvn` file for feature normalization

#### Stage 2: Vocabulary Preparation
```bash
./run.sh --stage 2 --stop-stage 2
```
- Creates BPE/character-level vocabulary
- Generates `*_units.txt` dictionary file
- Builds subword model if using BPE

#### Stage 3: Model Training
```bash
./run.sh --stage 3 --stop-stage 3
```
- **Main training stage**
- Trains the neural network model
- Saves checkpoints in `$dir`
- Monitor training via TensorBoard logs

#### Stage 4: Model Evaluation
```bash
./run.sh --stage 4 --stop-stage 4
```
- Averages multiple checkpoints for better performance
- Runs inference on test sets
- Computes Word Error Rate (WER) metrics

#### Stage 5: Model Export for Inference
```bash
./run.sh --stage 5 --stop-stage 5
```
- Packages model for ChunkFormer inference
- Creates `model_checkpoint_*` directory with all required files

#### Stage 6: Push Model to Hugging Face Hub (Optional)
```bash
./run.sh --stage 6 --stop-stage 6
```
- Uploads the prepared model directory to the Hugging Face Hub if `hf_token` and `hf_repo_id` are set in the script.


## Monitoring & Outputs

### 1. TensorBoard Monitoring
Monitor training progress with TensorBoard:

```bash
tensorboard --logdir=tensorboard/{your_experiment_name} --port=6006
```

Training logs are organized as:
```
tensorboard/
├── v0
├── v1
└── ...
```

### 2. Output Files

After successful training, you'll find:

```
$dir
|── epoch_{epoch}.pt                  # model checkpoint at each epoch
|── epoch_{epoch}.yaml                # Config and logs at each epoch
├── final.pt                          # Final model checkpoint
├── avg_5.pt                          # Averaged checkpoint
├── train.yaml                        # Training config
└── model_checkpoint_avg_5/           # Ready for inference
    ├── tokenizer                     # Tokenizer folder
    ├── pytorch_model.pt              # Model weights
    ├── config.yaml                   # Model config
    ├── global_cmvn                   # Normalization stats
    └── vocab.txt                     # Vocabulary file
```

The `model_checkpoint_*` directory can be used directly with ChunkFormer's inference API:

```python
import chunkformer
model = chunkformer.ChunkFormerModel.from_pretrained('$dir/model_checkpoint_avg_5')
```
