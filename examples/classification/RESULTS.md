# Classification Results

Comparison of multi-task classification performance with and without transfer learning from pre-trained ASR model.

> **⚠️ Important Note**: The dataset exhibits extreme class imbalance across Age, Dialect, and Emotion tasks. This severe imbalance causes highly unstable training when training from scratch, as the model tends to collapse to predicting only the majority class. Further investigation is needed to determine whether this instability is primarily due to:
> 1. **Data imbalance** - insufficient samples for minority classes
> 2. **Model architecture** - inadequate capacity or regularization for imbalanced learning
> 3. **Training strategy** - need for class weighting, focal loss, or other imbalance-handling techniques
>
> The dramatic improvement with transfer learning suggests that pre-trained representations provide crucial initialization that prevents majority class collapse.

## Experimental Setup

- **Dataset**: [LSVSC](doof-ferb/LSVSC) Vietnamese speech dataset
- **Training Set Size**: 40,102 samples
- **Dev/Test Set Size**: 4,457 samples each
- **Tasks**: 4 classification tasks (Gender, Age, Dialect, Emotion)
- **Model**: ChunkFormer encoder
- **Configuration**: `conf/multi_task.yaml`
- **Training**:
  - **From Pretrain**: Initialized with pre-trained ASR encoder (`khanhld/chunkformer-rnnt-large-vie`)
  - **From Scratch**: Randomly initialized encoder
- **Checkpoint**: [![Hugging Face](https://img.shields.io/badge/HuggingFace-chunkformer--gender--emotion--dialect--age--classification-orange?logo=huggingface)](https://huggingface.co/khanhld/chunkformer-gender-emotion-dialect-age-classification)

## Dataset Statistics

### Training Set (40,102 samples)

| Task | Classes | Distribution | Balance |
|------|---------|--------------|---------|
| **Gender** | 2 | Male: 49.65% / Female: 50.35% | ✅ Balanced |
| **Age** | 5 | 0: 0.08% / 1: 42.62% / 2: 5.30% / 3: 0.62% / 4: 51.38% | ❌ Highly Imbalanced |
| **Dialect** | 5 | 0: 3.39% / 1: 0.70% / 2: 0.05% / 3: 88.10% / 4: 7.76% | ❌ Highly Imbalanced |
| **Emotion** | 8 | 0: 0.21% / 1: 0.03% / 2: 0.08% / 3: 0.06% / 4: 0.66% / 5: 98.57% / 6: 0.35% / 7: 0.04% | ❌ Highly Imbalanced |


### Test Set (4,457 samples)

| Task | Classes | Distribution | Balance |
|------|---------|--------------|---------|
| **Gender** | 2 | Male: 50.33% / Female: 49.67% | ✅ Balanced |
| **Age** | 5 | 0: 0.07% / 1: 42.00% / 2: 5.12% / 3: 0.45% / 4: 52.37% | ❌ Highly Imbalanced |
| **Dialect** | 5 | 0: 3.46% / 1: 0.47% / 2: 0.07% / 3: 88.45% / 4: 7.56% | ❌ Highly Imbalanced |
| **Emotion** | 8 | 0: 0.04% / 1: 0.04% / 2: 0.04% / 3: 0.07% / 4: 0.56% / 5: 98.83% / 6: 0.38% / 7: 0.02% | ❌ Highly Imbalanced |



## Overall Results Summary

| Task | Metric | From Pretrain | From Scratch | Improvement |
|------|--------|--------------|--------------|-------------|
| **Gender** | Accuracy | **98.4%** | 51.5% | +46.9% |
| | Weighted F1 | **0.98** | 0.51 | +0.47 |
| **Age** | Accuracy | **80.5%** | 52.4% | +28.1% |
| | Weighted F1 | **0.80** | 0.36 | +0.44 |
| **Dialect** | Accuracy | **95.5%** | 88.5% | +7.0% |
| | Weighted F1 | **0.95** | 0.83 | +0.12 |
| **Emotion** | Accuracy | **98.9%** | 98.8% | +0.1% |
| | Weighted F1 | **0.99** | 0.98 | +0.01 |

## Detailed Results by Task

### 1. Gender Classification (2 classes: Male/Female)

**From Pretrain** (Best Performance):
```
                  precision    recall  f1-score   support

               0       0.98      0.99      0.98      2243
               1       0.99      0.98      0.98      2214

        accuracy                           0.98      4457
       macro avg       0.98      0.98      0.98      4457
    weighted avg       0.98      0.98      0.98      4457
```

**From Scratch**:
```
                  precision    recall  f1-score   support

               0       0.52      0.56      0.54      2243
               1       0.51      0.47      0.49      2214

        accuracy                           0.51      4457
       macro avg       0.51      0.51      0.51      4457
    weighted avg       0.51      0.51      0.51      4457
```

---

### 2. Age Classification (5 classes)

**From Pretrain**:
```
                  precision    recall  f1-score   support

               0       0.75      1.00      0.86         3
               1       0.78      0.75      0.76      1872
               2       0.88      0.71      0.78       228
               3       0.86      0.90      0.88        20
               4       0.81      0.85      0.83      2334

        accuracy                           0.80      4457
       macro avg       0.81      0.84      0.82      4457
    weighted avg       0.80      0.80      0.80      4457
```

**From Scratch**:
```
                  precision    recall  f1-score   support

               0       0.00      0.00      0.00         3
               1       0.00      0.00      0.00      1872
               2       0.00      0.00      0.00       228
               3       0.00      0.00      0.00        20
               4       0.52      1.00      0.69      2334

        accuracy                           0.52      4457
       macro avg       0.10      0.20      0.14      4457
    weighted avg       0.27      0.52      0.36      4457
```

---

### 3. Dialect Classification (5 classes)

**From Pretrain**:
```
                  precision    recall  f1-score   support

             0.0       0.71      0.56      0.63       154
             1.0       0.65      0.52      0.58        21
             2.0       1.00      0.67      0.80         3
             3.0       0.97      0.99      0.98      3942
             4.0       0.84      0.81      0.82       337

        accuracy                           0.96      4457
       macro avg       0.83      0.71      0.76      4457
    weighted avg       0.95      0.96      0.95      4457
```

**From Scratch**:
```
                  precision    recall  f1-score   support

             0.0       0.00      0.00      0.00       154
             1.0       0.00      0.00      0.00        21
             2.0       0.00      0.00      0.00         3
             3.0       0.88      1.00      0.94      3942
             4.0       0.00      0.00      0.00       337

        accuracy                           0.88      4457
       macro avg       0.18      0.20      0.19      4457
    weighted avg       0.78      0.88      0.83      4457
```
---

### 4. Emotion Classification (8 classes)

**From Pretrain**:
```
                  precision    recall  f1-score   support

             0.0       0.00      0.00      0.00         2
             1.0       0.00      0.00      0.00         2
             2.0       0.00      0.00      0.00         2
             3.0       0.00      0.00      0.00         3
             4.0       0.76      0.52      0.62        25
             5.0       0.99      1.00      0.99      4405
             6.0       0.27      0.18      0.21        17
             7.0       1.00      1.00      1.00         1

        accuracy                           0.99      4457
       macro avg       0.38      0.34      0.35      4457
    weighted avg       0.99      0.99      0.99      4457
```

**From Scratch**:
```
                  precision    recall  f1-score   support

             0.0       0.00      0.00      0.00         2
             1.0       0.00      0.00      0.00         2
             2.0       0.00      0.00      0.00         2
             3.0       0.00      0.00      0.00         3
             4.0       0.00      0.00      0.00        25
             5.0       0.99      1.00      0.99      4405
             6.0       0.00      0.00      0.00        17
             7.0       0.00      0.00      0.00         1

        accuracy                           0.99      4457
       macro avg       0.12      0.12      0.12      4457
    weighted avg       0.98      0.99      0.98      4457
```

---
