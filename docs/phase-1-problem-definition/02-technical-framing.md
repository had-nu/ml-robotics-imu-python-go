# Technical Framing & ML Problem Definition

## Machine Learning Problem Formulation

### Problem Type Classification
- **Learning Paradigm:** Supervised Learning  
- **Task Type:** Multi-class Classification  
- **Learning Mode:** Batch Learning (offline training) + Online Inference  
- **Instance-based vs Model-based:** Model-based Learning  

---

## The Classification Task

### Input
6-channel time series data from IMU sensors sampled at **100Hz**:

| Channel | Description | Unit |
|----------|--------------|------|
| acc_x | Acceleration in X-axis | m/s² |
| acc_y | Acceleration in Y-axis | m/s² |
| acc_z | Acceleration in Z-axis | m/s² |
| gyro_x | Angular velocity around X-axis | rad/s |
| gyro_y | Angular velocity around Y-axis | rad/s |
| gyro_z | Angular velocity around Z-axis | rad/s |

- **Window Size:** 100 samples (1 second of data at 100Hz)  
- **Features:** ~60 engineered features per window (statistical + frequency domain)  

### Output
6 Robot Operational States (classes):

| Class | Description | Importance |
|--------|--------------|-------------|
| STANDING | Robot is stationary, motors idle | Base state, energy saving |
| WALKING_FLAT | Normal navigation on flat surface | Most common state |
| WALKING_UPSTAIRS | Ascending ramp or incline | Requires increased torque |
| WALKING_DOWNSTAIRS | Descending ramp or decline | Requires regenerative braking |
| TURNING | Rotating or changing direction | Requires different control |
| OBSTACLE_DETECTED | Abrupt deceleration or collision avoidance | Critical for safety |

---

## Performance Metrics

### Primary Metric: Weighted F1-Score

**Why F1-Score?**
- Balances Precision and Recall  
- Handles class imbalance naturally  
- Industry standard for classification tasks  

**Formula**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Weighted F1 = Σ(weight_i × F1_i)
```

**Target:** F1-Score > 0.93  

---

### Secondary Metrics

1. **Overall Accuracy**  
   - Percentage of correct predictions  
   - **Target:** >95%  
   - **Limitation:** Can mislead with imbalanced data  

2. **Per-Class Precision & Recall**  
   - **Precision:** TP / (TP + FP) → Critical for OBSTACLE_DETECTED  
     - **Target:** >98%  
   - **Recall:** TP / (TP + FN) → Critical for OBSTACLE_DETECTED  
     - **Target:** >98%  

3. **Confusion Matrix**  
   - Visualize systematic errors  
   - Identify confused states (e.g., TURNING vs OBSTACLE_DETECTED)  

4. **Inference Latency**  
   - Input-to-prediction time  
   - **Target:** <10ms (95th percentile)  
   - **Measurement:** 1000 predictions, p95 latency  

5. **Model Size**  
   - Disk footprint of serialized model  
   - **Target:** <50MB (edge-compatible)  
   - Trade-off: Accuracy vs size  

6. **Throughput**  
   - Predictions per second  
   - **Target:** >100 predictions/s  
   - Important for multi-robot scenarios  

---

## Model Architecture Considerations

### Feature Engineering Pipeline

**Time Domain Features (per axis, per window):**
- Mean, Median, Std Dev  
- Min, Max, Range  
- Percentiles (25th, 75th)  
- Skewness, Kurtosis  
→ **9 features × 6 axes = 54 features**

**Frequency Domain Features (FFT):**
- Dominant frequencies  
- Band energy  
- Spectral entropy  
→ **~10 additional features**

**Total:** ~60–65 features per 1-second window  

---

### Model Candidates to Explore

**Traditional ML Models**
| Model | Key Strength | Expected F1 |
|--------|---------------|-------------|
| Logistic Regression | Fast, interpretable baseline | 0.85–0.90 |
| Random Forest | Handles non-linear relations | 0.92–0.95 |
| SVM (RBF kernel) | Works well in high-dimensional space | 0.90–0.93 |
| XGBoost / LightGBM | State-of-the-art gradient boosting | 0.94–0.96 |

**Deep Learning Models (Optional)**
| Model | Description | Trade-off |
|--------|--------------|-----------|
| 1D CNN | Learns directly from raw time series | Higher latency, larger model |
| LSTM / GRU | Captures temporal dependencies | Resource intensive |

Focus on traditional ML first for fast iteration and smaller model footprint.  

---

## Data Pipeline Architecture

### Training Pipeline (Python)
```
Raw Dataset (UCI HAR)
↓
[1] Load & Parse
↓
[2] Train/Val/Test Split (70/15/15)
↓
[3] Window Extraction (1-sec windows, 50% overlap)
↓
[4] Feature Engineering
↓
[5] Standardization (fit on train only)
↓
[6] Model Training (cross-validation)
↓
[7] Hyperparameter Tuning
↓
[8] Final Model Selection
↓
[9] Evaluation on Test Set
↓
[10] Export Model (pickle / ONNX)
```

### Inference Pipeline (Go)

```
Real-time IMU Stream (100Hz)
↓
[1] Buffer 1-second windows
↓
[2] Feature Extraction (same as training)
↓
[3] Standardization (training stats)
↓
[4] Model Inference
↓
[5] Return Prediction + Confidence
```
---

## Train/Validation/Test Strategy

### Split Strategy
- **Temporal Split (70/15/15)**  
  - Training: first 70%  
  - Validation: next 15%  
  - Test: final 15% (unseen data)

**Why Temporal Split?**
- Prevents data leakage from overlapping windows  
- Simulates real-world future data inference  
- Produces more realistic estimates  

### Cross-Validation
**5-Fold Time Series CV (on training set):**
| Fold | Train Range | Validation Range |
|------|--------------|------------------|
| 1 | 0–60% | 60–70% |
| 2 | 0–65% | 65–75% |
| 3 | 0–70% | 70–80% |
| 4 | 0–75% | 75–85% |
| 5 | 0–80% | 80–90% |

**Benefits**
- Better generalization estimate  
- Enables hyperparameter tuning  
- Detects overfitting early  

---

## Baseline Model

| Model | Strategy | Expected Accuracy | Purpose |
|--------|-----------|-------------------|----------|
| Dummy Classifier | Most frequent class | 20–30% | Sanity check |
| Rule-Based | Thresholds on raw data | 75–80% | Compare to ML |
| Logistic Regression | ML baseline | F1 ~0.85–0.90 | Minimum acceptable |

All models must outperform the baseline by **>5% F1-Score**.  

---

## Critical Failure Modes

| Failure Type | Description | Risk | Mitigation |
|---------------|-------------|------|-------------|
| False Negatives | OBSTACLE_DETECTED not triggered | Safety hazard | Optimize recall, tune threshold |
| False Positives | False OBSTACLE_DETECTED alarms | Efficiency loss | Balance precision/recall |
| Systematic Confusion | TURNING misclassified as OBSTACLE_DETECTED | False alarms | Improved feature engineering |

---

## Model Deployment Constraints

### Edge Device Specifications
| Component | Specification |
|------------|----------------|
| CPU | ARM Cortex-A53 (quad-core, 1.2GHz) |
| RAM | 512MB |
| Storage | 4GB |
| GPU | None |

### Latency Requirements
| Mode | Target |
|-------|---------|
| Single prediction | <10ms (p95) |
| Batch (10 samples) | <5ms per sample |

### Model Format
- Python: Pickle / Joblib (training artifacts)  
- Go: ONNX or custom serialized JSON coefficients  

---

## Technical Assumptions

**Data Quality**
- IMU sensors calibrated  
- Sampling rate stable (100Hz ±2Hz)  
- No missing data in production  

**Feature Consistency**
- Identical engineering in training/inference  
- Standardization parameters versioned  

**Class Distribution**
- Reflects real-world conditions  
- OBSTACLE_DETECTED may be rare  

**Model Updates**
- Monthly retraining  
- A/B testing for rollout validation  

---

## Learning Checkpoints

Ensure understanding of:
- Why supervised learning applies  
- Why F1-Score > Accuracy for imbalanced data  
- Why temporal splits are required  
- How feature engineering works  
- Difference between train/val/test  
- Benefits of cross-validation  
- Criteria for model selection  

---

## Technical Decisions Log

| Decision | Rationale | Date |
|-----------|------------|------|
| Use F1-Score as primary metric | Handles class imbalance | Nov 2025 |
| 70/15/15 temporal split | Prevents data leakage | Nov 2025 |
| Start with traditional ML | Faster iteration, smaller models | Nov 2025 |
| 1-second windows, 50% overlap | Balances granularity & smoothness | Nov 2025 |

---
