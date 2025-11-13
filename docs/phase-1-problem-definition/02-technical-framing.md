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
| gyro_x | Angular velocity around X-axis (pitch) | rad/s |
| gyro_y | Angular velocity around Y-axis (roll) | rad/s |
| gyro_z | Angular velocity around Z-axis (yaw) | rad/s |

- **Window Size:** 100 samples (1 second of data at 100Hz)
- **Features:** ~60 engineered features per window (statistical + frequency domain)

### Output
**6–7 Robot Movement States** (classes):

| Class | Description | IMU Pattern | Importance |
|--------|--------------|-------------|-------------|
| **STANDING** | Robot stationary, motors idle | acc ≈ 0, gyro ≈ 0 | Base state, energy saving |
| **MOVING_FORWARD** | Linear motion on flat surface | acc_x > 0, gyro ≈ 0 | Most common operational state |
| **MOVING_BACKWARD** | Reverse motion | acc_x < 0, gyro ≈ 0 | Safety-critical reversing |
| **TURNING_LEFT** | Rotating counterclockwise | gyro_z > threshold | Requires differential motor control |
| **TURNING_RIGHT** | Rotating clockwise | gyro_z < -threshold | Requires differential motor control |
| **ASCENDING_RAMP** | Moving upward on incline | acc_z increases, gyro_y > 0 | Requires increased torque |
| **DESCENDING_RAMP** | Moving downward on decline | acc_z decreases, gyro_y < 0 | Requires regenerative braking |

**Note:** For UCI HAR dataset compatibility, we'll initially map to 4–6 classes:
- **STANDING** ← (SITTING + STANDING + LAYING merged)
- **MOVING_FORWARD** ← (WALKING)
- **ASCENDING_RAMP** ← (WALKING_UPSTAIRS)
- **DESCENDING_RAMP** ← (WALKING_DOWNSTAIRS)

**Optional 7th class** (if enough data):
- **EMERGENCY_STOP** ← Sudden deceleration (collision response)

---

## What This System Does (And Doesn't Do)

### Capabilities (Proprioception)
**IMU detects internal state:**
- Robot's own movement patterns
- Orientation changes (pitch, roll, yaw)
- Acceleration/deceleration events
- State validation (confirm command execution)

**Use cases:**
- Optimize motor torque based on detected state
- Improve odometry (position estimation via dead reckoning)
- Detect anomalies (unexpected stops, collisions after-the-fact)
- Validate control commands ("Did the robot execute the turn?")

### Limitations (Not Exteroception)
**IMU does NOT detect:**
- External obstacles (walls, people, objects)
- Environment structure
- Distance to objects
- Future collisions (only past events)

**What's needed for obstacle avoidance:**
- **Phase 2 sensors:** Laser ToF (VL53L0X) for proximity (0.5–2m range)
- **Phase 3 fusion:** Combine IMU state + ToF distances for decisions
- Example: `IF MOVING_FORWARD AND front_ToF < 30cm → EMERGENCY_STOP`

---

## Performance Metrics

### Primary Metric: Weighted F1-Score

**Why F1-Score?**
- Balances Precision and Recall
- Handles class imbalance naturally
- Industry standard for classification tasks

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Weighted F1 = Σ(weight_i × F1_i) for each class i
```

**Target:** F1-Score > 0.93

**For safety-critical states** (EMERGENCY_STOP, if included):
- Use **F2-Score** (weights Recall 2× higher than Precision)
- Prefer detecting all critical events even with false positives

---

### Secondary Metrics

1. **Overall Accuracy**
   - Percentage of correct predictions
   - **Target:** >95%
   - **Limitation:** Misleading with imbalanced classes

2. **Per-Class Precision & Recall**
   - **Precision:** TP / (TP + FP) → Minimize false alarms
   - **Recall:** TP / (TP + FN) → Don't miss critical states
   - **Target for common states:** Precision/Recall > 90%
   - **Target for rare states:** Adjust based on cost of errors

3. **Confusion Matrix**
   - Visualize systematic misclassifications
   - Identify problematic pairs (e.g., TURNING vs STANDING during brief stops)

4. **Inference Latency**
   - Time from input to prediction
   - **Target:** <10ms (95th percentile)
   - **Measurement:** 1000 predictions, p95 latency
   - **Critical for:** Real-time motor control decisions

5. **Model Size**
   - Serialized model file size
   - **Target:** <50MB (edge device compatible)
   - Trade-off: Accuracy vs memory footprint

6. **Throughput**
   - Predictions per second
   - **Target:** >100 predictions/s
   - Important for: Multi-robot deployments, sensor fusion

---

## Model Architecture Considerations

### Feature Engineering Pipeline

**Time Domain Features** (per axis, per window):
- Mean, Median, Std Dev
- Min, Max, Range
- Percentiles (25th, 75th)
- Skewness, Kurtosis
→ **9 features × 6 axes = 54 features**

**Frequency Domain Features** (FFT):
- Dominant frequency components
- Energy in frequency bands
- Spectral entropy
→ **~10 additional features**

**Total:** ~60–65 features per 1-second window

---

### Model Candidates to Explore

**Traditional ML Models** (Primary Focus)

| Model | Key Strength | Expected F1 | Inference Speed |
|--------|---------------|-------------|-----------------|
| Logistic Regression | Fast, interpretable baseline | 0.85–0.90 | <2ms |
| Random Forest | Non-linear patterns, robust | 0.92–0.95 | ~8ms |
| SVM (RBF kernel) | High-dimensional data | 0.90–0.93 | ~12ms |
| XGBoost / LightGBM | State-of-the-art boosting | 0.94–0.96 | ~6ms |

**Deep Learning** (Optional, if time permits):

| Model | Description | Trade-off |
|--------|--------------|-----------|
| 1D CNN | Learns from raw time series | Higher latency (~50ms), larger model (>50MB) |
| LSTM / GRU | Temporal dependencies | Resource intensive, not edge-friendly |

**Strategy:** Start with traditional ML for fast iteration and edge compatibility.

---

## Data Pipeline Architecture

### Training Pipeline (Python)

```
UCI HAR Dataset (Raw)
    ↓
[1] Load & Parse
    ├── X_train.txt (7352 samples × 561 features)
    └── y_train.txt (activity labels 1–6)
    ↓
[2] Class Mapping
    ├── Merge: SITTING + STANDING + LAYING → STANDING
    └── Map: WALKING → MOVING_FORWARD, etc.
    ↓
[3] Train/Val/Test Split (70/15/15 temporal)
    ↓
[4] Feature Selection (optional)
    ├── Correlation analysis
    └── Feature importance from Random Forest
    ↓
[5] Standardization (fit on train only)
    ├── Mean = 0, Std = 1 per feature
    └── Save scaler params for inference
    ↓
[6] Model Training
    ├── 5-Fold Time Series Cross-Validation
    └── Hyperparameter tuning (GridSearchCV)
    ↓
[7] Model Selection
    ├── Compare F1-Score, Latency, Size
    └── Select best model meeting all criteria
    ↓
[8] Final Evaluation (Test Set - ONE TIME)
    ├── Confusion matrix
    ├── Per-class metrics
    └── Error analysis
    ↓
[9] Model Export
    ├── Python: Pickle (scikit-learn models)
    ├── Go-compatible: ONNX or JSON coefficients
    └── Save scaler params (mean/std per feature)
```

---

### Inference Pipeline (Go)

```
Real-time IMU Stream (100Hz)
    ↓
[1] Buffer Management
    ├── Maintain 1-second sliding window (100 samples)
    └── 50% overlap (new prediction every 0.5s)
    ↓
[2] Feature Extraction (matching training exactly)
    ├── Statistical features (mean, std, etc.)
    └── FFT features (dominant frequencies)
    ↓
[3] Standardization (using saved training params)
    ├── Load mean/std from training
    └── Transform: (x - mean) / std
    ↓
[4] Model Inference
    ├── Load model (startup)
    ├── Forward pass (prediction)
    └── Return: class label + confidence scores
    ↓
[5] Output
    ├── State: e.g., "MOVING_FORWARD"
    ├── Confidence: e.g., 0.94
    └── Latency: measured per prediction
```

---

## Train/Validation/Test Strategy

### Split Strategy

**Temporal Split** (respecting time series nature):
- **Training:** First 70% of data
- **Validation:** Next 15%
- **Test:** Final 15% (completely unseen)

**Why temporal?**
- Prevents data leakage from overlapping windows
- Simulates real deployment (model sees "future" data)
- More realistic performance estimate

### Cross-Validation Strategy

**5-Fold Time Series CV** (on training set only):

| Fold | Train Range | Validation Range |
|------|--------------|------------------|
| 1 | 0–60% | 60–70% |
| 2 | 0–65% | 65–75% |
| 3 | 0–70% | 70–80% |
| 4 | 0–75% | 75–85% |
| 5 | 0–80% | 80–90% |

**Benefits:**
- Robust generalization estimate
- Hyperparameter tuning without touching test set
- Early detection of overfitting

---

## Baseline Models

| Model | Strategy | Expected Performance | Purpose |
|--------|-----------|---------------------|----------|
| Dummy Classifier | Most frequent class | Accuracy ~20–30% | Sanity check (beat random) |
| Rule-Based | Thresholds on raw IMU | F1 ~0.75–0.80 | Demonstrate ML advantage |
| Logistic Regression | ML baseline | F1 ~0.85–0.90 | Minimum acceptable ML |

**Success Criteria:** All models must outperform rule-based by **>10% F1-Score**.

---

## Critical Failure Modes

| Failure Type | Description | Risk Level | Mitigation |
|---------------|-------------|------------|-------------|
| **Misclassification during motion** | MOVING_FORWARD predicted as STANDING | Medium | Better feature engineering, more training data |
| **Confusion on transitions** | TURNING confused with STANDING (brief stops) | Low | Accept as edge case, tune confidence thresholds |
| **EMERGENCY_STOP missed** | Collision not detected (if class included) | High | Optimize Recall (F2-Score), lower threshold |
| **False EMERGENCY_STOP** | Normal deceleration flagged as collision | Medium | Balance Precision/Recall trade-off |

---

## Model Deployment Constraints

### Edge Device Specifications

| Component | Specification |
|------------|----------------|
| CPU | ARM Cortex-A53 (quad-core, 1.2GHz) |
| RAM | 512MB available for inference |
| Storage | 4GB total (model <50MB) |
| GPU | None (CPU-only inference) |
| OS | Linux-based (Go runtime supported) |

### Latency Requirements

| Scenario | Target Latency |
|----------|----------------|
| Single prediction | <10ms (p95) |
| Batch (10 samples) | <5ms per sample |
| Startup (cold start) | <100ms |
| Model load time | <50ms |

### Model Format

**Python (Training):**
- Pickle / Joblib (scikit-learn models)
- Model + scaler saved together

**Go (Inference):**
- Option 1: ONNX (if using onnxruntime-go)
- Option 2: Export model coefficients as JSON (for linear models)
- Option 3: Implement Random Forest/XGBoost inference in native Go

---

## Technical Assumptions

### Data Quality
- IMU sensors properly calibrated
- Sampling rate stable (100Hz ±2Hz acceptable)
- No missing data in production streams
- Sensor noise within acceptable bounds

### Feature Consistency
- Feature engineering identical in training and inference
- Standardization parameters versioned with model
- Feature names/order match exactly

### Class Distribution
- Training data (UCI HAR) approximates real robot movements
- STANDING class may be overrepresented (merged 3 classes)
- Rare states (EMERGENCY_STOP) may need oversampling

### Deployment Assumptions
- Go runtime available on edge device
- Model retraining monthly with new data (future)
- A/B testing infrastructure for model rollout (future)

---

## Multi-Sensor Fusion Architecture (Future)

### Sensor Responsibilities

#### IMU (Phase 1: Current Project)
**Role:** Proprioceptive state classification  
**Output:** Movement state (e.g., MOVING_FORWARD, TURNING_LEFT)  
**Frequency:** 100Hz sampling, predictions every 0.5s  

#### Laser ToF (Phase 2: Future)
**Role:** Obstacle proximity detection  
**Sensors:** 4× VL53L0X (front, back, left, right)  
**Output:** Distance readings (cm) per direction  
**Frequency:** 10Hz  
**Processing:** Threshold-based (`if front < 30cm → obstacle`)

#### Ultrasonic (Phase 2: Future)
**Role:** Redundancy and coarse detection  
**Sensors:** 4× HC-SR04  
**Output:** Distance readings (cm)  
**Frequency:** 5Hz  
**Processing:** Sensor fusion with ToF (majority vote)

### Integration Logic (Phase 3: Future)

**Decision Table Example:**
| IMU State | Front ToF | Action |
|-----------|-----------|--------|
| MOVING_FORWARD | >50cm | Continue |
| MOVING_FORWARD | <30cm | EMERGENCY_STOP |
| TURNING_LEFT | <50cm left | Slow turn |
| STANDING | Any | Idle |

---

## Learning Objectives (Academic Context)

Ensure understanding of:
- [x] Why supervised learning (we have labeled data)
- [x] Why classification not regression (discrete states, not continuous values)
- [x] Why multi-class not binary (>2 states needed)
- [x] Why F1-Score > Accuracy (handles imbalance)
- [x] Why temporal splits matter (prevent data leakage)
- [x] How feature engineering works (time + frequency domain)
- [x] Difference between train/val/test sets
- [x] Benefits of cross-validation
- [x] Model selection criteria (F1 + latency + size)

---

## Technical Decisions Log

| Decision | Rationale | Date |
|-----------|------------|------|
| Use F1-Score as primary metric | Handles class imbalance | Nov 2025 |
| 70/15/15 temporal split | Prevents data leakage in time series | Nov 2025 |
| Start with traditional ML | Faster iteration, edge-compatible | Nov 2025 |
| 1-second windows, 50% overlap | Balances granularity & smoothness | Nov 2025 |
| Merge SITTING/STANDING/LAYING | Only movement states matter for robots | Nov 2025 |
| Remove OBSTACLE_DETECTED class | IMU cannot detect external obstacles | Nov 2025 |
| Target <10ms latency | Robot at 1m/s moves 1cm in 10ms (acceptable) | Nov 2025 |

---
**Last Updated:** November 2025  
**Status:** Approved (v2.0 - Corrected)