# Success Metrics & Evaluation Framework

## Success Criteria Definition
This document defines quantitative and qualitative metrics to evaluate project success from both business and technical perspectives.

---

## Primary Success Metrics

### 1) Model Performance Metrics

**Weighted F1-Score (Primary)**
- **Target:** ≥ 0.93
- **Minimum Acceptable:** 0.90
- **Measurement:** On held-out test set
- **Why it matters:** Balances precision and recall across all classes

**Overall Accuracy**
- **Target:** ≥ 95%
- **Minimum Acceptable:** 92%
- **Measurement:** (Correct Predictions / Total Predictions) × 100
- **Why it matters:** Easy to communicate to stakeholders

---

### 2) Per-Class Performance (Adjusted for Movement States)

**Standard States (Common Operations)**

| Class | Target F1 | Min F1 | Priority | Rationale |
|--------|-----------|--------|----------|-----------|
| STANDING | 0.90 | 0.85 | Medium | Base state, easy to detect |
| MOVING_FORWARD | 0.95 | 0.92 | High | Most common operational state |
| ASCENDING_RAMP | 0.92 | 0.88 | High | Critical for torque optimization |
| DESCENDING_RAMP | 0.92 | 0.88 | High | Critical for braking control |
| TURNING_LEFT/RIGHT | 0.90 | 0.85 | Medium | Complex patterns, motor control |

**Optional: Emergency States (If Included)**

| Class | Target Recall | Min Recall | Priority | Rationale |
|--------|---------------|------------|----------|-----------|
| EMERGENCY_STOP | 0.95 | 0.90 | **CRITICAL** | Must detect sudden collisions |

**Note:** For EMERGENCY_STOP, we prioritize **Recall > Precision** (better to have false alarms than miss real events).

**Weighted Average Target:** 0.93 (weighted by class frequency)

---

### 3) Inference Performance

**Latency**
- **Target:** <10ms (p95 latency)
- **Minimum Acceptable:** <15ms
- **Measurement:** 1000 predictions, 95th percentile
- **Test Environment:** Target edge hardware (ARM Cortex-A53) or equivalent

**Throughput**
- **Target:** >100 predictions/second
- **Minimum Acceptable:** >50 predictions/second
- **Measurement:** Sustained load test for 60 seconds
- **Scenario:** Simulating multiple concurrent prediction requests

---

### 4) Model Efficiency

**Model Size**
- **Target:** <50MB
- **Maximum Acceptable:** <100MB
- **Measurement:** Serialized model file size (including scaler params)
- **Why it matters:** Edge device storage constraints (4GB total)

**Memory Footprint**
- **Target:** <200MB RAM during inference
- **Maximum Acceptable:** <400MB
- **Measurement:** Peak memory usage during batch inference
- **Why it matters:** Edge device has only 512MB RAM total

---

## Business Success Metrics

### 1) Cost Reduction
- **Target:** 98–99% reduction in sensing hardware cost
- **Calculation:** (LIDAR cost − Sensor Fusion cost) / LIDAR cost
- **Baseline:** $5,000 LIDAR vs $85 (IMU + ToF + Ultrasonic)
- **Result:** $85 / $5,000 = **98.3% reduction** [x]

### 2) Time to Market
- **Target:** Complete Phase 1 (IMU classification) in 8 weeks
- **Measurement:** Time from project start to deployable model
- **Milestones:** See timeline in business context document

### 3) Technical Feasibility
- **Target:** Validate that low-cost sensors can achieve >93% F1
- **Measurement:** Test set performance on UCI HAR proxy dataset
- **Future validation:** Real robot hardware testing (Phase 2)

---

## Evaluation Methodology

### Test Set Evaluation Protocol

**Step 1: Final Model Selection**
- Train multiple models on training set
- Tune hyperparameters using validation set
- Select best model based on validation F1-score **AND** latency

**Step 2: Test Set Evaluation (ONE TIME ONLY)**
- Load completely unseen test set (final 15% of data)
- Run predictions without any modifications to model
- Calculate all metrics (F1, accuracy, per-class, confusion matrix)
- Document results and edge cases

**Step 3: Error Analysis**
- Generate confusion matrix (normalized and absolute counts)
- Identify systematic misclassification patterns
- Analyze failure cases (what IMU patterns confused the model?)
- Document edge cases for future improvement

---

### Cross-Validation Results

Report 5-fold Time Series CV results with:
- **Mean F1-Score:** Average across folds
- **Std Dev:** Measure of stability/variance
- **Min/Max:** Range of performance

**Example Format:**
```
Model: Random Forest
CV F1-Score: 0.94 ± 0.02 (min: 0.92, max: 0.96)
→ Stable performance, low variance ✅
```

**Red Flag:** High variance (std > 0.05) indicates overfitting or data quality issues.

---

## Per-Class Performance Targets (Detailed)

### Movement States (Primary Focus)

| Class | Target Precision | Target Recall | Target F1 | Why Important |
|--------|------------------|---------------|-----------|---------------|
| STANDING | 0.90 | 0.90 | 0.90 | Base state for energy saving |
| MOVING_FORWARD | 0.95 | 0.95 | 0.95 | Most common (optimize control) |
| MOVING_BACKWARD | 0.88 | 0.88 | 0.88 | Less common, safety-relevant |
| TURNING_LEFT | 0.88 | 0.88 | 0.88 | Differential motor control |
| TURNING_RIGHT | 0.88 | 0.88 | 0.88 | Differential motor control |
| ASCENDING_RAMP | 0.92 | 0.92 | 0.92 | Torque optimization critical |
| DESCENDING_RAMP | 0.92 | 0.92 | 0.92 | Braking control critical |

### Emergency States (Optional, if data available)

| Class | Target Precision | Target Recall | Target F1 | Priority |
|--------|------------------|---------------|-----------|----------|
| EMERGENCY_STOP | 0.85 | **0.95** | 0.90 | **High Recall** (don't miss collisions) |

**Rationale for EMERGENCY_STOP:**
- **False Negative Cost:** CRITICAL (missed collision)
- **False Positive Cost:** Medium (unnecessary stop, efficiency loss)
- **Strategy:** Lower decision threshold to maximize Recall

---

## Confusion Matrix Analysis

### Acceptable Confusion Patterns

[x] **MOVING_FORWARD ↔ ASCENDING/DESCENDING_RAMP**
- **Why:** Subtle acceleration differences, gradual transitions
- **Impact:** Low (motor control adjusts gradually)

[x] **TURNING_LEFT ↔ TURNING_RIGHT**
- **Why:** Symmetric gyro patterns if mislabeled
- **Impact:** Low (both require similar control)

[x] **STANDING ↔ MOVING (brief transitions)**
- **Why:** Acceleration ramp-up/down periods
- **Impact:** Low (transient states)

### Unacceptable Confusion Patterns

**MOVING_FORWARD → STANDING (frequent)**
- **Impact:** Robot thinks it's stopped when moving → dangerous
- **Action:** Improve feature engineering (velocity magnitude)

**ASCENDING_RAMP → STANDING**
- **Impact:** Robot doesn't increase torque on ramp → stall risk
- **Action:** Emphasize acc_z and gyro_y features

**EMERGENCY_STOP → MOVING_FORWARD**
- **Impact:** Collision not detected
- **Action:** Optimize Recall, lower threshold, collect more data

**Threshold:** If unacceptable patterns exceed **2%** of class samples, implement targeted fixes (feature engineering, threshold tuning, class reweighting).

---

## Performance Benchmarking

### Python vs Go Comparison Metrics

| Metric | Python Baseline | Go Target | Measurement Method |
|--------|-----------------|-----------|---------------------|
| Inference Latency (single) | ~50ms | <10ms | 1000 predictions, p95 |
| Throughput | ~20 req/s | >100 req/s | Sustained 60s load test |
| Memory Usage | ~500MB | <200MB | Peak during inference |
| Startup Time | ~2s | <100ms | Cold start to first prediction |
| Model Load Time | ~500ms | <50ms | Load serialized model |

**Goal:** Demonstrate Go's advantages for production inference (5× faster, 2.5× less memory) while Python excels at training and experimentation.

---

## Visualization Requirements

### Required Plots for Evaluation

1. **Confusion Matrix** (2 versions)
   - Normalized (percentages)
   - Absolute counts

2. **Per-Class Metrics Bar Chart**
   - Precision, Recall, F1 for each class
   - Horizontal bars for easy comparison

3. **ROC Curves** (if binary relevance needed)
   - One-vs-rest approach for each class
   - AUC score displayed

4. **Precision-Recall Curves**
   - Especially for EMERGENCY_STOP (if included)
   - Shows threshold trade-offs

5. **Learning Curves**
   - Train vs Validation F1-Score over training samples
   - Detect overfitting/underfitting

6. **Feature Importance** (for tree-based models)
   - Top 20 features
   - Understand what IMU patterns matter most

7. **Latency Distribution**
   - Histogram of inference times
   - Show p50, p95, p99

8. **Throughput over Time**
   - Line chart for 60-second sustained test
   - Detect performance degradation

---

## Model Comparison Framework

### Comparison Table Template

| Model | F1-Score | Accuracy | Latency (p95) | Size | Train Time | Notes |
|--------|----------|----------|---------------|------|------------|-------|
| Dummy | 0.20 | 20% | <1ms | <1KB | <1s | Random guess (sanity check) |
| Rule-Based | 0.78 | 80% | <1ms | N/A | N/A | Thresholds on raw IMU |
| Logistic Reg | 0.87 | 88% | 2ms | 5KB | 10s | Fast, interpretable baseline |
| Random Forest | 0.94 | 95% | 8ms | 30MB | 5min | Best F1, acceptable latency |
| XGBoost | 0.95 | 96% | 6ms | 25MB | 10min | **Winner** (best balance) |
| SVM (RBF) | 0.91 | 92% | 12ms | 15MB | 30min | Too slow for <10ms target |
| 1D CNN | 0.96 | 97% | 50ms | 80MB | 2h | Best accuracy, fails latency |

**Selection Criteria (in order of priority):**
1. [x] F1-Score > 0.93
2. [x] Latency < 10ms (p95)
3. [x] Size < 50MB
4. [x] Simplicity/Maintainability

**Selected Model:** XGBoost (meets all criteria, best overall)

---

## Red Flags & Stop Conditions

### When to Stop and Reassess

**Test F1-Score < 0.85**
- **Action:** Revisit feature engineering, try different models, collect more data

**EMERGENCY_STOP Recall < 0.90** (if class included)
- **Action:** Adjust class weights, lower threshold, oversample minority class

**Latency > 20ms consistently**
- **Action:** Model compression (pruning), simpler architecture, optimize Go code

**Overfitting (Train F1 > 0.99, Test F1 < 0.90)**
- **Action:** Regularization (L1/L2), reduce model complexity, more training data

**Underfitting (Train F1 < 0.90)**
- **Action:** More complex model, better features, more training epochs

**High CV variance (std > 0.05)**
- **Action:** Check data quality, ensure proper shuffling, increase training data

---

## Reporting Template

### Model Evaluation Report Structure

#### Model: [Model Name]

##### Summary
- **F1-Score:** X.XX (Target: ≥0.93) [ [x] / x ]
- **Accuracy:** XX% (Target: ≥95%) [ [x] / x ]
- **Latency:** XXms (Target: <10ms) [ [x] / x ]
- **Model Size:** XXMB (Target: <50MB) [ [x] / x ]

##### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| STANDING | 0.XX | 0.XX | 0.XX | NNN |
| MOVING_FORWARD | 0.XX | 0.XX | 0.XX | NNN |
| ... | ... | ... | ... | ... |

##### Confusion Matrix
[Visualization + written analysis of patterns]

##### Strengths
- [What this model does well]
- [Specific classes with high F1]

##### Weaknesses
- [What needs improvement]
- [Problematic class pairs]

##### Edge Cases Identified
- [Specific scenarios where model fails]
- [Examples from test set]

##### Production Readiness
- [ [x] / x ] Meets all criteria
- **Recommendation:** Deploy / Needs work / Reject

---

## Learning Validation

### Concept Checklist (Self-Assessment)

After completing evaluation, ensure understanding of:
- [ ] Why F1-score matters more than accuracy (handles imbalance)
- [ ] How to interpret a confusion matrix (patterns of errors)
- [ ] What precision vs recall trade-off means (FP vs FN cost)
- [ ] Why separate train/val/test sets are critical (prevent overfitting)
- [ ] How to detect overfitting from metrics (high train, low test)
- [ ] What makes a model "production-ready" (meets all constraints)
- [ ] How to compare models fairly (same data, same metrics)
- [ ] Why latency matters in real-time systems (robot reaction time)
- [ ] Difference between proprioception (IMU) and exteroception (ToF/ultrasonic)

---

## Go/No-Go Decision Criteria

### Proceed to Go Implementation if:
[x] Test F1-Score ≥ 0.93  
[x] Per-class F1 meets minimum targets  
[x] No critical confusion patterns (>2% unacceptable errors)  
[x] Model size <50MB  
[x] Python inference latency reasonable (<100ms, for baseline)  
[x] Cross-validation stable (std < 0.05)

### Proceed to Deployment if (after Go implementation):
[x] Go inference latency <10ms (p95)  
[x] Throughput >100 req/s  
[x] Memory usage <200MB  
[x] All functional tests pass  
[x] Documentation complete  

### Return to Development if:
X Any critical metric below minimum  
X Unacceptable confusion patterns persist  
X High variance across CV folds  
X Go implementation fails performance targets  
X Edge device testing reveals issues

---

## Evaluation Timeline

| Phase | Evaluation Type | When | Duration |
|--------|----------------|------|----------|
| Development | CV on training set | Continuous | Per model |
| Selection | Validation set evaluation | After training | 1 day |
| Final Eval | Test set (one-time) | Before Go impl | 1 day |
| Benchmarking | Python baseline latency | After final eval | 0.5 day |
| Go Implementation | Port model to Go | Week 7 | 3 days |
| Go Benchmarking | Latency/throughput tests | After Go impl | 1 day |
| Sign-off | Documentation review | Before final report | 1 day |

---

## Success Declaration

**Phase 1 (IMU Classification) is successful when:**
1. [x] Test F1-Score ≥ 0.3
2. [x] Per-class metrics met targets
3. [x] Go implementation acieves <10ms latency
4. [x] Python vs Go comparison demonstrates clear advantages
5. [x] Documentation complete and reviewed
6. [x] Model artifacts properly versioned (Git + model file)
7. [x] Reproducible training pipeline documented

**Phase 2 (Physical Prototype) success criteria:**
- Arduino prototype successfully collects IMU data
- Models validate on real robot hardware
- Domain adaptation quantified (performance drop from proxy dataset)

**Phase 3 (Sensor Fusion) success criteria:**
- ToF/ultrasonic integration demonstrates improved safety
- Multi-sensor decision logic reduces false negatives
- Cost remains <$100 total for sensor suite

---

**Next Step:** Proceed to **Phase 2: Data Acquisition** — download the UCI HAR dataset and perform initial exploration.

**Last Updated:** November 2025  
**Status:** Approved (v2.0 - Corrected)