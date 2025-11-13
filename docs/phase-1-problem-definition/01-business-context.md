# Business Context & Problem Definition

## Project Overview
**Project Name:** RoboSense - Low-Cost Sensor Fusion for AMR State Classification  
**Date:** November 2025  
**Version:** 2.0  
**Team:** ML Engineering (Academic Project)

---

## Business Context

### Company Profile
**RoboSense Labs** is a fictional startup developing autonomous mobile robots (AMR) for industrial warehouses and manufacturing facilities. Their robots navigate complex environments, transport goods, and collaborate with human workers.

### The Business Problem

#### Current Market State
**High-end AMRs** use expensive sensor suites:
- **LIDAR:** $5,000–$15,000 (comprehensive 3D mapping, 30m range)
- **High-grade IMU:** $500–$1,000 (precision navigation)
- **Total sensing cost:** ~$6,000+ per unit

**Market Gap:** Small and medium enterprises (SMEs) need affordable AMRs (<$5k total cost) for:
- **Structured environments** (pre-mapped warehouses)
- **Predictable routes** (repetitive tasks, line-following)
- **Collaborative operation** (working alongside humans)

#### Cost Challenge
- Premium sensors increase unit cost by **60%**
- Limits market to large enterprises only
- SMEs cannot justify $15k+ robots for simple tasks

#### Technical Challenge
- **LIDAR is overkill** for structured warehouses with known layouts
- Need **state awareness** for control optimization (torque, speed, safety)
- Current low-cost solutions lack reliability (~75–80% accuracy)

#### Safety Challenge
- Must detect critical states (sudden stops, collisions) instantly
- Need redundancy in sensing for fail-safe operation
- Low-cost sensors often unreliable individually

---

## The Proposed Solution

Develop an **ML-based multi-sensor fusion system** using affordable components:

### Hardware Stack (Prototipável com Arduino)
| Sensor | Model | Cost | Purpose |
|--------|-------|------|---------|
| **IMU** | MPU6050/9250 | $3–5 | Movement state classification |
| **Laser ToF** | VL53L0X × 4 | $48–60 | Proximity detection (4 directions) |
| **Ultrasonic** | HC-SR04 × 4 | $8–20 | Redundancy & validation |
| **Total** | | **$60–85** | **99% cost reduction vs LIDAR** |

### Objectives
1. **Proprioceptive awareness:** Classify robot movement states using IMU
2. **Obstacle proximity:** Detect nearby objects with ToF/ultrasonic (future phase)
3. **Fast decision-making:** <10ms latency for real-time control
4. **Cost reduction:** 98–99% sensing cost vs premium solutions
5. **Edge deployment:** Run on low-power embedded hardware

### Trade-offs Accepted
- X **No 3D mapping** (requires pre-mapped environments)
- X **Limited range** (2m detection vs 30m LIDAR)
- [x] **Sufficient for:** Warehouse corridors, pick-and-place, guided navigation
- [x] **Not for:** Dynamic outdoor environments, complex obstacle avoidance

---

## Business Objectives

### Primary Objective
**Validate low-cost sensor fusion for AMR state classification** with:
- **>93% F1-score** on movement state classification (IMU)
- **<10ms inference latency** for real-time control
- **$85 total sensing cost** (99% reduction vs LIDAR baseline)

### Secondary Objectives
- **Performance:** Achieve real-time inference on edge devices (512MB RAM)
- **Scalability:** Architecture supports future multi-sensor fusion
- **Deployment:** Compare Python (training) vs Go (inference) performance
- **Reproducibility:** Build physical prototype with Arduino

---

## Success Metrics

| Metric | Target | Critical? |
|---------|---------|-----------|
| IMU State Classification F1-Score | >0.93 | Yes |
| Overall Accuracy | >95% | Yes |
| Inference Latency | <10ms | Yes |
| Model Size | <50MB | Yes |
| Sensing Hardware Cost | <$100 | Yes |
| Throughput | >100 req/s | Nice-to-have |

---

## Stakeholders

### Primary Stakeholders
| Role | Need | Concern |
|------|------|----------|
| Robotics Engineering | Fast, reliable state classification API | Integration complexity, latency |
| Product Management | Cost reduction, market accessibility | Time-to-market, feature completeness |
| Safety & Compliance | High reliability on critical states | System failures, certification |

### Secondary Stakeholders
- **Operations Team:** Deployment and monitoring
- **Hardware Team:** Sensor integration and prototyping
- **Executive Leadership:** ROI and strategic positioning

---

## Business Impact

### Financial Impact
- **Cost Savings:** $5,900 per robot unit (sensing only)
- **Market Expansion:** Access to SME segment ($5k robot vs $15k)
- **Competitive Advantage:** 10× cheaper sensing than competitors

### Operational Impact
- **Energy Efficiency:** Optimize motor power based on detected state
- **Predictive Maintenance:** Detect abnormal movement patterns
- **Better Control:** Real-time torque/speed adjustments

### Strategic Impact
- **Technology Leadership:** Demonstrate practical ML for embedded systems
- **Prototyping Platform:** Arduino-based solution for rapid iteration
- **Scalability:** Foundation for multi-sensor fusion (Phase 2)

---

## How the Solution Will Be Used

### Project Scope (Phase 1: Current)
**Focus:** IMU-based movement state classification

#### Robot Hardware (Edge)
```
IMU Sensor (MPU6050, 100Hz sampling)
    ↓
├── Accelerometer (3-axis) → detect linear motion
└── Gyroscope (3-axis) → detect rotation
    ↓
Preprocessing Module
├── Window extraction (1 second = 100 samples)
└── Feature computation (statistical + FFT)
    ↓
ML Inference Engine (Go)
├── Load trained model
├── Real-time classification
└── Send state to control system
    ↓
Control System Actions
├── STANDING → idle motors
├── MOVING_FORWARD → maintain speed
├── TURNING → adjust differential torque
└── ASCENDING_RAMP → increase torque
```

#### Cloud (Training Pipeline)
```
UCI HAR Dataset (Proxy for robot IMU data)
    ↓
Python Training Pipeline
├── Data preprocessing
├── Feature engineering
├── Model training (scikit-learn, XGBoost)
└── Model export (pickle, ONNX)
    ↓
Go Inference Server
├── Load model
├── Real-time classification
└── Benchmark (latency, throughput)
```

---

### Real-Time Workflow

1. **Data Collection (100Hz)**
   - IMU streams 6 channels: `acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`

2. **Windowing (1-second windows)**
   - Extract rolling windows of 100 samples
   - 50% overlap for smooth transitions

3. **Feature Extraction**
   - Statistical features (mean, std, min, max, percentiles)
   - Frequency domain features (FFT components)

4. **Classification**
   - Model predicts one of 6–7 states
   - Returns confidence scores

5. **Control Action**
   - Adjust motor torque/speed
   - Trigger safety protocols if needed

---

## Use Cases

### Use Case 1: Energy Optimization
- Detect `MOVING_FORWARD` → normal power
- Detect `ASCENDING_RAMP` → increase motor torque by 20%
- Detect `DESCENDING_RAMP` → engage regenerative braking
→ **Result:** 15% battery life improvement

### Use Case 2: Motion Validation
- Command issued: "Turn left"
- IMU detects: `TURNING_LEFT` state
- System confirms: Command executed correctly
→ **Result:** Improved odometry and localization

### Use Case 3: Anomaly Detection
- Expected state: `MOVING_FORWARD`
- IMU detects: `EMERGENCY_STOP` (sudden deceleration)
- System triggers: Alert to investigate collision/obstruction
→ **Result:** Early fault detection

---

## Current Baseline

**Existing Approach:** Rule-based thresholds on raw IMU data
- Accuracy: ~75–80%
- High false positive rate on state transitions
- No learning or adaptation

**Why ML is Needed:**
- **Complex patterns:** Multi-dimensional state transitions
- **Adaptability:** Learn from diverse operational data
- **Robustness:** Handle sensor noise and drift
- **Scalability:** Easily add new states

---

## Constraints & Assumptions

### Technical Constraints
- Must run on edge devices (ARM Cortex-A53, 512MB RAM)
- Real-time requirement: <10ms inference
- Model size: <50MB
- No internet connectivity (offline inference)

### Data Constraints
- Using **UCI HAR dataset** as proxy for robot IMU data
- Assumes human activity patterns approximate robot movements
- May require domain adaptation for real robot deployment

### Business Constraints
- 8-week academic project timeline
- Limited budget (Arduino prototyping only)
- No custom data collection initially

### Assumptions
- IMU sampling rate consistent (100Hz ±2Hz)
- 6–7 movement states sufficient for MVP
- Edge devices support Go runtime
- Future integration of ToF/ultrasonic sensors (Phase 2)

---

## Multi-Sensor Fusion Roadmap

### Phase 1 (Current Project): IMU Classification
**Deliverables:**
- Train ML models (Python) on UCI HAR dataset
- Deploy inference server (Go) for real-time classification
- Benchmark Python vs Go (accuracy, latency, resource usage)
- Document complete ML lifecycle

### Phase 2 (Future): Physical Prototype
**Hardware:**
- Arduino Mega 2560
- MPU6050 IMU
- 4× VL53L0X Laser ToF sensors
- 4× HC-SR04 Ultrasonic sensors

**Validation:**
- Test models on real robot hardware
- Collect custom dataset in warehouse environment
- Fine-tune models for domain adaptation

### Phase 3 (Future): Sensor Fusion
**Integration:**
- Combine IMU states + ToF proximity data
- Decision logic: `IF MOVING_FORWARD AND front_ToF < 30cm → EMERGENCY_STOP`
- Multi-sensor failure detection (redundancy)

---

## Learning Objectives (Academic Context)

This project focuses on:
1. **End-to-end ML pipeline:** Problem definition → deployment
2. **Language comparison:** Python (training) vs Go (production inference)
3. **Best practices:** Documentation, versioning, reproducibility
4. **Real-world constraints:** Latency vs accuracy trade-offs
5. **Evaluation rigor:** Proper train/val/test splits, cross-validation
6. **Embedded ML:** Running models on resource-constrained hardware

---

## Project Timeline

| Phase | Duration | Deliverable |
|--------|-----------|-------------|
| 1. Problem Definition | Week 1 | This document |
| 2. Data Acquisition | Week 2 | UCI HAR dataset, EDA |
| 3. Data Preparation | Week 3 | Preprocessing pipeline |
| 4–5. Modeling | Week 4–5 | Trained models, comparison |
| 6. Fine-tuning | Week 6 | Optimized model |
| 7. Go Implementation | Week 7 | Inference server, benchmarks |
| 8. Documentation | Week 8 | Final report, presentation |

### Pre-Data Acquisition Checklist
- [x] Business problem clearly defined
- [x] Success metrics established
- [x] Stakeholders identified
- [x] Use cases documented
- [x] Constraints acknowledged
- [x] Timeline agreed upon

---
**Last Updated:** November 2025  
**Status:** Approved (v2.0 - Corrected)