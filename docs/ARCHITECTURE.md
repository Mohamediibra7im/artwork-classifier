# 🏗️ System Architecture - Artwork Classification

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit)                    │
├─────────────────────────────────────────────────────────────────┤
│  Main Page  │  Comparison  │  Metrics  │  Multi-Task  │  Docs   │
└──────┬──────┴──────┬───────┴────┬──────┴──────┬───────┴────┬────┘
       │             │            │             │            │
       v             v            v             v            v
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  Image Upload  │  Preprocessing  │  Model Selection  │  Display │
└────────┬───────┴────────┬────────┴────────┬──────────┴──────┬───┘
         │                │                 │                  │
         v                v                 v                  v
┌─────────────────────────────────────────────────────────────────┐
│                       MODEL LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  VGG16  │  ResNet50  │  EfficientNet-B1  │  Swin Transformer   │
└────┬────┴──────┬─────┴────────┬──────────┴──────────┬──────────┘
     │           │              │                      │
     v           v              v                      v
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE ENGINE                             │
├─────────────────────────────────────────────────────────────────┤
│  PyTorch Runtime  │  CUDA (GPU)  │  CPU Fallback  │  Cache     │
└────────┬──────────┴──────────────┴────────────────┴────────┬────┘
         │                                                     │
         v                                                     v
┌─────────────────────────────────────────────────────────────────┐
│                   VISUALIZATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  Grad-CAM  │  Charts  │  Confusion Matrix  │  Reports         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
1. IMAGE INPUT
   └─> Upload Image / Select Sample
       │
       v
2. PREPROCESSING
   ├─> Resize to 224×224
   ├─> Convert to Tensor
   └─> Normalize (ImageNet stats)
       │
       v
3. MODEL SELECTION
   ├─> VGG16 (138M params)
   ├─> ResNet50 (25.6M params)
   ├─> EfficientNet-B1 (7.8M params)
   └─> Swin Transformer (87.8M params)
       │
       v
4. INFERENCE
   ├─> Forward Pass
   ├─> GPU/CPU Computation
   └─> Softmax Activation
       │
       v
5. POST-PROCESSING
   ├─> Top-K Predictions
   ├─> Confidence Scores
   └─> Class Names Mapping
       │
       v
6. VISUALIZATION
   ├─> Prediction Display
   ├─> Grad-CAM Heatmap
   ├─> Confidence Bars
   └─> Download Options
       │
       v
7. OUTPUT
   └─> Results to User
```

---

## Model Architecture Details

### VGG16
```
Input (224×224×3)
    ↓
Conv Block 1 (64 filters)
    ↓
Conv Block 2 (128 filters)
    ↓
Conv Block 3 (256 filters)
    ↓
Conv Block 4 (512 filters)
    ↓
Conv Block 5 (512 filters)
    ↓
Flatten
    ↓
FC 4096
    ↓
FC 4096
    ↓
FC 5 (classes)
    ↓
Softmax → Predictions
```

### ResNet50
```
Input (224×224×3)
    ↓
Conv1 (7×7, 64)
    ↓
MaxPool
    ↓
Residual Block 1 (64→256) × 3
    ↓ [skip connections]
Residual Block 2 (128→512) × 4
    ↓ [skip connections]
Residual Block 3 (256→1024) × 6
    ↓ [skip connections]
Residual Block 4 (512→2048) × 3
    ↓ [skip connections]
Global Average Pool
    ↓
FC 5 (classes)
    ↓
Softmax → Predictions
```

### EfficientNet-B1
```
Input (224×224×3)
    ↓
Stem (Conv 3×3, 32)
    ↓
MBConv Blocks (Stage 1-7)
├─> Depthwise Conv
├─> Squeeze-Excitation
└─> Project
    ↓
Conv Head (1280 features)
    ↓
Global Average Pool
    ↓
Dropout (0.2)
    ↓
FC 5 (classes)
    ↓
Softmax → Predictions
```

### Swin Transformer
```
Input (224×224×3)
    ↓
Patch Partition (4×4)
    ↓
Linear Embedding
    ↓
Swin Transformer Block 1
├─> Window Attention
└─> Shifted Window Attention
    ↓
Patch Merging
    ↓
Swin Transformer Block 2
    ↓
Patch Merging
    ↓
Swin Transformer Block 3
    ↓
Patch Merging
    ↓
Swin Transformer Block 4
    ↓
Layer Norm
    ↓
Global Average Pool
    ↓
FC 5 (classes)
    ↓
Softmax → Predictions
```

---

## Grad-CAM Pipeline

```
Input Image
    ↓
Forward Pass → Extract Activations (Last Conv Layer)
    ↓
Get Predicted Class
    ↓
Backward Pass → Compute Gradients
    ↓
Global Average Pooling on Gradients → Weights
    ↓
Weighted Sum of Activations
    ↓
ReLU (Remove negative values)
    ↓
Normalize to [0, 1]
    ↓
Resize to Original Image Size
    ↓
Apply Colormap (JET)
    ↓
Overlay on Original Image
    ↓
Display Heatmap + Overlay
```

---

## Multi-Task Learning Architecture

```
Input Image (224×224×3)
    ↓
Swin Transformer Backbone (Shared)
    ↓
Feature Extraction (1024-dim)
    ↓
    ├─────────────────┬─────────────────┐
    │                 │                 │
    v                 v                 v
Style Head       Artist Head      (Future Tasks)
    │                 │
Linear(1024→512) Linear(1024→512)
    │                 │
ReLU             ReLU
    │                 │
Dropout(0.3)     Dropout(0.3)
    │                 │
Linear(512→5)    Linear(512→10)
    │                 │
Softmax          Softmax
    │                 │
    v                 v
5 Style Classes  10 Artist Classes
```

---

## Training Pipeline

```
1. DATA LOADING
   ├─> Load WikiArt Dataset
   ├─> Split (80% train, 20% val)
   └─> Apply Transformations

2. MODEL INITIALIZATION
   ├─> Load Pre-trained Weights (ImageNet)
   ├─> Modify Final Layer (5 classes)
   └─> Move to Device (GPU/CPU)

3. TRAINING LOOP
   │
   ├─> Epoch 1-10: Frozen Backbone
   │   ├─> Forward Pass
   │   ├─> Calculate Loss (CrossEntropy)
   │   ├─> Backward Pass
   │   ├─> Update Classifier Weights Only
   │   └─> Validate
   │
   └─> Epoch 11-20: Full Fine-tuning
       ├─> Forward Pass
       ├─> Calculate Loss (CrossEntropy)
       ├─> Backward Pass
       ├─> Update All Weights
       └─> Validate

4. EVALUATION
   ├─> Calculate Metrics
   ├─> Generate Confusion Matrix
   ├─> Test Grad-CAM
   └─> Save Best Model

5. DEPLOYMENT
   └─> Save Model Weights (.pth)
```

---

## Streamlit Application Structure

```
gui/
│
├── app.py (Main Application)
│   ├─> Page Config
│   ├─> Sidebar (Model Selection)
│   ├─> Main Area (Upload & Results)
│   │   ├─> Image Upload
│   │   ├─> Prediction Display
│   │   └─> Grad-CAM Visualization
│   └─> Footer
│
└── pages/
    │
    ├── 1_📊_Model_Comparison.py
    │   ├─> Metrics Overview
    │   ├─> Performance Charts
    │   └─> Recommendations
    │
    ├── 2_📈_Evaluation_Metrics.py
    │   ├─> Confusion Matrices
    │   ├─> Per-Class Metrics
    │   └─> Misclassification Analysis
    │
    ├── 3_🎭_Multi-Task_Learning.py
    │   ├─> Style Prediction
    │   ├─> Artist Prediction
    │   └─> Combined Visualization
    │
    └── 4_📚_Documentation.py
        ├─> Project Overview
        ├─> Architecture Details
        ├─> Usage Guide
        └─> Team Information
```

---

## Performance Optimization

```
┌─────────────────────────────────────┐
│   OPTIMIZATION STRATEGIES           │
├─────────────────────────────────────┤
│                                     │
│  1. Model Caching                   │
│     └─> @st.cache_resource          │
│                                     │
│  2. Mixed Precision Training        │
│     └─> torch.cuda.amp              │
│                                     │
│  3. Progressive Unfreezing          │
│     └─> Gradual layer unfreezing    │
│                                     │
│  4. Batch Processing                │
│     └─> DataLoader batching         │
│                                     │
│  5. Model Quantization (Future)     │
│     └─> INT8 inference              │
│                                     │
└─────────────────────────────────────┘
```

---

## Technology Stack Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                            │
│  Streamlit │ HTML/CSS │ JavaScript (auto-generated)         │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                  APPLICATION LAYER                           │
│  Python 3.11+ │ Async I/O │ Caching │ Session Management   │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                    ML FRAMEWORK                              │
│  PyTorch 2.0+ │ torchvision │ timm │ CUDA/cuDNN           │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                  DATA PROCESSING                             │
│  NumPy │ Pandas │ PIL │ OpenCV │ scikit-learn             │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                   VISUALIZATION                              │
│  Matplotlib │ Seaborn │ Plotly │ Grad-CAM                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

```
Development Environment
    │
    ├─> Local Machine
    │   ├─> Python 3.11+
    │   ├─> Virtual Environment
    │   └─> Streamlit Dev Server
    │
Production Environment (Future)
    │
    ├─> Cloud Platform (AWS/Azure/GCP)
    │   ├─> Container (Docker)
    │   ├─> Load Balancer
    │   ├─> Multiple Instances
    │   └─> GPU Instances (optional)
    │
    ├─> CDN for Static Assets
    │
    └─> Database (for logging/analytics)
```

---

## Security & Privacy

```
┌─────────────────────────────────────┐
│     SECURITY MEASURES               │
├─────────────────────────────────────┤
│                                     │
│  ✓ Input Validation                 │
│  ✓ File Type Checking               │
│  ✓ Size Limits (max upload)         │
│  ✓ No Data Persistence              │
│  ✓ Secure Model Loading             │
│  ✓ Error Handling                   │
│  ✓ HTTPS (production)               │
│                                     │
└─────────────────────────────────────┘
```

---

This architecture document provides a comprehensive view of the system design, data flow, and technical implementation of the Artwork Classification project.
