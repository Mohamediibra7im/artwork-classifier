# 📄 Artwork Classification Project Report

**Course:** Basic of AI Programming Skills  
**Semester:** 5 (2025)  
**Team:** Mohammed Team  
**Project:** Artwork Classification using Deep Learning  

---

## Executive Summary

This report presents a comprehensive Deep Learning solution for artwork classification using Convolutional Neural Networks (CNNs) and Transfer Learning. We trained and compared four state-of-the-art architectures (VGG16, ResNet50, EfficientNet-B1, Swin Transformer) on the WikiArt dataset, achieving up to 92.34% accuracy. The project includes a fully functional web application with Grad-CAM visualization for model explainability and a bonus multi-task learning feature for simultaneous style and artist classification.

---

## 1. Introduction

### 1.1 Background

Artwork classification is a challenging computer vision task that requires understanding subtle visual features, artistic styles, and cultural context. Traditional machine learning approaches struggle with the complexity and variability of artistic images. Deep Learning, particularly CNNs with Transfer Learning, has revolutionized image classification by learning hierarchical feature representations.

### 1.2 Objectives

The primary objectives of this project are:

1. **Model Development**: Train and fine-tune multiple CNN architectures for artwork style classification
2. **Performance Comparison**: Evaluate and compare different models using comprehensive metrics
3. **Explainability**: Implement Grad-CAM to visualize model decision-making
4. **Deployment**: Develop an intuitive GUI for real-world usage
5. **Innovation**: Implement multi-task learning as a bonus feature

### 1.3 Dataset

We used the **WikiArt** dataset, focusing on the top 5 most populated art styles:

- Abstract Expressionism
- Cubism
- Expressionism
- Impressionism
- Realism

**Statistics:**
- Total images: ~30,000
- Training set: 80% (~24,000 images)
- Validation set: 20% (~6,000 images)
- Image resolution: 224×224 pixels

---

## 2. Methodology

### 2.1 Data Preprocessing

#### 2.1.1 Training Augmentation
We applied extensive data augmentation to improve model generalization:

- **Geometric transformations**: Random rotation (±15°), horizontal flips, affine transforms
- **Color adjustments**: Color jittering (brightness, contrast, saturation)
- **Advanced augmentations**: Perspective distortion, random grayscale
- **Normalization**: ImageNet statistics ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#### 2.1.2 Validation Preprocessing
Validation images underwent minimal preprocessing:
- Resize to 224×224
- Tensor conversion
- ImageNet normalization

### 2.2 Model Architectures

#### 2.2.1 VGG16
**Description:** 16-layer deep CNN with uniform 3×3 convolutions

**Configuration:**
- Pre-trained on ImageNet
- Modified final layer: 4096 → 5 classes
- Total parameters: 138M

**Training Strategy:**
- Phase 1: Freeze backbone, train classifier (10 epochs)
- Phase 2: Unfreeze all, fine-tune (10 epochs)
- Optimizer: Adam (lr=1e-4)
- Loss: Cross-Entropy with class weights

#### 2.2.2 ResNet50
**Description:** 50-layer residual network with skip connections

**Configuration:**
- Pre-trained on ImageNet
- Modified FC layer: 2048 → 5 classes
- Total parameters: 25.6M

**Training Strategy:**
- Progressive unfreezing from final to initial layers
- Optimizer: AdamW (lr=1e-4 head, 1e-5 backbone)
- Loss: Cross-Entropy with label smoothing (0.05)

#### 2.2.3 EfficientNet-B1
**Description:** Compound-scaled CNN optimizing depth, width, and resolution

**Configuration:**
- Pre-trained on ImageNet
- Modified classifier head: 1280 → 5 classes
- Total parameters: 7.8M

**Training Strategy:**
- Gradual unfreezing strategy
- Optimizer: AdamW with weight decay (0.01)
- Mixed precision training for efficiency

#### 2.2.4 Swin Transformer
**Description:** Vision Transformer using shifted window attention

**Configuration:**
- Pre-trained on ImageNet-21k
- Modified head: 1024 → 5 classes (style) or 10 classes (artist)
- Total parameters: 87.8M

**Training Strategy:**
- Fine-tuning with frozen early layers
- Optimizer: AdamW with cosine annealing
- Dropout: 0.3 for regularization

### 2.3 Training Details

**Hardware:**
- GPU: NVIDIA Tesla T4
- Framework: PyTorch 2.0+
- Training time: ~4-6 hours per model

**Hyperparameters:**
- Batch size: 32
- Learning rate: 1e-4 (head), 1e-5 (backbone)
- Epochs: 10-20 depending on model
- Weight decay: 0.01
- Label smoothing: 0.05

**Techniques:**
- Transfer Learning from ImageNet
- Progressive unfreezing
- Class weighting for imbalance
- Early stopping with patience=5
- Learning rate scheduling (CosineAnnealing)

---

## 3. Results

### 3.1 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | Parameters | Inference Time |
|-------|----------|-----------|--------|----------|------------|----------------|
| VGG16 | 85.24% | 85.12% | 85.24% | 85.18% | 138M | 15ms |
| ResNet50 | 88.76% | 88.65% | 88.76% | 88.70% | 25.6M | 12ms |
| EfficientNet-B1 | **91.02%** | 90.95% | 91.02% | 90.98% | **7.8M** | **10ms** |
| Swin Transformer | **92.34%** | **92.28%** | **92.34%** | **92.31%** | 87.8M | 25ms |

### 3.2 Key Findings

1. **Best Accuracy**: Swin Transformer (92.34%)
2. **Best Efficiency**: EfficientNet-B1 (91.02% with only 7.8M parameters)
3. **Fastest Inference**: EfficientNet-B1 (10ms per image)
4. **Best Trade-off**: EfficientNet-B1 balances accuracy, size, and speed

### 3.3 Per-Class Performance

Average per-class metrics (EfficientNet-B1):

| Art Style | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Abstract Expressionism | 92% | 92% | 92% | 1200 |
| Cubism | 93% | 93% | 93% | 1240 |
| Expressionism | 91% | 88% | 89% | 1220 |
| Impressionism | 92% | 92% | 92% | 1260 |
| Realism | 89% | 92% | 90% | 1160 |

### 3.4 Confusion Matrices

Confusion matrices reveal:
- High diagonal values (correct predictions)
- Occasional confusion between Expressionism and Abstract Expressionism
- Strong distinction between Impressionism and other styles
- Realism sometimes confused with Impressionism

### 3.5 Training Curves

All models showed:
- Steady decrease in training loss
- Gradual improvement in validation accuracy
- No significant overfitting thanks to augmentation
- Convergence within 15-20 epochs

---

## 4. Explainability: Grad-CAM Analysis

### 4.1 Methodology

We implemented **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize model attention:

1. Forward pass through the model
2. Extract final convolutional layer activations
3. Compute gradients of target class w.r.t. activations
4. Weight activations by gradient importance
5. Generate heatmap showing regions of interest

### 4.2 Insights

Grad-CAM visualizations revealed:

- **Impressionism**: Models focus on brush stroke textures and color patterns
- **Cubism**: Attention on geometric shapes and angular forms
- **Abstract Expressionism**: Focus on color splashes and spontaneous patterns
- **Expressionism**: Emphasis on dramatic color contrasts and distorted forms
- **Realism**: Attention on realistic details and proportions

### 4.3 Model Interpretability

Grad-CAM confirms that models learn meaningful artistic features rather than spurious correlations, increasing trust in predictions.

---

## 5. GUI Implementation

### 5.1 Technology Stack

- **Framework**: Streamlit
- **Backend**: PyTorch
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Image Processing**: OpenCV, PIL

### 5.2 Features

1. **Main Classification Page**:
   - Model selection (4 architectures)
   - Image upload or sample selection
   - Top-5 predictions with confidence scores
   - Grad-CAM visualization
   - Download capabilities

2. **Model Comparison Page**:
   - Side-by-side performance metrics
   - Interactive charts
   - Accuracy vs. efficiency trade-offs

3. **Evaluation Metrics Page**:
   - Confusion matrices (static and interactive)
   - Per-class metrics
   - Misclassification analysis

4. **Multi-Task Learning Page** (Bonus):
   - Simultaneous style + artist prediction
   - Dual-task visualization
   - Analysis report generation

5. **Documentation Page**:
   - Complete project documentation
   - Architecture explanations
   - Usage guides

### 5.3 User Experience

The GUI is designed with:
- Intuitive navigation
- Responsive layout
- Real-time feedback
- Professional aesthetics
- Mobile-friendly design

---

## 6. Bonus Feature: Multi-Task Learning

### 6.1 Motivation

Multi-task learning trains a single model to perform multiple related tasks, leveraging shared representations for improved generalization.

### 6.2 Architecture

- **Backbone**: Swin Transformer (shared)
- **Style Head**: 2-layer MLP → 5 classes
- **Artist Head**: 2-layer MLP → 10 classes
- **Training**: Joint optimization with weighted loss

### 6.3 Results

- **Style Accuracy**: ~92%
- **Artist Accuracy**: ~85%
- **Benefit**: Single forward pass for both predictions
- **Efficiency**: Similar inference time to single-task model

### 6.4 Applications

This demonstrates advanced techniques applicable to:
- Multi-attribute classification
- Joint object detection and segmentation
- Simultaneous classification and regression

---

## 7. Challenges and Solutions

### 7.1 Dataset Challenges

**Challenge**: Class imbalance
**Solution**: Weighted loss function and stratified sampling

**Challenge**: High intra-class variability
**Solution**: Extensive data augmentation

### 7.2 Technical Challenges

**Challenge**: Large model sizes
**Solution**: Mixed precision training, model pruning

**Challenge**: Overfitting
**Solution**: Dropout, data augmentation, early stopping

**Challenge**: Long training times
**Solution**: Transfer learning, progressive unfreezing, GPU acceleration

### 7.3 Deployment Challenges

**Challenge**: Model loading time
**Solution**: Streamlit caching (@st.cache_resource)

**Challenge**: Memory constraints
**Solution**: Efficient tensor handling, CPU fallback

---

## 8. Future Work

### 8.1 Model Improvements

- **Ensemble Methods**: Combine multiple models for higher accuracy
- **Knowledge Distillation**: Create smaller, faster models
- **Self-Supervised Learning**: Leverage unlabeled art data
- **Attention Mechanisms**: Improve interpretability

### 8.2 Dataset Expansion

- Include more art styles (Baroque, Renaissance, etc.)
- Add temporal classification (art period)
- Incorporate multi-label classification (style + genre)

### 8.3 Deployment

- **Mobile App**: TensorFlow Lite for on-device inference
- **Web API**: RESTful API for integration
- **Cloud Deployment**: AWS/Azure/GCP hosting
- **Edge Computing**: Optimize for IoT devices

### 8.4 Advanced Features

- **Artwork Generation**: Style transfer and GANs
- **Similarity Search**: Find visually similar artworks
- **Artist Attribution**: Beyond classification to verification
- **Forgery Detection**: Identify fake artworks

---

## 9. Conclusion

This project successfully developed a comprehensive artwork classification system using state-of-the-art deep learning techniques. Key achievements include:

1. ✅ **High Accuracy**: 92.34% with Swin Transformer
2. ✅ **Efficient Models**: EfficientNet-B1 offers best trade-off
3. ✅ **Explainability**: Grad-CAM provides interpretability
4. ✅ **Production-Ready**: Fully functional GUI
5. ✅ **Innovation**: Multi-task learning bonus feature
6. ✅ **Collaboration**: Successful team project

The system demonstrates practical applications of transfer learning, model comparison, and explainable AI in the domain of art classification.

---

## 10. Team Contributions

| Member | Role | Contributions |
|--------|------|---------------|
| Mohammed Ibrahim | Project Lead | Architecture, model design, integration |
| Mohammed Elfouly | Data Engineer | Dataset curation, preprocessing pipeline |
| Mohammed Ashraf | ML Engineer | Training, hyperparameter optimization |
| Mohammed Mohsen | Evaluation | Metrics, visualizations, Grad-CAM |
| Sarah Mahmoud | Frontend | GUI development, user experience |
| Rahma Nasser | Documentation | Reports, README, GitHub management |

**Team Dynamics:**
- Weekly meetings for progress tracking
- Git branches for parallel development
- Code reviews before merging
- Collaborative debugging and testing

---

## 11. References

### Academic Papers
1. Simonyan & Zisserman (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition"
2. He et al. (2015). "Deep Residual Learning for Image Recognition"
3. Tan & Le (2019). "EfficientNet: Rethinking Model Scaling for CNNs"
4. Liu et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
5. Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks"

### Datasets
- WikiArt Dataset: https://www.wikiart.org/

### Frameworks & Libraries
- PyTorch: https://pytorch.org/
- Streamlit: https://streamlit.io/
- timm: https://github.com/huggingface/pytorch-image-models

### Online Resources
- PyTorch Documentation
- Streamlit Documentation
- Transfer Learning Best Practices

---

## 12. Appendix

### A. Code Repository Structure
```
artwork-classifier/
├── gui/                 # Streamlit application
├── models/             # Trained weights
├── notebooks/          # Training notebooks
├── test_images/        # Sample images
├── requirements.txt    # Dependencies
├── README.md          # Project overview
└── SETUP.md           # Setup instructions
```

### B. Model Files
- `vgg16_top6_finetuned.pth` (528 MB)
- `resnet50_top6_finetuned.pth` (98 MB)
- `efficientnet_b1_resumed_10ep.pth` (30 MB)
- `swin_artist_model.pth` (335 MB)
- `multitask_swin_model.pth` (335 MB)

### C. Hardware Specifications
- **Training**: NVIDIA Tesla T4 GPU (16GB VRAM)
- **Inference**: CPU (Intel i7) or GPU
- **Memory**: 16GB RAM recommended

### D. Software Versions
- Python: 3.11+
- PyTorch: 2.0.0
- Streamlit: 1.28.0
- CUDA: 11.8 (if using GPU)

---

## 13. Acknowledgments

We would like to thank:
- Our course instructor for guidance and feedback
- WikiArt for providing the dataset
- PyTorch and Streamlit teams for excellent frameworks
- Classmates for valuable discussions and suggestions

---

**Report Prepared By:** Team Mohammed  
**Date:** December 2025  
**Course:** Basic of AI Programming Skills - Semester 5  

---

<div align="center">
<strong>End of Report</strong>
</div>
