# 🎨 Artwork Classification Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An advanced **Deep Learning system** for classifying artworks into different art styles using state-of-the-art CNN architectures and Transfer Learning. This project includes a comprehensive **Streamlit web application** with Grad-CAM visualizations, model comparison tools, and multi-task learning capabilities.

---

## 🎯 Overview

This project implements a complete **artwork classification system** as part of the AI Skills course. We train and compare **4 different CNN architectures** (VGG16, ResNet50, EfficientNet-B1, Swin Transformer) on the WikiArt dataset to classify artworks into 5 major art styles.

### Project Goals

- ✅ Classify artwork types and styles using deep learning
- ✅ Train and compare multiple CNN architectures
- ✅ Evaluate models using accuracy, precision, recall, and confusion matrices
- ✅ Implement Grad-CAM for explainable AI
- ✅ Build an interactive Streamlit web application
- ✅ Develop multi-task learning for style + artist classification (bonus)

---

## ✨ Features

### Core Features

- **🤖 Multiple CNN Architectures**: VGG16, ResNet50, EfficientNet-B1, Swin Transformer
- **🔄 Transfer Learning**: Pre-trained on ImageNet, fine-tuned on WikiArt
- **📊 Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices
- **🔥 Grad-CAM Visualization**: Explainable AI showing what the model focuses on
- **💻 Interactive GUI**: Beautiful Streamlit web application

### Advanced Features

- **📈 Model Comparison Dashboard**: Side-by-side performance analysis
- **🎭 Multi-Task Learning**: Predict both art style AND artist simultaneously (bonus)
- **📥 Export Capabilities**: Download Grad-CAM visualizations and analysis reports
- **🖼️ Sample Images**: Pre-loaded test images for quick testing
- **⚡ Real-time Inference**: Fast predictions with optimized models

---

## 📁 Project Structure

```
artwork-classifier/
│
├── gui/                          # Streamlit web application
│   ├── app.py                    # Main application (classification)
│   └── pages/                    # Multi-page app sections
│       ├── 1_📊_Model_Comparison.py
│       ├── 2_📈_Evaluation_Metrics.py
│       ├── 3_🎭_Multi-Task_Learning.py
│       └── 4_📚_Documentation.py
│
├── models/                       # Trained model weights
│   ├── vgg16_top6_finetuned.pth
│   ├── resnet50_top6_finetuned.pth
│   ├── efficientnet_b1_resumed_10ep.pth
│   ├── swin_artist_model.pth
│   └── multitask_swin_model.pth
│
├── notebooks/                    # Jupyter notebooks for training
│   ├── Artwork-Classification.ipynb
│   ├── Multi-Task.ipynb
│   └── wikiart.ipynb
│
├── test_images/                  # Sample images for testing
│   └── test_images/
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── Term project.pdf              # Project requirements

```

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/artwork-classifier.git
cd artwork-classifier
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; import streamlit; import timm; print('All dependencies installed successfully!')"
```

---

## 🎬 Quick Start

### Launch the Web Application

```bash
streamlit run gui/app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Select a Model** from the sidebar (VGG16, ResNet50, EfficientNet-B1, or Swin Transformer)
2. **Upload an Image** or select a sample image
3. **View Predictions** with confidence scores
4. **Explore Grad-CAM** visualizations to understand model decisions
5. **Navigate Pages** to compare models, view metrics, or try multi-task learning

---

## 🧠 Models

### 1. VGG16

- **Parameters**: 138M
- **Accuracy**: 85.24%
- **Inference**: ~15ms
- **Best For**: Baseline comparison, educational purposes

### 2. ResNet50

- **Parameters**: 25.6M
- **Accuracy**: 88.76%
- **Inference**: ~12ms
- **Best For**: Balanced performance

### 3. EfficientNet-B1 ⭐

- **Parameters**: 7.8M
- **Accuracy**: 91.02%
- **Inference**: ~10ms
- **Best For**: Production deployment, efficiency

### 4. Swin Transformer 🏆

- **Parameters**: 87.8M
- **Accuracy**: 92.34%
- **Inference**: ~25ms
- **Best For**: Maximum accuracy

---

## 📊 Dataset

### WikiArt Dataset

We use a curated subset of the **WikiArt** dataset focusing on the top 5 most populated art styles:

- **Abstract Expressionism** (~6,000 images)
- **Cubism** (~6,200 images)
- **Expressionism** (~6,100 images)
- **Impressionism** (~6,300 images)
- **Realism** (~5,800 images)

**Total**: ~30,000 images  
**Split**: 80% training / 20% validation

### Data Augmentation

- Random resizing and cropping
- Random horizontal flips
- Random rotation (±15°)
- Color jittering
- Affine transformations
- Perspective distortion
- ImageNet normalization

---

## 📈 Results

### Model Comparison

| Model            | Accuracy   | Precision  | Recall     | F1-Score   | Parameters | Speed    |
| ---------------- | ---------- | ---------- | ---------- | ---------- | ---------- | -------- |
| VGG16            | 85.24%     | 85.12%     | 85.24%     | 85.18%     | 138M       | 15ms     |
| ResNet50         | 88.76%     | 88.65%     | 88.76%     | 88.70%     | 25.6M      | 12ms     |
| EfficientNet-B1  | **91.02%** | 90.95%     | 91.02%     | 90.98%     | **7.8M**   | **10ms** |
| Swin Transformer | **92.34%** | **92.28%** | **92.34%** | **92.31%** | 87.8M      | 25ms     |

### Key Findings

- ✅ **Swin Transformer** achieves the highest accuracy (92.34%)
- ✅ **EfficientNet-B1** offers the best efficiency-accuracy trade-off
- ✅ Transfer learning significantly improves performance
- ✅ Data augmentation reduces overfitting
- ✅ Grad-CAM confirms models focus on relevant artistic features

---

## 💡 Usage

### Command Line Interface

```python
# Example: Load model and make prediction
import torch
from torchvision import transforms, models
from PIL import Image

# Load model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load('models/resnet50_top6_finetuned.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('path/to/artwork.jpg')
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = probabilities.argmax(1).item()

print(f"Predicted class: {predicted_class}")
```

### Web Application Pages

1. **🎨 Main Page**: Upload and classify artworks
2. **📊 Model Comparison**: Compare all models side-by-side
3. **📈 Evaluation Metrics**: View detailed confusion matrices
4. **🎭 Multi-Task Learning**: Predict style + artist (bonus feature)
5. **📚 Documentation**: Complete project documentation

---

## 👥 Team

**AI Skills Project Team 2025**

| Name                 |
| -------------------- |
| **Mohammed Ibrahim** |
| **Mohammed Elfouly** |
| **Mohammed Ashraf**  |
| **Mohammed Mohsen**  |
| **Sarah Mahmoud**    |
| **Rahma Nasser**     |

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Guidelines

- Write clear commit messages
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Test your changes before submitting
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **WikiArt** for providing the dataset
- **PyTorch** team for the deep learning framework
- **Streamlit** for the amazing web framework
- **timm** (PyTorch Image Models) for pre-trained models
- Our instructor and classmates for valuable feedback

---

## 📞 Contact

**Project Repository**: https://github.com/yourusername/artwork-classifier

**Issues & Questions**: Open an issue on GitHub

**Course**: Basic of AI Programming Skills - Semester 5 (2025)

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

</div>
