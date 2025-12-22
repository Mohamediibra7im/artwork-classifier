# 🚀 Setup Guide - Artwork Classification System

This guide will help you set up and run the Artwork Classification System on your machine.

---

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.11 or higher** installed
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- (Optional) **CUDA-capable GPU** for faster inference
- At least **2GB free disk space**

---

## 🔧 Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/artwork-classifier.git

# Navigate to the project directory
cd artwork-classifier
```

### Step 2: Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

This will install:
- PyTorch and torchvision
- Streamlit
- timm (PyTorch Image Models)
- OpenCV
- matplotlib, seaborn, plotly
- scikit-learn
- And other dependencies

### Step 4: Verify Installation

```bash
python -c "import torch; import streamlit; import timm; print('✅ All dependencies installed successfully!')"
```

If you see the success message, you're ready to go!

---

## 🎬 Running the Application

### Method 1: Using the Run Script

**On Windows:**
```bash
run_app.bat
```

**On macOS/Linux:**
```bash
chmod +x run_app.sh
./run_app.sh
```

### Method 2: Using Streamlit Command

```bash
streamlit run gui/app.py
```

The application will automatically open in your default browser at:
```
http://localhost:8501
```

---

## 📱 Using the Application

### 1. Main Classification Page

1. **Select a Model** from the sidebar:
   - VGG16
   - ResNet50
   - EfficientNet-B1
   - Swin Transformer

2. **Upload an Image**:
   - Click "Browse files" or drag and drop
   - Supported formats: JPG, JPEG, PNG
   - Or use sample images provided

3. **View Results**:
   - Top prediction with confidence score
   - Top-5 predictions
   - Grad-CAM heatmap visualization

4. **Download**:
   - Save Grad-CAM visualizations

### 2. Model Comparison Page

- Compare performance metrics across all models
- View accuracy, speed, and model size
- Interactive charts and visualizations
- Find the best model for your needs

### 3. Evaluation Metrics Page

- Confusion matrices for each model
- Per-class precision, recall, F1-score
- Misclassification analysis
- Interactive exploration

### 4. Multi-Task Learning Page (Bonus)

- Predict both art style AND artist
- Dual-task predictions from single model
- Download analysis reports

### 5. Documentation Page

- Complete project documentation
- Architecture details
- Dataset information
- Team information

---

## 🔍 Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: CUDA out of memory

**Solution:**
The app automatically falls back to CPU if GPU is unavailable or out of memory.

### Issue: Streamlit won't start

**Solution:**
```bash
# Check if port 8501 is already in use
streamlit run gui/app.py --server.port 8502
```

### Issue: Model files not found

**Solution:**
Ensure all model files are in the `models/` directory:
- `vgg16_top6_finetuned.pth`
- `resnet50_top6_finetuned.pth`
- `efficientnet_b1_resumed_10ep.pth`
- `swin_artist_model.pth`
- `multitask_swin_model.pth`

### Issue: Import errors on Windows

**Solution:**
```bash
# Install Visual C++ Build Tools if needed
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

---

## ⚙️ Configuration

### Streamlit Configuration

Create `.streamlit/config.toml` for custom settings:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
port = 8501
```

### Model Configuration

Edit `gui/app.py` to change:
- Class names
- Model paths
- Image preprocessing parameters

---

## 📊 Testing with Sample Images

Sample images are provided in `test_images/test_images/`:

1. Check the "Or use a sample image" checkbox
2. Select from the dropdown menu
3. View predictions immediately

---

## 🐳 Docker Deployment (Optional)

If you want to deploy using Docker:

```dockerfile
# Create Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "gui/app.py"]
```

Build and run:
```bash
docker build -t artwork-classifier .
docker run -p 8501:8501 artwork-classifier
```

---

## 🔐 Security Notes

- Do not commit API keys or credentials
- Keep model files secure
- Use HTTPS in production
- Sanitize user inputs

---

## 📈 Performance Tips

1. **Use GPU**: Significantly faster inference
2. **Batch Processing**: Process multiple images together
3. **Model Selection**: Use EfficientNet-B1 for best speed/accuracy
4. **Image Size**: Resize large images before upload
5. **Cache Models**: Models are cached after first load

---

## 🆘 Getting Help

- **GitHub Issues**: https://github.com/yourusername/artwork-classifier/issues
- **Documentation**: See `gui/pages/4_📚_Documentation.py`
- **Contact Team**: See README.md for team contacts

---

## ✅ Checklist

Before running the application, ensure:

- [ ] Python 3.11+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Model files present in `models/` directory
- [ ] Virtual environment activated (recommended)
- [ ] Port 8501 is available
- [ ] No firewall blocking localhost

---

## 🎉 Success!

If everything is set up correctly, you should see:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Enjoy using the Artwork Classification System! 🎨

---

**Need help?** Open an issue on GitHub or contact the team.
