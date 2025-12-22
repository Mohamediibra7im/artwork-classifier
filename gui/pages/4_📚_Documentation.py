import streamlit as st
import pandas as pd

st.set_page_config(page_title="Documentation", page_icon="📚", layout="wide")

st.title("📚 Project Documentation")
st.markdown("Complete guide to the Artwork Classification project")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📖 Overview", "🏗️ Architecture", "📊 Dataset", "🚀 Usage Guide", "👥 Team"]
)

with tab1:
    st.header("Project Overview")

    st.markdown(
        """
    ### 🎯 Objective
    
    This project implements a **Deep Learning-based Artwork Classification System** using 
    Convolutional Neural Networks (CNNs) and Transfer Learning to classify artworks into 
    different art styles.
    
    ### ✨ Key Features
    
    - **Multiple CNN Architectures**: VGG16, ResNet50, EfficientNet-B1, Swin Transformer
    - **Transfer Learning**: Pre-trained on ImageNet, fine-tuned on WikiArt
    - **Interactive GUI**: Built with Streamlit for easy testing
    - **Grad-CAM Visualization**: Explainable AI to understand model decisions
    - **Model Comparison**: Side-by-side performance analysis
    - **Multi-Task Learning**: Bonus feature for style + artist classification
    
    ### 🎓 Learning Outcomes
    
    Through this project, we have:
    
    - ✅ Designed and trained CNN-based classifiers for art datasets
    - ✅ Applied transfer learning and data augmentation techniques
    - ✅ Compared multiple network architectures
    - ✅ Developed an intuitive GUI for model deployment
    - ✅ Implemented explainability using Grad-CAM
    - ✅ Used GitHub for collaborative development
    """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            """
        **📦 Technologies**
        - Python 3.11+
        - PyTorch 2.0+
        - Streamlit
        - timm (PyTorch Image Models)
        - OpenCV
        - scikit-learn
        """
        )

    with col2:
        st.success(
            """
        **🎨 Art Styles**
        - Abstract Expressionism
        - Cubism
        - Expressionism
        - Impressionism
        - Realism
        """
        )

    with col3:
        st.warning(
            """
        **⚡ Performance**
        - Best Accuracy: 92.34%
        - Fastest Model: 10ms
        - Smallest Model: 7.8M params
        - GPU/CPU Compatible
        """
        )

with tab2:
    st.header("🏗️ Model Architectures")

    st.markdown(
        """
    We implemented and compared **4 different CNN architectures**, each with unique 
    characteristics and strengths.
    """
    )

    # VGG16
    st.subheader("1️⃣ VGG16")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
        **Architecture Details:**
        - 16 layers (13 convolutional, 3 fully connected)
        - Small 3×3 filters throughout
        - Max pooling for downsampling
        - 138M parameters
        
        **Strengths:**
        - Simple and uniform architecture
        - Good baseline performance
        - Easy to understand and modify
        
        **Weaknesses:**
        - Large model size
        - Slower inference
        - Memory intensive
        """
        )

    with col2:
        st.code(
            """
# VGG16 Implementation
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, num_classes)

# Fine-tuning strategy:
# 1. Freeze backbone
# 2. Train classifier (10 epochs)
# 3. Unfreeze all layers
# 4. Fine-tune (10 epochs)
        """,
            language="python",
        )

    st.markdown("---")

    # ResNet50
    st.subheader("2️⃣ ResNet50")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
        **Architecture Details:**
        - 50 layers with residual connections
        - Skip connections to prevent vanishing gradients
        - Bottleneck blocks for efficiency
        - 25.6M parameters
        
        **Strengths:**
        - Deeper network without degradation
        - Better feature learning
        - More efficient than VGG
        
        **Weaknesses:**
        - More complex architecture
        - Requires more training time
        """
        )

    with col2:
        st.code(
            """
# ResNet50 Implementation
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Residual Block:
# out = F.relu(x + conv2(conv1(x)))
# Skip connection allows gradient flow

# Training: Progressive unfreezing
# from final layers to earlier layers
        """,
            language="python",
        )

    st.markdown("---")

    # EfficientNet
    st.subheader("3️⃣ EfficientNet-B1")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
        **Architecture Details:**
        - Compound scaling (depth, width, resolution)
        - Mobile inverted bottleneck blocks
        - Squeeze-and-excitation optimization
        - Only 7.8M parameters
        
        **Strengths:**
        - Best accuracy-to-size ratio
        - Fast inference
        - Memory efficient
        - Production-ready
        
        **Weaknesses:**
        - Complex architecture
        - Requires careful hyperparameter tuning
        """
        )

    with col2:
        st.code(
            """
# EfficientNet-B1 Implementation
model = timm.create_model(
    'efficientnet_b1',
    pretrained=True,
    num_classes=num_classes
)

# Compound Scaling:
# - Depth: α^φ
# - Width: β^φ  
# - Resolution: γ^φ
# where α·β²·γ² ≈ 2

# Best trade-off for deployment
        """,
            language="python",
        )

    st.markdown("---")

    # Swin Transformer
    st.subheader("4️⃣ Swin Transformer")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
        **Architecture Details:**
        - Vision Transformer with shifted windows
        - Hierarchical feature maps
        - Self-attention mechanism
        - 87.8M parameters
        
        **Strengths:**
        - State-of-the-art accuracy
        - Global context understanding
        - Excellent for complex patterns
        
        **Weaknesses:**
        - Largest model size
        - Slowest inference
        - High computational requirements
        """
        )

    with col2:
        st.code(
            """
# Swin Transformer Implementation
model = timm.create_model(
    'swin_base_patch4_window7_224',
    pretrained=True,
    num_classes=num_classes
)

# Shifted Window Mechanism:
# - Local attention in windows
# - Shift windows between layers
# - Enables cross-window connections

# Best for maximum accuracy
        """,
            language="python",
        )

    st.markdown("---")

    # Comparison table
    st.subheader("📊 Architecture Comparison")

    comparison_data = {
        "Model": ["VGG16", "ResNet50", "EfficientNet-B1", "Swin Transformer"],
        "Parameters": ["138M", "25.6M", "7.8M", "87.8M"],
        "Accuracy": ["85.24%", "88.76%", "91.02%", "92.34%"],
        "Inference Time": ["15ms", "12ms", "10ms", "25ms"],
        "Best For": [
            "Baseline/Simple",
            "Balanced Performance",
            "Production Deployment",
            "Maximum Accuracy",
        ],
    }

    df = pd.DataFrame(comparison_data)
    st.table(df)

with tab3:
    st.header("📊 Dataset Information")

    st.markdown(
        """
    ### WikiArt Dataset
    
    The **WikiArt** dataset is a large-scale collection of fine-art images organized by 
    art style, artist, and genre. It's widely used for art classification tasks.
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Dataset Statistics")

        stats_data = {
            "Metric": [
                "Total Images",
                "Number of Styles",
                "Training Set",
                "Validation Set",
                "Image Resolution",
                "File Format",
            ],
            "Value": [
                "~30,000",
                "5 (top styles)",
                "80% (~24,000)",
                "20% (~6,000)",
                "224×224 (resized)",
                "JPEG/PNG",
            ],
        }

        st.table(pd.DataFrame(stats_data))

    with col2:
        st.subheader("🎨 Class Distribution")

        class_data = {
            "Art Style": [
                "Abstract Expressionism",
                "Cubism",
                "Expressionism",
                "Impressionism",
                "Realism",
            ],
            "Samples": ["~6,000", "~6,200", "~6,100", "~6,300", "~5,800"],
        }

        st.table(pd.DataFrame(class_data))

    st.markdown("---")

    st.subheader("🔄 Data Preprocessing & Augmentation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Training Augmentations:**
        - Random resizing (256×256)
        - Random cropping (224×224)
        - Random horizontal flip
        - Random rotation (±15°)
        - Color jittering
        - Random affine transformations
        - Random perspective distortion
        - Random grayscale (10%)
        - ImageNet normalization
        """
        )

    with col2:
        st.code(
            """
# Training Transform
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25
    ),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=5
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
        """,
            language="python",
        )

with tab4:
    st.header("🚀 Usage Guide")

    st.markdown(
        """
    ### Installation & Setup
    """
    )

    st.code(
        """
# 1. Clone the repository
git clone https://github.com/yourusername/artwork-classifier.git
cd artwork-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models (if not included)
# Models should be in models/ directory:
# - vgg16_top6_finetuned.pth
# - resnet50_top6_finetuned.pth
# - efficientnet_b1_resumed_10ep.pth
# - swin_artist_model.pth
# - multitask_swin_model.pth (optional)

# 4. Run the application
streamlit run gui/app.py
    """,
        language="bash",
    )

    st.markdown("---")

    st.subheader("📱 Using the Application")

    st.markdown(
        """
    #### Main Classification Page
    
    1. **Select Model**: Choose from VGG16, ResNet50, EfficientNet-B1, or Swin Transformer
    2. **Upload Image**: Click "Browse files" or drag and drop an artwork image
    3. **View Predictions**: See top-5 predictions with confidence scores
    4. **Explore Grad-CAM**: Visualize what the model is focusing on
    5. **Download Results**: Save Grad-CAM visualizations for later analysis
    
    #### Model Comparison Page
    
    - View side-by-side performance metrics
    - Compare accuracy, speed, and model size
    - Identify the best model for your use case
    - Interactive charts and visualizations
    
    #### Evaluation Metrics Page
    
    - Detailed confusion matrices for each model
    - Per-class precision, recall, and F1-score
    - Misclassification analysis
    - Interactive confusion matrix exploration
    
    #### Multi-Task Learning Page (Bonus)
    
    - Predict both art style AND artist simultaneously
    - Single model for dual tasks
    - Compare predictions across tasks
    - Download detailed analysis reports
    """
    )

    st.markdown("---")

    st.subheader("💡 Tips & Best Practices")

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            """
        **For Best Results:**
        - Use clear, well-lit images
        - Avoid heavily cropped artworks
        - Ensure the artwork is the main subject
        - Use images with resolution ≥224×224
        - JPEG or PNG format recommended
        """
        )

    with col2:
        st.info(
            """
        **Model Selection Guide:**
        - **VGG16**: Good baseline, educational
        - **ResNet50**: Balanced performance
        - **EfficientNet-B1**: Best for production
        - **Swin Transformer**: Maximum accuracy
        """
        )

with tab5:
    st.header("👥 Team Information")

    st.markdown(
        """
    ### Team Mohammed - AI Skills Project 2025
    
    This project was developed collaboratively by a team of 6 members, each contributing 
    to different aspects of the system.
    """
    )

    st.markdown("---")

    # Team members with roles
    team_data = {
        "Name": [
            "Mohammed Ibrahim",
            "Mohammed Elfouly",
            "Mohammed Ashraf",
            "Mohammed Mohsen",
            "Sarah Mahmoud",
            "Rahma Nasser",
        ],
    }

    df_team = pd.DataFrame(team_data)

    for idx, row in df_team.iterrows():
        with st.expander(f"👤 {row['Name']}"):
            st.markdown(
                """
            **Technical Skills:**
            - Python, PyTorch, Deep Learning
            - Data preprocessing and visualization
            - Git/GitHub for version control
            """
            )

    st.markdown("---")

    st.subheader("📊 Project Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Team Size", "6 Members")

    with col2:
        st.metric("Models Trained", "4 Architectures")

    with col3:
        st.metric("Code Files", "20+ Files")

    with col4:
        st.metric("Project Duration", "4 Weeks")

    st.markdown("---")

    st.subheader("🏆 Project Achievements")

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            """
        **Technical Achievements:**
        - ✅ 4 CNN architectures implemented
        - ✅ 92.34% best accuracy achieved
        - ✅ Transfer learning successfully applied
        - ✅ Grad-CAM explainability integrated
        - ✅ Multi-task learning (bonus feature)
        - ✅ Production-ready GUI deployed
        """
        )

    with col2:
        st.info(
            """
        **Collaboration & Documentation:**
        - ✅ Well-structured GitHub repository
        - ✅ All members contributed commits
        - ✅ Comprehensive README documentation
        - ✅ Clean code with comments
        - ✅ Detailed project report
        - ✅ Interactive visualizations
        """
        )

    st.markdown("---")

    st.subheader("📞 Contact & Resources")

    st.markdown(
        """
    **GitHub Repository:** https://github.com/yourusername/artwork-classifier
    
    **Project Report:** Available in `docs/` directory
    
    **Trained Models:** Available in `models/` directory
    
    **Dataset Source:** WikiArt - https://www.wikiart.org/
    
    ---
    
    **Course:** Basic of AI Programming Skills
    
    **Semester:** 5 (2025)

    """
    )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #262730;'>
    <p><strong>Artwork Classification Project</strong></p>
    <p>Developed with ❤️ using Python, PyTorch, and Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
