import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import timm
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Artwork Classification",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    /* Fix text visibility */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #262730 !important;
    }
    
    /* Ensure dark text on light backgrounds */
    [data-testid="stMarkdownContainer"] {
        color: #262730 !important;
    }
    
    .main-header {
        font-size: 3rem;
        color: #FF6B6B !important;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4 !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #262730 !important;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 5px;
        height: 30px;
        margin: 5px 0;
    }
    
    /* Fix info boxes */
    .stAlert {
        color: #262730 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Define class names for the 6 art styles
CLASS_NAMES = [
    "Abstract_Expressionism",
    "Cubism",
    "Expressionism",
    "Impressionism",
    "Realism",
]

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Image preprocessing
def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# Load models
@st.cache_resource
def load_vgg16():
    """Load VGG16 model"""
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, len(CLASS_NAMES))
    model.load_state_dict(
        torch.load("models/vgg16_top6_finetuned.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_resnet50():
    """Load ResNet50 model"""
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(
        torch.load("models/resnet50_top6_finetuned.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_efficientnet():
    """Load EfficientNet-B1 model"""
    try:
        # Try torchvision version first (has 'features.' in keys)
        from torchvision.models import efficientnet_b1

        model = efficientnet_b1(pretrained=False)
        # Modify classifier for 5 classes
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, len(CLASS_NAMES)
        )
        model.load_state_dict(
            torch.load("models/efficientnet_b1_resumed_10ep.pth", map_location=DEVICE)
        )
        model.to(DEVICE)
        model.eval()
        return model
    except:
        # Fallback to timm version
        model = timm.create_model(
            "efficientnet_b1", pretrained=False, num_classes=len(CLASS_NAMES)
        )
        model.load_state_dict(
            torch.load("models/efficientnet_b1_resumed_10ep.pth", map_location=DEVICE)
        )
        model.to(DEVICE)
        model.eval()
        return model


@st.cache_resource
def load_swin_transformer():
    """Load Swin Transformer model - modified for style classification"""
    # Try different Swin variants
    for model_size in ["small", "tiny", "base"]:
        try:
            # Load with 10 classes (original)
            model = timm.create_model(
                f"swin_{model_size}_patch4_window7_224",
                pretrained=False,
                num_classes=10,
            )
            model.load_state_dict(
                torch.load("models/swin_artist_model.pth", map_location=DEVICE)
            )

            # Replace head for 5 style classes
            if hasattr(model, "head"):
                if hasattr(model.head, "fc"):
                    model.head.fc = nn.Linear(
                        model.head.fc.in_features, len(CLASS_NAMES)
                    )
                else:
                    model.head = nn.Linear(model.head.in_features, len(CLASS_NAMES))

            model.to(DEVICE)
            model.eval()
            return model, model_size
        except:
            continue

    st.error("Could not load Swin Transformer")
    return None, None


# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class


def get_target_layer(model, model_name):
    """Get the target layer for Grad-CAM based on model architecture"""
    if model_name == "VGG16":
        return model.features[-1]
    elif model_name == "ResNet50":
        return model.layer4[-1]
    elif model_name == "EfficientNet-B1":
        # Check if it's torchvision or timm version
        if hasattr(model, "features"):
            # Torchvision version
            return model.features[-1]
        else:
            # Timm version
            return model.conv_head
    elif model_name == "Swin Transformer":
        # Swin transformer
        if hasattr(model, "layers"):
            return model.layers[-1]
        return None
    return None


def apply_gradcam_overlay(img, cam, alpha=0.5):
    """Apply Grad-CAM heatmap overlay on image"""
    # Resize CAM to match image size
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = heatmap * alpha + img * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return heatmap, overlay


def predict_image(model, image, model_name):
    """Make prediction and generate Grad-CAM"""
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidences, indices = torch.topk(probabilities, k=min(5, len(CLASS_NAMES)))

    # Generate Grad-CAM
    target_layer = get_target_layer(model, model_name)
    if target_layer is not None:
        gradcam = GradCAM(model, target_layer)
        cam, predicted_class = gradcam.generate_cam(input_tensor)
    else:
        cam = None
        predicted_class = indices[0, 0].item()

    # Prepare results
    results = []
    for conf, idx in zip(confidences[0], indices[0]):
        results.append(
            {"class": CLASS_NAMES[idx.item()], "confidence": conf.item() * 100}
        )

    return results, cam, predicted_class


# Main app
def main():
    st.markdown(
        '<div class="main-header">🎨 Artwork Classification System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">AI-Powered Art Style Recognition using Deep Learning</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        model_name = st.selectbox(
            "Select Model",
            ["VGG16", "ResNet50", "EfficientNet-B1"],
            help="Choose the CNN architecture for style classification",
        )

        st.info(
            "💡 **Swin Transformer** for artist + style classification is in the **🎭 Multi-Task Learning** page!"
        )

        st.markdown("---")

        # Information
        st.header("📊 Model Information")
        if model_name == "VGG16":
            st.info("**VGG16**: 16-layer deep CNN known for simplicity and depth.")
        elif model_name == "ResNet50":
            st.info(
                "**ResNet50**: 50-layer network with residual connections for better gradient flow."
            )
        else:  # EfficientNet-B1
            st.info(
                "**EfficientNet-B1**: Compound scaling method balancing depth, width, and resolution. Best for style classification!"
            )

        st.markdown("---")

        # Class information
        st.header("🎭 Art Styles")
        for i, class_name in enumerate(CLASS_NAMES, 1):
            st.write(f"{i}. {class_name.replace('_', ' ')}")

        st.markdown("---")

        # About
        with st.expander("ℹ️ About"):
            st.write(
                """
            This application classifies artworks into different art styles using 
            state-of-the-art deep learning models trained on the WikiArt dataset.
            
            **Features:**
            - Multiple CNN architectures
            - Top-5 predictions with confidence scores
            - Grad-CAM visualizations
            - Real-time inference
            """
            )

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")

        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["📁 Upload File", "📷 Use Webcam", "🖼️ Sample Image"],
            horizontal=True,
            help="Choose how to provide the artwork image",
        )

        uploaded_file = None

        if input_method == "📁 Upload File":
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an artwork image...",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of an artwork",
            )

        elif input_method == "📷 Use Webcam":
            # Webcam input
            camera_image = st.camera_input(
                "Take a picture of the artwork",
                help="Position the artwork in front of your camera and click to capture",
            )
            if camera_image is not None:
                uploaded_file = camera_image

        else:  # Sample Image
            import os

            test_images_dir = "test_images/test_images"
            if os.path.exists(test_images_dir):
                sample_images = [
                    f
                    for f in os.listdir(test_images_dir)
                    if f.endswith((".jpg", ".jpeg", ".png"))
                ]
                if sample_images:
                    selected_sample = st.selectbox(
                        "Select a sample image:", sample_images
                    )
                    if selected_sample:
                        uploaded_file = os.path.join(test_images_dir, selected_sample)
                else:
                    st.warning("No sample images found in the test_images directory.")
            else:
                st.warning("Test images directory not found.")

    with col2:
        st.header("🔍 Results")

    if uploaded_file is not None:
        try:
            # Load and display image
            if isinstance(uploaded_file, str):
                image = Image.open(uploaded_file).convert("RGB")
            else:
                image = Image.open(uploaded_file).convert("RGB")

            with col1:
                st.image(image, caption="Uploaded Artwork", use_container_width=True)

            # Load model
            with st.spinner(f"Loading {model_name} model..."):
                if model_name == "VGG16":
                    model = load_vgg16()
                elif model_name == "ResNet50":
                    model = load_resnet50()
                else:  # EfficientNet-B1
                    model = load_efficientnet()

            # Make prediction
            with st.spinner("Analyzing artwork..."):
                results, cam, predicted_class = predict_image(model, image, model_name)

            # Display results
            with col2:
                st.success(
                    f"### 🎯 Predicted Style: **{results[0]['class'].replace('_', ' ')}**"
                )
                st.metric("Confidence", f"{results[0]['confidence']:.2f}%")

                st.markdown("### 📊 Top 5 Predictions")
                for i, result in enumerate(results, 1):
                    col_rank, col_bar = st.columns([1, 1])
                    with col_rank:
                        st.write(f"**{i}. {result['class'].replace('_', ' ')}**")
                    with col_bar:
                        st.progress(result["confidence"] / 100)
                        st.caption(f"{result['confidence']:.2f}%")

            # Grad-CAM visualization
            st.markdown("---")
            st.header("🔥 Grad-CAM Visualization")
            st.write(
                "Grad-CAM highlights the regions in the image that the model focuses on when making predictions."
            )

            if cam is not None:
                # Prepare image for overlay
                img_array = np.array(image.resize((224, 224)))
                heatmap, overlay = apply_gradcam_overlay(img_array, cam)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(
                        img_array, caption="Original Image", use_container_width=True
                    )

                with col2:
                    st.image(
                        heatmap, caption="Grad-CAM Heatmap", use_container_width=True
                    )

                with col3:
                    st.image(overlay, caption="Overlay", use_container_width=True)

                # Download button for Grad-CAM
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img_array)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                axes[1].imshow(heatmap)
                axes[1].set_title("Grad-CAM Heatmap")
                axes[1].axis("off")

                axes[2].imshow(overlay)
                axes[2].set_title("Overlay")
                axes[2].axis("off")

                plt.tight_layout()

                # Save to buffer
                from io import BytesIO

                buf = BytesIO()
                plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)

                st.download_button(
                    label="📥 Download Grad-CAM Visualization",
                    data=buf,
                    file_name=f"gradcam_{model_name}_{results[0]['class']}.png",
                    mime="image/png",
                )

                plt.close()
            else:
                st.warning(
                    "Grad-CAM visualization not available for this model configuration."
                )

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #262730;'>
        <p>Developed by AI Skills Project Team 2025</p>
        <p><strong>Style Classification:</strong> VGG16, ResNet50, EfficientNet-B1</p>
        <p><strong>Artist Classification:</strong> Swin Transformer (Multi-Task Learning page)</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
