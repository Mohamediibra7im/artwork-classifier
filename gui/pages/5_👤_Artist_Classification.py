import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import timm
import sys
from pathlib import Path
from io import BytesIO

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Artist Classification",
    page_icon="👤",
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

# Artist class names (based on notebook - sorted alphabetically)
ARTIST_NAMES = [
    "abraham-manievich",
    "adam-baltatu",
    "adolf-hitler",
    "adolphe-joseph-thomas-monticelli",
    "adriaen-brouwer",
    "adriaen-van-de-velde",
    "adriaen-van-de-venne",
    "adriaen-van-ostade",
    "agnes-martin",
    "a.y.-jackson",
]

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_artist_name(name):
    """Format artist name for display"""
    return name.replace("-", " ").title()


# Image preprocessing
def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# Load Swin Transformer artist model
@st.cache_resource
def load_swin_artist_model():
    """Load Swin Transformer model for artist classification"""
    # Try different Swin variants
    for model_size in ["tiny", "small", "base"]:
        try:
            # Load with 10 classes
            model = timm.create_model(
                f"swin_{model_size}_patch4_window7_224",
                pretrained=False,
                num_classes=len(ARTIST_NAMES),
            )
            
            # Load state dict
            state_dict = torch.load(
                "models/swin_artist_model.pth", map_location=DEVICE
            )
            
            # Handle different state dict formats
            if isinstance(state_dict, dict):
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif "artist2idx" in state_dict:
                    # Remove artist2idx if present
                    state_dict = {k: v for k, v in state_dict.items() if k != "artist2idx"}
            
            # Load state dict (strict=False to handle head differences)
            model.load_state_dict(state_dict, strict=False)
            model.to(DEVICE)
            model.eval()
            return model, model_size
        except Exception as e:
            continue
    
    st.error("Could not load Swin Transformer artist model")
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


def get_target_layer(model):
    """Get the target layer for Grad-CAM for Swin Transformer"""
    try:
        # Try to get the last layer's last block's norm1
        if hasattr(model, "layers") and len(model.layers) > 0:
            last_layer = model.layers[-1]
            if hasattr(last_layer, "blocks") and len(last_layer.blocks) > 0:
                return last_layer.blocks[-1].norm1
        # Fallback to last layer
        if hasattr(model, "layers") and len(model.layers) > 0:
            return model.layers[-1]
        return None
    except:
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


def predict_artist(model, image):
    """Make prediction and generate Grad-CAM"""
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidences, indices = torch.topk(probabilities, k=min(5, len(ARTIST_NAMES)))

    # Generate Grad-CAM
    target_layer = get_target_layer(model)
    cam = None
    predicted_class = indices[0, 0].item()
    
    if target_layer is not None:
        try:
            gradcam = GradCAM(model, target_layer)
            cam, predicted_class = gradcam.generate_cam(input_tensor)
        except Exception as e:
            st.warning(f"Grad-CAM visualization unavailable: {str(e)}")

    # Prepare results
    results = []
    for conf, idx in zip(confidences[0], indices[0]):
        results.append(
            {
                "artist": ARTIST_NAMES[idx.item()],
                "confidence": conf.item() * 100,
            }
        )

    return results, cam, predicted_class


# Main app
def main():
    st.markdown(
        '<div class="main-header">👤 Artist Classification System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">AI-Powered Artist Recognition using Swin Transformer</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        st.info(
            "💡 This model uses **Swin Transformer** architecture trained specifically for artist classification."
        )

        st.markdown("---")

        # Model information
        st.header("📊 Model Information")
        st.info(
            """
            **Swin Transformer (Tiny)**
            - Architecture: Vision Transformer with shifted windows
            - Classes: 10 artists
            - Trained on: WikiArt dataset
            - Best for: Artist identification
            """
        )

        st.markdown("---")

        # Artist information
        st.header("👥 Supported Artists")
        for i, artist_name in enumerate(ARTIST_NAMES, 1):
            st.write(f"{i}. {format_artist_name(artist_name)}")

        st.markdown("---")

        # About
        with st.expander("ℹ️ About"):
            st.write(
                """
            This application identifies the artist of an artwork using a 
            Swin Transformer model trained on the WikiArt dataset.
            
            **Features:**
            - Swin Transformer architecture
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
            with st.spinner("Loading Swin Transformer artist model..."):
                model, model_size = load_swin_artist_model()
                if model is None:
                    st.error("Failed to load the artist classification model.")
                    st.stop()

            # Make prediction
            with st.spinner("Analyzing artwork..."):
                results, cam, predicted_class = predict_artist(model, image)

            # Display results
            with col2:
                st.success(
                    f"### 🎯 Predicted Artist: **{format_artist_name(results[0]['artist'])}**"
                )
                st.metric("Confidence", f"{results[0]['confidence']:.2f}%")

                st.markdown("### 📊 Top 5 Predictions")
                for i, result in enumerate(results, 1):
                    col_rank, col_bar = st.columns([1, 1])
                    with col_rank:
                        st.write(f"**{i}. {format_artist_name(result['artist'])}**")
                    with col_bar:
                        st.progress(result["confidence"] / 100)
                        st.caption(f"{result['confidence']:.2f}%")

            # Grad-CAM visualization
            st.markdown("---")
            st.header("🔥 Grad-CAM Visualization")
            st.write(
                "Grad-CAM highlights the regions in the image that the model focuses on when identifying the artist."
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
                buf = BytesIO()
                plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)

                st.download_button(
                    label="📥 Download Grad-CAM Visualization",
                    data=buf,
                    file_name=f"gradcam_artist_{results[0]['artist']}.png",
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
        <p><strong>Artist Classification:</strong> Swin Transformer</p>
        <p><strong>Model:</strong> swin_artist_model.pth</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

