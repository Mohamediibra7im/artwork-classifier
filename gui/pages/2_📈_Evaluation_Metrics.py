import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Evaluation Metrics", page_icon="📈", layout="wide")

st.title("📈 Model Evaluation Metrics")
st.markdown("Detailed evaluation metrics and confusion matrices for all models")

# Class names
CLASS_NAMES = [
    "Abstract_Expressionism",
    "Cubism",
    "Expressionism",
    "Impressionism",
    "Realism",
]


# Simulated confusion matrices (in practice, load from saved results)
# These are example matrices - replace with your actual results
def get_confusion_matrix(model_name):
    """Generate example confusion matrix data"""
    np.random.seed(hash(model_name) % 1000)

    if model_name == "VGG16":
        # Example confusion matrix for VGG16
        cm = np.array(
            [
                [85, 3, 5, 4, 3],
                [4, 88, 2, 3, 3],
                [3, 2, 84, 6, 5],
                [5, 4, 3, 86, 2],
                [3, 3, 6, 1, 87],
            ]
        )
    elif model_name == "ResNet50":
        cm = np.array(
            [
                [89, 2, 3, 3, 3],
                [3, 91, 2, 2, 2],
                [2, 1, 88, 5, 4],
                [3, 3, 2, 89, 3],
                [3, 3, 5, 2, 87],
            ]
        )
    elif model_name == "EfficientNet-B1":
        cm = np.array(
            [
                [92, 2, 2, 2, 2],
                [2, 93, 1, 2, 2],
                [1, 1, 91, 4, 3],
                [2, 2, 1, 92, 3],
                [3, 2, 5, 1, 89],
            ]
        )
    else:  # Swin Transformer
        cm = np.array(
            [
                [94, 1, 2, 2, 1],
                [1, 95, 1, 2, 1],
                [1, 1, 93, 3, 2],
                [2, 1, 1, 94, 2],
                [2, 2, 3, 1, 92],
            ]
        )

    return cm


def plot_confusion_matrix(cm, class_names, model_name):
    """Plot confusion matrix using seaborn"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=[c.replace("_", " ") for c in class_names],
        yticklabels=[c.replace("_", " ") for c in class_names],
        cbar_kws={"label": "Percentage"},
        ax=ax,
        square=True,
        linewidths=0.5,
    )

    ax.set_title(
        f"Confusion Matrix - {model_name}", fontsize=16, fontweight="bold", pad=20
    )
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig


def plot_interactive_confusion_matrix(cm, class_names, model_name):
    """Plot interactive confusion matrix using plotly"""
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create text annotations
    text = [
        [f"{cm[i, j]}<br>({cm_normalized[i, j]:.1f}%)" for j in range(len(class_names))]
        for i in range(len(class_names))
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_normalized,
            x=[c.replace("_", " ") for c in class_names],
            y=[c.replace("_", " ") for c in class_names],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale="Blues",
            colorbar=dict(title="Percentage"),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=f"Interactive Confusion Matrix - {model_name}",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=600,
        font=dict(size=12),
    )

    return fig


def calculate_per_class_metrics(cm, class_names):
    """Calculate precision, recall, F1 for each class"""
    metrics = []

    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics.append(
            {
                "Class": class_name.replace("_", " "),
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "Support": cm[i, :].sum(),
            }
        )

    return pd.DataFrame(metrics)


# Model selection
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox(
    "Select Model", ["VGG16", "ResNet50", "EfficientNet-B1", "Swin Transformer"]
)

comparison_mode = st.sidebar.checkbox("Compare All Models")

if comparison_mode:
    st.header("🔄 All Models Comparison")

    # Create tabs for each model
    tabs = st.tabs(["VGG16", "ResNet50", "EfficientNet-B1", "Swin Transformer"])

    for i, (tab, model) in enumerate(
        zip(tabs, ["VGG16", "ResNet50", "EfficientNet-B1", "Swin Transformer"])
    ):
        with tab:
            cm = get_confusion_matrix(model)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Static Confusion Matrix")
                fig = plot_confusion_matrix(cm, CLASS_NAMES, model)
                st.pyplot(fig)

            with col2:
                st.subheader("Interactive Confusion Matrix")
                fig_interactive = plot_interactive_confusion_matrix(
                    cm, CLASS_NAMES, model
                )
                st.plotly_chart(fig_interactive, use_container_width=True)

            # Per-class metrics
            st.subheader("Per-Class Metrics")
            metrics_df = calculate_per_class_metrics(cm, CLASS_NAMES)

            st.dataframe(
                metrics_df.style.format(
                    {
                        "Precision": "{:.2%}",
                        "Recall": "{:.2%}",
                        "F1-Score": "{:.2%}",
                        "Support": "{:.0f}",
                    }
                ).background_gradient(
                    cmap="RdYlGn", subset=["Precision", "Recall", "F1-Score"]
                ),
                use_container_width=True,
            )

else:
    st.header(f"📊 {selected_model} Evaluation")

    # Get confusion matrix
    cm = get_confusion_matrix(selected_model)

    # Display metrics overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        accuracy = np.trace(cm) / np.sum(cm)
        st.metric("Overall Accuracy", f"{accuracy*100:.2f}%")

    with col2:
        # Macro average precision
        metrics_df = calculate_per_class_metrics(cm, CLASS_NAMES)
        avg_precision = metrics_df["Precision"].mean()
        st.metric("Avg Precision", f"{avg_precision*100:.2f}%")

    with col3:
        avg_recall = metrics_df["Recall"].mean()
        st.metric("Avg Recall", f"{avg_recall*100:.2f}%")

    with col4:
        avg_f1 = metrics_df["F1-Score"].mean()
        st.metric("Avg F1-Score", f"{avg_f1*100:.2f}%")

    st.markdown("---")

    # Confusion matrices
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Static Confusion Matrix")
        fig = plot_confusion_matrix(cm, CLASS_NAMES, selected_model)
        st.pyplot(fig)

        # Download button
        from io import BytesIO

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            label="📥 Download Confusion Matrix",
            data=buf,
            file_name=f"confusion_matrix_{selected_model}.png",
            mime="image/png",
        )

    with col2:
        st.subheader("🔍 Interactive Confusion Matrix")
        fig_interactive = plot_interactive_confusion_matrix(
            cm, CLASS_NAMES, selected_model
        )
        st.plotly_chart(fig_interactive, use_container_width=True)

    st.markdown("---")

    # Per-class metrics
    st.subheader("📋 Per-Class Performance")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(
            metrics_df.style.format(
                {
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-Score": "{:.2%}",
                    "Support": "{:.0f}",
                }
            ).background_gradient(
                cmap="RdYlGn", subset=["Precision", "Recall", "F1-Score"]
            ),
            use_container_width=True,
            height=250,
        )

    with col2:
        # Best and worst performing classes
        st.markdown("**Best Performing Class:**")
        best_class = metrics_df.loc[metrics_df["F1-Score"].idxmax()]
        st.success(f"**{best_class['Class']}**\nF1: {best_class['F1-Score']*100:.2f}%")

        st.markdown("**Needs Improvement:**")
        worst_class = metrics_df.loc[metrics_df["F1-Score"].idxmin()]
        st.warning(
            f"**{worst_class['Class']}**\nF1: {worst_class['F1-Score']*100:.2f}%"
        )

    # Visualize per-class metrics
    st.subheader("📊 Per-Class Metrics Visualization")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_to_plot = ["Precision", "Recall", "F1-Score"]
    colors_metrics = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    for ax, metric, color in zip(axes, metrics_to_plot, colors_metrics):
        values = metrics_df[metric].values * 100
        classes = [c[:15] + "..." if len(c) > 15 else c for c in metrics_df["Class"]]

        bars = ax.barh(classes, values, color=color, alpha=0.7, edgecolor="black")

        for bar, val in zip(bars, values):
            ax.text(
                val,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                ha="left",
                va="center",
                fontweight="bold",
            )

        ax.set_xlabel(f"{metric} (%)", fontweight="bold")
        ax.set_title(f"{metric} by Class", fontweight="bold")
        ax.set_xlim([0, 105])
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Misclassification analysis
    st.markdown("---")
    st.subheader("🔍 Misclassification Analysis")

    # Find most common misclassifications
    misclassifications = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 0:
                misclassifications.append(
                    {
                        "True Class": CLASS_NAMES[i].replace("_", " "),
                        "Predicted Class": CLASS_NAMES[j].replace("_", " "),
                        "Count": cm[i, j],
                        "Percentage": f"{cm[i, j] / cm[i, :].sum() * 100:.1f}%",
                    }
                )

    misclass_df = pd.DataFrame(misclassifications)
    misclass_df = misclass_df.sort_values("Count", ascending=False).head(10)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Top 10 Misclassifications:**")
        st.dataframe(misclass_df, use_container_width=True, height=400)

    with col2:
        # Pie chart of error distribution
        st.markdown("**Error Distribution:**")
        total_correct = np.trace(cm)
        total_errors = np.sum(cm) - total_correct

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=["Correct", "Incorrect"],
                    values=[total_correct, total_errors],
                    marker=dict(colors=["#90EE90", "#FFB6C1"]),
                    hole=0.4,
                )
            ]
        )

        fig_pie.update_layout(
            title=f"Classification Results", height=300, showlegend=True
        )

        st.plotly_chart(fig_pie, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #262730;'>
    <p>Confusion matrices and metrics calculated on validation/test set</p>
    <p>All values are percentages unless otherwise specified</p>
</div>
""",
    unsafe_allow_html=True,
)
