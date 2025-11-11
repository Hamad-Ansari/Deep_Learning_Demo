import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.model_loader import load_model, predict

# Page configuration
st.set_page_config(
    page_title="Deep Learning App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Deep Learning Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Image Classification", "Object Detection", "About"]
    )
    
    if app_mode == "Image Classification":
        image_classification()
    elif app_mode == "Object Detection":
        object_detection()
    else:
        about()

def image_classification():
    st.header("Image Classification")
    
    # Model selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Settings")
        model_type = st.selectbox(
            "Select Model",
            ["ResNet50", "VGG16", "Custom Model"]
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
    with col2:
        # Image upload
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image for classification"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess and predict
            if st.button("Classify Image"):
                with st.spinner("Processing..."):
                    # Load model
                    model, preprocess = load_model(model_type)
                    
                    # Preprocess image
                    processed_image = preprocess(image).unsqueeze(0)
                    
                    # Make prediction
                    predictions, top_classes, top_probs = predict(model, processed_image)
                    
                    # Display results
                    display_results(top_classes, top_probs, confidence_threshold)

def display_results(classes, probabilities, threshold):
    st.subheader("Prediction Results")
    
    # Create columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Predictions")
        for class_name, prob in zip(classes, probabilities):
            if prob >= threshold:
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>{class_name}</h4>
                    <p>Confidence: {prob:.2%}</p>
                    <progress value="{prob}" max="1"></progress>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Create bar chart
        fig = plot_prediction_barchart(classes, probabilities)
        st.pyplot(fig)

def plot_prediction_barchart(classes, probabilities):
    fig, ax = plt.subplots()
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, probabilities)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Probability')
    ax.set_title('Top Predictions')
    ax.invert_yaxis()
    return fig

def object_detection():
    st.header("Object Detection")
    st.info("Object detection feature coming soon!")
    
    # Placeholder for object detection functionality
    uploaded_file = st.file_uploader(
        "Upload image for object detection",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Objects"):
            st.warning("Object detection model not implemented yet")

def about():
    st.header("About This App")
    st.markdown("""
    This is a deep learning demo app built with Streamlit.
    
    **Features:**
    - Image classification using pre-trained models
    - Real-time predictions
    - Interactive visualization of results
    
    **Technologies used:**
    - Streamlit for the web interface
    - PyTorch for deep learning
    - OpenCV for image processing
    
    **Supported models:**
    - ResNet50
    - VGG16
    - Custom models
    """)
    # Add this to your main app file for additional functionality

def model_training_demo():
    st.header("Model Training Demo")
    
    st.info("""
    This section demonstrates a simple neural network training process.
    Note: Actual training happens locally and may take time.
    """)
    
    # Training parameters
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of epochs", 1, 10, 3)
        learning_rate = st.select_slider(
            "Learning rate",
            options=[0.001, 0.01, 0.1, 0.5],
            value=0.01
        )
    
    with col2:
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128])
        optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
    
    if st.button("Start Training Demo"):
        training_progress(epochs)

def training_progress(epochs):
    """Simulate training progress"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        # Simulate training
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch + 1}/{epochs} completed")
        
        # Simulate metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Loss", f"{0.5 - epoch * 0.1:.3f}")
        with col2:
            st.metric("Validation Loss", f"{0.6 - epoch * 0.08:.3f}")
        with col3:
            st.metric("Accuracy", f"{(epoch + 1) * 15}%")
        
        st.write("---")

if __name__ == "__main__":
    main()