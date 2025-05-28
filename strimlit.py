import os
import streamlit as st
from PIL import Image
import numpy as np
import cv2 # OpenCV for heatmap processing
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.amp import autocast

# --- Albumentations Import ---
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------- CONFIGURATION PARAMETERS ----------------
# --- Paths & Model Info ---

# Get the directory where the current script (strimlit.py) is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the model
# This assumes your model is in: your_repo_root/checkpoints/xray_classification/efnetb1_tes.pth
# and your script (strimlit.py) is in: your_repo_root/
MODEL_PATH = os.path.join(script_dir, "checkpoints", "xray_classification", "efnetb1_tes.pth")

# You can uncomment the line below for local debugging to verify the path
# print(f"Attempting to load model from: {MODEL_PATH}")

MODEL_NAME = 'EfficientNet-B1'
IMAGE_SIZE = 240
NUM_CLASSES = 3
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary'] # Must match your trained model's classes

# --- Grad-CAM Parameters ---
GRADCAM_TARGET_LAYER_NAME = 'features.8.0' # Specific target for EfficientNet-B1 head
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- GRAD-CAM CLASS ----------------
class GradCAM:
    def __init__(self, model, target_layer_module):
        self.model = model
        self.target_layer = target_layer_module
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        if self.target_layer is None:
            raise ValueError("Target layer module is None. Cannot register hooks.")
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook) # Use register_full_backward_hook

    def generate_heatmap(self):
        if self.gradients is None or self.activations is None:
            st.error("Error: Gradients or activations not captured in generate_heatmap.")
            return None
        
        # Ensure gradients and activations have compatible dimensions for pooling/mean
        if self.gradients.ndim < 4 or self.activations.ndim < 4:
            st.error(f"Unexpected dimensions for gradients ({self.gradients.ndim}D) or activations ({self.activations.ndim}D). Expected 4D.")
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3]) # Pool spatial dimensions
        
        weighted_activations = self.activations.clone()
        for i in range(pooled_gradients.shape[0]):
            if i < weighted_activations.shape[1]: # Ensure channel index is valid
                 weighted_activations[:, i, :, :] *= pooled_gradients[i]
            else:
                st.warning(f"Gradient channel index {i} out of bounds for activations shape {weighted_activations.shape}")


        heatmap = torch.mean(weighted_activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        if heatmap.sum() == 0:
            st.warning("Generated heatmap is all zeros before normalization. Activations might not highlight relevant areas.")
            if self.activations is not None and self.activations.ndim >= 4:
                return np.zeros(self.activations.shape[2:], dtype=np.float32)
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        else:
            if self.activations is not None and self.activations.ndim >=4 :
                 return np.zeros(self.activations.shape[2:], dtype=np.float32)
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            
        return heatmap.cpu().numpy()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        autocast_enabled = (DEVICE.type == 'cuda')
        
        with autocast(DEVICE.type, enabled=autocast_enabled):
            output = self.model(input_tensor)
            
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        
        try:
            output.backward(gradient=one_hot, retain_graph=True) 
            if self.gradients is None:
                st.warning("Warning: Gradients are None after backward pass. Check hooks and model structure.")
        except RuntimeError as e:
            st.error(f"RuntimeError during backward pass: {e}. Gradients might not be available.")
            self.gradients = None 
            
        return self.generate_heatmap(), class_idx, output.detach()

# ---------------- MODEL LOADING FUNCTION ----------------
@st.cache_resource(show_spinner="Loading AI model...") 
def load_application_model_and_gradcam():
    try:
        st.write(f"Loading model: {MODEL_NAME} from {MODEL_PATH} to {DEVICE}")
        temp_model = None
        if MODEL_NAME == 'EfficientNet-B1':
            temp_model = models.efficientnet_b1(weights=None) 
            num_ftrs = temp_model.classifier[1].in_features
            temp_model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
        elif MODEL_NAME == 'EfficientNet-B0':
            temp_model = models.efficientnet_b0(weights=None)
            num_ftrs = temp_model.classifier[1].in_features
            temp_model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
        else:
            st.error(f"Model {MODEL_NAME} not supported.")
            return None, None

        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found: {MODEL_PATH}")
            return None, None
        
        temp_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        temp_model = temp_model.to(DEVICE)
        temp_model.eval() 
        st.success("Model loaded successfully.")

        target_layer_module = get_target_layer_module_by_name(temp_model, GRADCAM_TARGET_LAYER_NAME)
        if target_layer_module is None:
            st.error(f"Failed to get target layer: {GRADCAM_TARGET_LAYER_NAME}")
            return temp_model, None 

        grad_cam_instance = GradCAM(model=temp_model, target_layer_module=target_layer_module)
        st.success(f"Grad-CAM initialized with target layer: {GRADCAM_TARGET_LAYER_NAME}")
        return temp_model, grad_cam_instance
        
    except Exception as e:
        st.error(f"Failed to load model or init Grad-CAM: {e}")
        print(f"Exception during model load: {e}") 
        return None, None

def get_target_layer_module_by_name(model_obj, target_layer_name_str):
    try:
        layers = target_layer_name_str.split('.')
        module = model_obj
        for layer_name_part in layers:
            if layer_name_part.isdigit(): 
                module = module[int(layer_name_part)]
            else: 
                module = getattr(module, layer_name_part)
        return module
    except Exception as e:
        st.error(f"Could not find layer '{target_layer_name_str}'. Error: {e}")
        return None


# ---------------- IMAGE PREPROCESSING ----------------
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
preprocess_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

def preprocess_image_pil(image_pil):
    try:
        original_image_np = np.array(image_pil.convert('RGB'))
    except Exception as e:
        st.error(f"Error converting uploaded image to NumPy array: {e}")
        return None, None
    
    try:
        augmented = preprocess_transform(image=original_image_np.copy()) 
        input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        return original_image_np, input_tensor
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return original_image_np, None

# ------------- CORE PROCESSING AND VISUALIZATION LOGIC (Streamlit specific) ---------------
def generate_and_display_visuals(original_np, input_tensor, model, grad_cam_instance):
    if model is None:
        st.error("Model not available for processing.")
        return "Error", "N/A", None, None, None
    if grad_cam_instance is None:
        st.warning("Grad-CAM not available. Showing prediction only.")
        with torch.no_grad(), autocast(DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            output_logits = model(input_tensor)
        predicted_class_idx = torch.argmax(output_logits, dim=1).item()
        predicted_class_name = CLASS_NAMES[predicted_class_idx]
        probabilities = F.softmax(output_logits, dim=1)
        confidence = probabilities[0, predicted_class_idx].item() * 100
        confidence_str = f"{confidence:.2f}%"
        return predicted_class_name, confidence_str, original_np, None, None 

    heatmap_np, predicted_class_idx, output_logits = grad_cam_instance(input_tensor) 
    
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    probabilities = F.softmax(output_logits, dim=1)
    confidence = probabilities[0, predicted_class_idx].item() * 100
    confidence_str = f"{confidence:.2f}%"

    if heatmap_np is None:
        st.error("Could not generate heatmap.")
        return predicted_class_name, confidence_str, original_np, None, None 
    
    if not isinstance(heatmap_np, np.ndarray) or heatmap_np.ndim != 2 or heatmap_np.size == 0:
        st.error("Generated heatmap is invalid (not 2D or empty).")
        return predicted_class_name, confidence_str, original_np, None, None
    if np.isnan(heatmap_np).any() or np.isinf(heatmap_np).any():
        st.warning("Heatmap contains NaN or Inf. Clamping values.")
        heatmap_np = np.nan_to_num(heatmap_np, nan=0.0, posinf=1.0, neginf=0.0)
    if heatmap_np.dtype != np.float32: 
        heatmap_np = heatmap_np.astype(np.float32)

    overlayed_image_np = None
    heatmap_resized_display = None
    try:
        heatmap_resized = cv2.resize(heatmap_np, (original_np.shape[1], original_np.shape[0]))
        heatmap_resized_display = heatmap_resized 
        
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) 
        
        alpha = 0.5
        overlayed_image_np = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)
    except cv2.error as e:
        st.error(f"OpenCV Error during heatmap processing: {e}")
    except Exception as e:
        st.error(f"Unexpected error during heatmap processing: {e}")

    return predicted_class_name, confidence_str, original_np, heatmap_resized_display, overlayed_image_np


# ------------------- STREAMLIT UI SETUP -------------------
st.set_page_config(layout="wide", page_title="Grad-CAM Visualizer")
st.title("🧠 Grad-CAM Visualizer for Brain Tumor Classification")
st.caption(f"Model: {MODEL_NAME} | Target Layer for Grad-CAM: {GRADCAM_TARGET_LAYER_NAME} | Device: {DEVICE}")

loaded_model, grad_cam_instance = load_application_model_and_gradcam()

if loaded_model:
    st.sidebar.success(f"Model '{MODEL_NAME}' loaded successfully on {DEVICE}.")
    if grad_cam_instance:
        st.sidebar.success("Grad-CAM initialized.")
    else:
        st.sidebar.warning("Grad-CAM failed to initialize. Only predictions will be available.")
else:
    st.sidebar.error("Model loading failed. Please check the logs and ensure MODEL_PATH is correct relative to the script in your repository.")
    st.stop() 

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    st.write(f"Uploaded: {uploaded_file.name}")

    with st.spinner("Preprocessing image..."):
        original_np, input_tensor = preprocess_image_pil(image_pil)

    if original_np is not None and input_tensor is not None:
        st.subheader("Processing Results")
        
        with st.spinner("Generating Grad-CAM and prediction..."):
            pred_name, conf_str, displayed_original, heatmap_display, overlay_display = \
                generate_and_display_visuals(original_np, input_tensor, loaded_model, grad_cam_instance)

        col_pred, col_conf = st.columns(2)
        with col_pred:
            st.metric("Predicted Class", pred_name)
        with col_conf:
            st.metric("Confidence", conf_str)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(displayed_original, caption="Original Image", use_column_width=True)
        
        if heatmap_display is not None:
            with col2:
                st.image(heatmap_display, caption="Grad-CAM Heatmap", use_column_width=True, clamp=True, channels="GRAY")
        else:
            with col2:
                st.info("Heatmap not available.")

        if overlay_display is not None:
            with col3:
                st.image(overlay_display, caption="Overlayed Image (Original + Heatmap)", use_column_width=True)
        elif grad_cam_instance is not None: 
             with col3:
                st.info("Overlay not available due to processing error.")
        
    else:
        st.error("Failed to preprocess the image. Please try another file.")
else:
    st.info("Please upload an image file to begin analysis.")

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This application uses a pre-trained EfficientNet model to classify brain tumor images "
    "(glioma, meningioma, pituitary) and visualizes the model's attention using Grad-CAM."
)
st.sidebar.markdown(f"**Class Names:** {', '.join(CLASS_NAMES)}")