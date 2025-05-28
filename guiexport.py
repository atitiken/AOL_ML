import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2 # OpenCV for heatmap processing
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.amp import autocast

# --- Albumentations Import ---
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------- CONFIGURATION PARAMETERS ----------------
# --- Paths & Model Info (USER: UPDATE THESE) ---
# Ensure this path is correct or the .exe won't find the model unless bundled.
MODEL_FILENAME = "efnetb1_tes.pth" # Just the filename
base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
MODEL_PATH = os.path.join(base_path, 'efnetb1_tes.pth')
MODEL_NAME = 'EfficientNet-B1'
IMAGE_SIZE = 240
NUM_CLASSES = 3
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary'] # Must match your trained model's classes

# --- Grad-CAM Parameters ---
GRADCAM_TARGET_LAYER_NAME = 'features.8.0' # Specific target for EfficientNet-B1 head
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variable to hold the loaded model and GradCAM instance
loaded_model = None
grad_cam_instance = None
is_model_loaded = False

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
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self):
        if self.gradients is None or self.activations is None:
            print("Error: Gradients or activations not captured in generate_heatmap.")
            return None
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        weighted_activations = self.activations.clone()
        for i in range(pooled_gradients.shape[0]):
            weighted_activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(weighted_activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        if heatmap.sum() == 0:
            if self.activations is not None and self.activations.ndim >= 4:
                return np.zeros(self.activations.shape[2:], dtype=np.float32)
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        else: # Should be caught by sum == 0
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
                print("Warning: Gradients are None after backward pass. CUDA context issue likely.")
        except RuntimeError as e:
            print(f"RuntimeError during backward pass: {e}. Gradients might not be available.")
            self.gradients = None
        return self.generate_heatmap(), class_idx, output.detach()

# ---------------- MODEL LOADING FUNCTION ----------------
def load_application_model():
    global loaded_model, grad_cam_instance, is_model_loaded
    if is_model_loaded:
        return True
    try:
        print(f"Loading model: {MODEL_NAME} from {MODEL_PATH} to {DEVICE}")
        temp_model = None
        if MODEL_NAME == 'EfficientNet-B1':
            temp_model = models.efficientnet_b1(weights=None)
            num_ftrs = temp_model.classifier[1].in_features
            temp_model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
        elif MODEL_NAME == 'EfficientNet-B0': # Add other models if needed
            temp_model = models.efficientnet_b0(weights=None)
            num_ftrs = temp_model.classifier[1].in_features
            temp_model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
        else:
            messagebox.showerror("Error", f"Model {MODEL_NAME} not supported.")
            return False

        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", f"Model file not found: {MODEL_PATH}")
            return False
        
        temp_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        temp_model = temp_model.to(DEVICE)
        temp_model.eval()
        loaded_model = temp_model
        print("Model loaded successfully.")

        # Initialize Grad-CAM
        target_layer_module = get_target_layer_module_by_name(loaded_model, GRADCAM_TARGET_LAYER_NAME)
        grad_cam_instance = GradCAM(model=loaded_model, target_layer_module=target_layer_module)
        print(f"Grad-CAM initialized with target layer: {GRADCAM_TARGET_LAYER_NAME}")
        is_model_loaded = True
        return True
    except Exception as e:
        messagebox.showerror("Model Loading Error", f"Failed to load model or init Grad-CAM: {e}")
        print(f"Exception during model load: {e}")
        is_model_loaded = False
        return False

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
        raise ValueError(f"Could not find layer '{target_layer_name_str}'. Error: {e}")


# ---------------- IMAGE PREPROCESSING ----------------
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
preprocess_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

def preprocess_image_from_path(image_path):
    try:
        image_pil = Image.open(image_path).convert('RGB')
        original_image_np = np.array(image_pil) # For display and Grad-CAM overlay
    except Exception as e:
        messagebox.showerror("Image Error", f"Error loading image {os.path.basename(image_path)}: {e}")
        return None, None
    
    # Transform for model input
    try:
        augmented = preprocess_transform(image=original_image_np.copy())
        input_tensor = augmented['image'].unsqueeze(0).to(DEVICE) # Add batch dim
        return original_image_np, input_tensor
    except Exception as e:
        messagebox.showerror("Preprocessing Error", f"Error preprocessing image: {e}")
        return original_image_np, None # Return original for context if tensor fails

# ------------- CORE PROCESSING AND VISUALIZATION LOGIC ---------------
def process_and_visualize(image_path_str, fig, canvas_widget):
    global loaded_model, grad_cam_instance

    if not is_model_loaded:
        messagebox.showerror("Error", "Model not loaded. Please restart the application.")
        return None, "Error", "N/A"

    original_np, input_tensor = preprocess_image_from_path(image_path_str)

    if original_np is None or input_tensor is None:
        return None, "Error processing image", "N/A"

    # Get prediction and Grad-CAM
    heatmap_np, predicted_class_idx, output_logits = grad_cam_instance(input_tensor) # No specific class_idx, uses max
    
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    probabilities = F.softmax(output_logits, dim=1)
    confidence = probabilities[0, predicted_class_idx].item() * 100
    confidence_str = f"{confidence:.2f}%"

    if heatmap_np is None:
        messagebox.showerror("Grad-CAM Error", "Could not generate heatmap.")
        return None, predicted_class_name, confidence_str
    
    # Robust checks for heatmap before resize
    if not isinstance(heatmap_np, np.ndarray) or heatmap_np.ndim != 2 or heatmap_np.size == 0:
        messagebox.showerror("Grad-CAM Error", "Generated heatmap is invalid (not 2D or empty).")
        return None, predicted_class_name, confidence_str
    if np.isnan(heatmap_np).any() or np.isinf(heatmap_np).any():
        print("Warning: heatmap_np contains NaN or Inf. Clamping values.")
        heatmap_np = np.nan_to_num(heatmap_np, nan=0.0, posinf=1.0, neginf=0.0)
    if heatmap_np.dtype != np.float32:
        heatmap_np = heatmap_np.astype(np.float32)

    # Resize heatmap, color, and overlay
    try:
        heatmap_resized = cv2.resize(heatmap_np, (original_np.shape[1], original_np.shape[0]))
    except cv2.error as e:
        messagebox.showerror("OpenCV Error", f"Error resizing heatmap: {e}")
        return None, predicted_class_name, confidence_str
        
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # OpenCV is BGR
    alpha = 0.5
    overlayed_image_np = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)

    # Update Matplotlib figure
    fig.clear() # Clear previous plot
    axs = fig.subplots(1, 3) # Recreate subplots
    
    base_filename = os.path.basename(image_path_str)
    # True label isn't known from single upload, so we omit it or set to "Uploaded"
    fig.suptitle(f"Grad-CAM: {base_filename}\nPredicted: {predicted_class_name} ({confidence_str})", fontsize=10)

    axs[0].imshow(original_np)
    axs[0].set_title("Original", fontsize=8)
    axs[0].axis('off')

    axs[1].imshow(heatmap_resized, cmap='jet')
    axs[1].set_title("Heatmap", fontsize=8)
    axs[1].axis('off')

    axs[2].imshow(overlayed_image_np)
    axs[2].set_title("Overlay", fontsize=8)
    axs[2].axis('off')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
    canvas_widget.draw()
    
    return predicted_class_name, confidence_str

# ------------------- TKINTER GUI SETUP -------------------
class App:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Grad-CAM Visualizer")
        self.root.geometry("800x700")

        # Style
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabel", padding=6, font=('Helvetica', 10))
        style.configure("Header.TLabel", padding=6, font=('Helvetica', 14, 'bold'))


        # Frame for controls
        control_frame = ttk.Frame(root_window, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.upload_button = ttk.Button(control_frame, text="Upload Image", command=self.upload_and_process_image)
        self.upload_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.status_label_var = tk.StringVar(value="Status: Waiting for model to load...")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_label_var)
        self.status_label.pack(side=tk.LEFT, padx=5, pady=5)


        # Frame for results
        results_frame = ttk.Frame(root_window, padding="10")
        results_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Matplotlib Figure and Canvas
        self.fig = plt.Figure(figsize=(7.5, 2.5), dpi=100) # Adjusted for 1x3 plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)
        # Initial empty plot
        self.ax_init = self.fig.add_subplot(111)
        self.ax_init.text(0.5, 0.5, "Upload an image to see Grad-CAM visualization", 
                          horizontalalignment='center', verticalalignment='center',
                          fontsize=10, color='grey')
        self.ax_init.axis('off')
        self.canvas.draw()


        # Prediction and Confidence Labels
        info_frame = ttk.Frame(results_frame)
        info_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.pred_label_var = tk.StringVar(value="Predicted Class: N/A")
        pred_title_label = ttk.Label(info_frame, text="Prediction:", font=('Helvetica', 10, 'bold'))
        pred_title_label.pack(side=tk.LEFT, padx=(0,5))
        pred_val_label = ttk.Label(info_frame, textvariable=self.pred_label_var)
        pred_val_label.pack(side=tk.LEFT)

        self.conf_label_var = tk.StringVar(value="Confidence: N/A")
        conf_title_label = ttk.Label(info_frame, text="Confidence:", font=('Helvetica', 10, 'bold'))
        conf_title_label.pack(side=tk.LEFT, padx=(20,5))
        conf_val_label = ttk.Label(info_frame, textvariable=self.conf_label_var)
        conf_val_label.pack(side=tk.LEFT)
        
        # Load model after GUI is set up (non-blocking would be better for UX but complex)
        self.root.after(100, self.initialize_model) # Use 'after' for a slight delay

    def initialize_model(self):
        self.status_label_var.set("Status: Loading model, please wait...")
        self.root.update_idletasks() # Update GUI to show message
        if load_application_model():
            self.status_label_var.set(f"Status: Model loaded ({DEVICE}). Ready to upload.")
            self.upload_button.config(state=tk.NORMAL)
        else:
            self.status_label_var.set("Status: Model loading failed. Check console.")
            self.upload_button.config(state=tk.DISABLED)


    def upload_and_process_image(self):
        if not is_model_loaded:
            messagebox.showwarning("Model Not Ready", "The model is still loading or failed to load.")
            return

        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"), ("All files", "*.*"))
        )
        if not file_path:
            return # User cancelled

        self.status_label_var.set(f"Status: Processing {os.path.basename(file_path)}...")
        self.pred_label_var.set("Predicted Class: Processing...")
        self.conf_label_var.set("Confidence: Processing...")
        self.root.update_idletasks()


        pred_name, conf_str = process_and_visualize(file_path, self.fig, self.canvas)
        
        if pred_name: # If processing was successful (even if heatmap failed)
            self.pred_label_var.set(f"{pred_name}")
            self.conf_label_var.set(f"{conf_str}")
            self.status_label_var.set(f"Status: Done. {os.path.basename(file_path)}")
        else: # Error occurred during processing
            self.pred_label_var.set("Predicted Class: Error")
            self.conf_label_var.set("Confidence: Error")
            self.status_label_var.set(f"Status: Error processing. Check console/messages.")
            # Clear the plot or show an error image
            self.fig.clear()
            ax_err = self.fig.add_subplot(111)
            ax_err.text(0.5, 0.5, "Error processing image.", color='red', ha='center', va='center')
            ax_err.axis('off')
            self.canvas.draw()


if __name__ == '__main__':
    if DEVICE.type == 'cuda':
        try:
            torch.cuda.init() # Explicitly initialize CUDA if available
            print("CUDA initialized for application.")
        except Exception as e:
            print(f"Warning: Could not explicitly initialize CUDA: {e}")

    root = tk.Tk()
    app = App(root)
    root.mainloop()