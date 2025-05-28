import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models 
import torchvision.transforms.functional as TF 
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy
import datetime
import time
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn.functional as F


# ---------------- CONFIGURATION PARAMETERS ----------------
# --- Paths ---
ALL_DATA_DIR = r"D:\Punya dede\AOL ML\Tumor\3_class"
TRAIN_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# --- Model & Training ---
MODEL_NAME = 'EfficientNet-B1' 
IMAGE_SIZE = 240              
NUM_CLASSES = 3
NUM_EPOCHS = 50                
BATCH_SIZE = 16                
LEARNING_RATE = 1e-4
LEARNING_RATE_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE_EARLY_STOPPING = 10   
LABEL_SMOOTHING = 0.0         

# --- Fine-tuning Strategy ---
FREEZE_BACKBONE_INITIALLY = True 
EPOCHS_TO_TRAIN_CLASSIFIER_ONLY = 5

# --- Data Handling ---
USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTS_IN_LOSS = True 

# --- Reproducibility & Device ---
MANUAL_SEED = 42
torch.manual_seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(MANUAL_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False

# Global variables for class information
CLASS_TO_IDX = None
CLASS_NAMES = None

# ---------------- FOCAL LOSS CLASS ----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
       
        # Calculate Cross Entropy loss without reduction
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities of the true class: pt = exp(-CE_loss)
        pt = torch.exp(-CE_loss)
        
        # Calculate Focal Loss: F_loss = (1-pt)^gamma * CE_loss
        F_loss = (1 - pt)**self.gamma * CE_loss

        if self.alpha is not None:
            # Ensure alpha is on the same device and of the same type as inputs
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            
            # Select alpha for each sample based on its target class
            at = self.alpha.gather(0, targets.data.view(-1))
            F_loss = at * F_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else: 
            return F_loss

# ---------------- CUSTOM IMAGE DATASET CLASS (MODIFIED FOR ALBUMENTATIONS) ----------------
class CustomImageDataset(Dataset):
    def __init__(self, image_paths_list, labels_list,
                 master_class_to_idx, master_classes_list,
                 transform=None, dataset_name="Dataset"): 
        self.image_paths = image_paths_list
        self.labels = labels_list
        self.transform = transform
        self.class_to_idx = master_class_to_idx
        self.classes = master_classes_list

        if not self.image_paths:
            raise ValueError(f"{dataset_name}: No image paths provided.")
        if len(self.image_paths) != len(self.labels):
            raise ValueError(f"{dataset_name}: image_paths and labels_list must have the same length.")

        self.class_counts = [0] * len(self.classes)
        for label_idx in self.labels:
            if 0 <= label_idx < len(self.classes):
                self.class_counts[label_idx] += 1
            else:
                print(f"Warning: Invalid label index {label_idx} encountered in {dataset_name}.")

        print(f"Initialized {dataset_name} with {len(self.image_paths)} images.")
        print(f"  Class distribution for {dataset_name}:")
        for i, class_name_val in enumerate(self.classes):
            print(f"    Class '{class_name_val}' ({self.class_to_idx[class_name_val]}): {self.class_counts[i]} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            # Load image as PIL and convert to NumPy array for Albumentations
            image_pil = Image.open(img_path).convert('RGB')
            image_np = np.array(image_pil)
        except FileNotFoundError:
            print(f"ERROR: Image not found at {img_path}.")
            # Return a dummy tensor of the correct expected size by Albumentations' ToTensorV2
            return torch.randn(3, IMAGE_SIZE, IMAGE_SIZE), torch.tensor(-1)
        except Exception as e:
            print(f"ERROR: Could not read image {img_path}. Error: {e}")
            return torch.randn(3, IMAGE_SIZE, IMAGE_SIZE), torch.tensor(-1)

        if self.transform:
            augmented = self.transform(image=image_np)
            image = augmented['image'] 
        else:
            image = TF.to_tensor(image_pil) 
           

        return image, label

    def get_class_weights(self):
        if sum(self.class_counts) == 0 or len(self.classes) == 0:
            return None
        total_samples = float(sum(self.class_counts))
        # Standard weighting: N / (C * N_c)
        weights = [total_samples / (len(self.classes) * count) if count > 0 else 0 for count in self.class_counts]
        # For Focal Loss alpha, sometimes weights are normalized (e.g., sum to 1) or directly used.
        # The provided FocalLoss class can handle unnormalized weights like these.
        return torch.tensor(weights, dtype=torch.float) 

    def get_sampler_weights(self):
        if sum(self.class_counts) == 0:
            return None
        class_weights_for_sampler = [1.0 / count if count > 0 else 0 for count in self.class_counts]
        sample_weights = [class_weights_for_sampler[label] for label in self.labels]
        return torch.DoubleTensor(sample_weights)

# ---------------- DATA SCANNING AND SPLITTING FUNCTION ----------------
def scan_and_split_data(base_dir, train_ratio, val_ratio, test_ratio, num_expected_classes, random_seed):
    global CLASS_TO_IDX, CLASS_NAMES

    print(f"Scanning data directory: {base_dir}")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base data directory not found: {base_dir}")

    all_image_paths = []
    all_labels_int = []

    temp_classes = sorted([d.name for d in os.scandir(base_dir) if d.is_dir()])
    if not temp_classes:
        raise FileNotFoundError(f"No subdirectories (classes) found in {base_dir}.")

    CLASS_NAMES = temp_classes
    CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}

    if num_expected_classes != len(CLASS_NAMES):
        raise ValueError(
            f"Global NUM_CLASSES ({num_expected_classes}) does not match inferred classes ({len(CLASS_NAMES)}: {CLASS_NAMES})."
        )

    print(f"\nFound {len(CLASS_NAMES)} classes: {CLASS_TO_IDX}")
    initial_class_counts = [0] * len(CLASS_NAMES)
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    for class_name in CLASS_NAMES:
        class_label_int = CLASS_TO_IDX[class_name]
        class_dir_path = os.path.join(base_dir, class_name)
        img_count_in_class = 0
        for img_name in os.listdir(class_dir_path):
            if img_name.lower().endswith(supported_extensions):
                img_path = os.path.join(class_dir_path, img_name)
                all_image_paths.append(img_path)
                all_labels_int.append(class_label_int)
                initial_class_counts[class_label_int] += 1
                img_count_in_class +=1
        if img_count_in_class == 0:
            print(f"Warning: No images found in class directory: {class_dir_path}")

    if not all_image_paths:
        raise FileNotFoundError(f"No images with supported extensions found in subdirectories of {base_dir}.")

    print(f"\nTotal images found: {len(all_image_paths)}")
    print("Overall Class Distribution (before split):")
    for i, class_name_val in enumerate(CLASS_NAMES):
        print(f"  Class '{class_name_val}': {initial_class_counts[i]} samples")
        if initial_class_counts[i] < (1/test_ratio) and initial_class_counts[i] < (1/val_ratio) :
             print(f"    Warning: Class '{class_name_val}' has few samples ({initial_class_counts[i]}).")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_image_paths, all_labels_int,
        test_size=(val_ratio + test_ratio),
        stratify=all_labels_int,
        random_state=random_seed
    )
    relative_test_size = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=random_seed
    )
    
    print(f"\nData Splitting Complete (Stratified):")
    print(f"  Training set: {len(train_paths)} images")
    print(f"  Validation set: {len(val_paths)} images")
    print(f"  Test set: {len(test_paths)} images")

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

# ---------------- DATA TRANSFORMS (USING ALBUMENTATIONS) ----------------
# ImageNet mean and std (still relevant for normalization)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform_alb = A.Compose([
    A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.0), p=1.0), 
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5), 
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),   
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),                                   
    A.OneOf([
        A.MotionBlur(p=0.5),
        A.MedianBlur(blur_limit=3, p=0.5),
        A.Blur(blur_limit=3, p=0.5),
    ], p=0.3), # p for OneOf itself
   
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()                    
])

val_test_transform_alb = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE), 
   
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

# ---------------- DATA LOADERS (SETUP AFTER SPLITTING) ----------------
print(f"Using device: {DEVICE}")
print("Scanning data, performing splits, and initializing datasets/dataloaders...")

(train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = scan_and_split_data(
    ALL_DATA_DIR, TRAIN_SPLIT_RATIO, VAL_SPLIT_RATIO, TEST_SPLIT_RATIO, NUM_CLASSES, MANUAL_SEED
)

train_dataset = CustomImageDataset(
    image_paths_list=train_paths, labels_list=train_labels,
    master_class_to_idx=CLASS_TO_IDX, master_classes_list=CLASS_NAMES,
    transform=train_transform_alb, dataset_name="Training Set"
)
val_dataset = CustomImageDataset(
    image_paths_list=val_paths, labels_list=val_labels,
    master_class_to_idx=CLASS_TO_IDX, master_classes_list=CLASS_NAMES,
    transform=val_test_transform_alb, dataset_name="Validation Set"
)
test_dataset = CustomImageDataset(
    image_paths_list=test_paths, labels_list=test_labels,
    master_class_to_idx=CLASS_TO_IDX, master_classes_list=CLASS_NAMES,
    transform=val_test_transform_alb, dataset_name="Test Set"
)

sampler = None
shuffle_train = True
if USE_WEIGHTED_SAMPLER and len(train_dataset) > 0:
    print("\nUsing WeightedRandomSampler for training data.")
    sampler_weights = train_dataset.get_sampler_weights()
    if sampler_weights is not None and len(sampler_weights) > 0:
        sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))
        shuffle_train = False
    else:
        print("Warning: Could not get sampler weights. Proceeding without sampler.")
        USE_WEIGHTED_SAMPLER = False

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle_train, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS
)

print(f"\nTraining set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}, Test set size: {len(test_dataset)}")
if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0 :
    raise ValueError("One or more datasets (Train/Val/Test) are empty after splitting.")

# ---------------- MODEL SETUP ----------------
print(f"\nSetting up {MODEL_NAME} model with {NUM_CLASSES} output classes.")

# Using torchvision.models for EfficientNet
if MODEL_NAME == 'EfficientNet-B1':
    weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
    model = models.efficientnet_b1(weights=weights)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
elif MODEL_NAME == 'EfficientNet-B0': # Keep B0 as an option
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
else:
    raise ValueError(f"Model {MODEL_NAME} not supported or weights not specified correctly.")
model = model.to(DEVICE)

def set_parameter_requires_grad(model_to_set, feature_extracting):
    if feature_extracting:
        print("Freezing backbone layers, training only classifier.")
        for param in model_to_set.parameters():
            param.requires_grad = False
        # EfficientNet classifier is typically model.classifier
        if hasattr(model_to_set, 'classifier') and isinstance(model_to_set.classifier, nn.Sequential):
            for param in model_to_set.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model_to_set, 'classifier') and isinstance(model_to_set.classifier, nn.Linear): 
             for param in model_to_set.classifier.parameters():
                param.requires_grad = True
        else:
            print("Warning: Could not automatically unfreeze classifier. Ensure it's done manually if needed.")

    else:
        print("Unfreezing all layers for full model training.")
        for param in model_to_set.parameters():
            param.requires_grad = True
set_parameter_requires_grad(model, FREEZE_BACKBONE_INITIALLY)

# ---------------- LOSS FUNCTION, OPTIMIZER, SCHEDULER ----------------
focal_loss_alpha = None
if USE_CLASS_WEIGHTS_IN_LOSS:
    print("\nCalculating class weights for Focal Loss alpha (based on training set)...")
    # get_class_weights returns tensor, already suitable for FocalLoss alpha
    focal_loss_alpha = train_dataset.get_class_weights()
    if focal_loss_alpha is not None:
        print(f"Using class weights for Focal Loss alpha: {focal_loss_alpha.cpu().numpy()}")
        # FocalLoss class will move alpha to device
    else:
        print("Warning: Could not get class weights for Focal Loss. Proceeding without alpha.")

# Using FocalLoss
criterion = FocalLoss(alpha=focal_loss_alpha, gamma=2.0) 
print(f"Using FocalLoss with gamma=2.0 and alpha={'calculated' if focal_loss_alpha is not None else 'None'}")


params_to_optimize = []
# Determine parameters to optimize based on freeze strategy
if not FREEZE_BACKBONE_INITIALLY or (FREEZE_BACKBONE_INITIALLY and EPOCHS_TO_TRAIN_CLASSIFIER_ONLY == 0) :
    print(f"Setting up optimizer for full model training: Backbone LR={LEARNING_RATE_BACKBONE}, Classifier LR={LEARNING_RATE}")
   
    if hasattr(model, 'features') and hasattr(model, 'classifier'):
        params_to_optimize = [
            {"params": filter(lambda p: p.requires_grad, model.features.parameters()), "lr": LEARNING_RATE_BACKBONE},
            {"params": filter(lambda p: p.requires_grad, model.classifier.parameters()), "lr": LEARNING_RATE}
        ]
    else: # Fallback if model structure is different
        print("Warning: Model structure not as expected for differential LR. Optimizing all grad-requiring params with LEARNING_RATE.")
        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())

else: # Initially training only classifier
    print(f"Setting up optimizer for classifier-only training: LR={LEARNING_RATE}")
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())


optimizer = optim.AdamW(params_to_optimize, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7) 
scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))

# ---------------- EVALUATION FUNCTION ----------------
def evaluate_model(model_eval, dataloader_eval, criterion_eval, phase="Validation"):
    model_eval.eval()
    all_labels_eval = []
    all_preds_eval = []
    total_eval_loss = 0.0
    total_eval_corrects = 0

    print(f"\nRunning {phase} evaluation...")
    with torch.no_grad():
        for inputs_eval, labels_eval in tqdm(dataloader_eval, desc=f"Evaluating ({phase})"):
            inputs_eval = inputs_eval.to(DEVICE)
            labels_eval = labels_eval.to(DEVICE)
            autocast_enabled_eval = (DEVICE.type == 'cuda')
            autocast_dtype_eval = torch.float16 if DEVICE.type == 'cuda' else None # AMP
            with autocast(DEVICE.type, enabled=autocast_enabled_eval, dtype=autocast_dtype_eval):
                outputs_eval = model_eval(inputs_eval)
                loss_eval = criterion_eval(outputs_eval, labels_eval)
            total_eval_loss += loss_eval.item() * inputs_eval.size(0)
            _, preds_eval = torch.max(outputs_eval, 1)
            total_eval_corrects += torch.sum(preds_eval == labels_eval.data)
            all_labels_eval.extend(labels_eval.cpu().numpy())
            all_preds_eval.extend(preds_eval.cpu().numpy())

    avg_eval_loss = total_eval_loss / len(dataloader_eval.dataset) if len(dataloader_eval.dataset) > 0 else 0
    avg_eval_acc = total_eval_corrects.double() / len(dataloader_eval.dataset) if len(dataloader_eval.dataset) > 0 else 0
    avg_eval_f1 = f1_score(all_labels_eval, all_preds_eval, average='weighted', zero_division=0)

    print(f"\n--- {phase} Set Evaluation Report ---")
    print(f"Overall {phase} Loss: {avg_eval_loss:.4f}")
    print(f"Overall {phase} Accuracy: {avg_eval_acc:.4f}")
    print(f"Overall {phase} Weighted F1-Score: {avg_eval_f1:.4f}")

    if CLASS_NAMES and len(all_labels_eval) > 0:
        print(f"\nClassification Report ({phase} Set):")
        try:
            report = classification_report(all_labels_eval, all_preds_eval, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES, digits=4, zero_division=0)
            print(report)
        except Exception as e:
            print(f"Could not generate full classification report: {e}")
            try:
                report_fallback = classification_report(all_labels_eval, all_preds_eval, digits=4, zero_division=0)
                print("Fallback report (without all target names):\n", report_fallback)
            except Exception as e_fallback:
                print(f"Fallback classification report also failed: {e_fallback}")

        print(f"\nConfusion Matrix ({phase} Set):")
        cm = confusion_matrix(all_labels_eval, all_preds_eval, labels=range(len(CLASS_NAMES)))
        header = "      " + " ".join([f"{name[:6]:>6}" for name in CLASS_NAMES])
        print(header)
        print("      " + "-" * (len(CLASS_NAMES) * 7 -1))
        for i, row_cm in enumerate(cm):
            row_str = f"{CLASS_NAMES[i][:6]:>6}|"
            row_str += " ".join([f"{val:6}" for val in row_cm])
            print(row_str)
    else:
        print("Not enough data or class names not available for detailed report.")
    print(f"--- End of {phase} Set Evaluation Report ---")
    return avg_eval_loss, avg_eval_acc, avg_eval_f1

# ---------------- TRAINING LOOP ----------------
def train_model_main(model_train, criterion_train, optimizer_train, scheduler_train,
                     train_loader_loop, val_loader_loop, test_loader_loop):
    since = time.time()
    best_model_wts = copy.deepcopy(model_train.state_dict())
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # This flag tracks if the backbone is currently frozen or not during training
    current_phase_backbone_frozen = FREEZE_BACKBONE_INITIALLY

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print('-' * 15)

        # --- Phase: Unfreeze backbone if conditions met ---
        # Check if it's time to unfreeze the backbone
        if current_phase_backbone_frozen and epoch >= EPOCHS_TO_TRAIN_CLASSIFIER_ONLY:
            print(f"\nEpoch {epoch + 1}: Unfreezing backbone and re-initializing optimizer for full model training.")
            current_phase_backbone_frozen = False # Mark backbone as unfrozen
            set_parameter_requires_grad(model_train, False) # Unfreeze all layers

            # Re-initialize optimizer with differential learning rates for the whole model
            if hasattr(model_train, 'features') and hasattr(model_train, 'classifier'):
                params_to_optimize_full = [
                    {"params": filter(lambda p: p.requires_grad, model_train.features.parameters()), "lr": LEARNING_RATE_BACKBONE},
                    {"params": filter(lambda p: p.requires_grad, model_train.classifier.parameters()), "lr": LEARNING_RATE}
                ]
            else: # Fallback
                print("Warning: Re-initializing optimizer for all grad-requiring params with main LEARNING_RATE.")
                params_to_optimize_full = filter(lambda p: p.requires_grad, model_train.parameters())
            
            optimizer_train = optim.AdamW(params_to_optimize_full, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            # Reset scheduler with the new optimizer. T_max could be remaining epochs.
            scheduler_train = CosineAnnealingLR(optimizer_train, T_max=(NUM_EPOCHS - epoch), eta_min=1e-7) # Use new eta_min
            print(f"Optimizer re-initialized. Backbone LR: {LEARNING_RATE_BACKBONE}, Classifier LR: {LEARNING_RATE}")


        # --- Training Phase ---
        model_train.train()
        running_loss_train = 0.0
        running_corrects_train = 0
        all_labels_train_epoch = []
        all_preds_train_epoch = []
        progress_bar_train = tqdm(train_loader_loop, desc=f"Train Epoch {epoch+1}", leave=False)
        for inputs, labels in progress_bar_train:
            inputs = inputs.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
            optimizer_train.zero_grad(set_to_none=True) 
            autocast_enabled_train = (DEVICE.type == 'cuda')
            autocast_dtype_train = torch.float16 if DEVICE.type == 'cuda' else None # AMP
            with autocast(DEVICE.type, enabled=autocast_enabled_train, dtype=autocast_dtype_train):
                outputs = model_train(inputs)
                loss = criterion_train(outputs, labels)
                _, preds = torch.max(outputs, 1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_train) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=1.0) # Gradient clipping
            scaler.step(optimizer_train)
            scaler.update()
            running_loss_train += loss.item() * inputs.size(0)
            running_corrects_train += torch.sum(preds == labels.data)
            all_labels_train_epoch.extend(labels.cpu().numpy())
            all_preds_train_epoch.extend(preds.cpu().numpy())
            progress_bar_train.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{torch.sum(preds == labels.data).item()/inputs.size(0):.4f}",
                lr=f"{optimizer_train.param_groups[0]['lr']:.1e}"
            )
        epoch_loss_train = running_loss_train / len(train_loader_loop.dataset) if len(train_loader_loop.dataset) > 0 else 0
        epoch_acc_train = running_corrects_train.double() / len(train_loader_loop.dataset) if len(train_loader_loop.dataset) > 0 else 0
        epoch_f1_train = f1_score(all_labels_train_epoch, all_preds_train_epoch, average='weighted', zero_division=0)
        current_lr = optimizer_train.param_groups[0]['lr']
        print(f"Train Epoch {epoch+1} Summary: Loss: {epoch_loss_train:.4f} Acc: {epoch_acc_train:.4f} F1: {epoch_f1_train:.4f} LR: {current_lr:.1e}")
        scheduler_train.step()

        # --- Validation Phase ---
        val_loss, val_acc, val_f1 = evaluate_model(model_train, val_loader_loop, criterion_train, phase="Validation")
        if val_f1 > best_val_f1:
            print(f"Validation F1 improved from {best_val_f1:.4f} to {val_f1:.4f}. Saving model.")
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model_train.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation F1 did not improve for {epochs_no_improve} epoch(s). Best F1: {best_val_f1:.4f}")
        if epochs_no_improve >= PATIENCE_EARLY_STOPPING:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs. No improvement in Val F1 for {PATIENCE_EARLY_STOPPING} epochs.")
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation F1-Score achieved: {best_val_f1:.4f} (Loss: {best_val_loss:.4f})')
    model_train.load_state_dict(best_model_wts)

    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints', 'xray_classification')
    os.makedirs(checkpoint_dir, exist_ok=True)
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    best_model_filename = f"{MODEL_NAME.replace('-', '_')}_{current_date}_best_val_f1_{best_val_f1:.4f}.pth"
    best_model_path = os.path.join(checkpoint_dir, best_model_filename)
    torch.save(model_train.state_dict(), best_model_path)
    print(f"Best model (based on validation F1) saved to {best_model_path}")

    print("\n--- Final Evaluation on Validation Set with Best Model ---")
    evaluate_model(model_train, val_loader_loop, criterion_train, phase="Final Validation")

    print("\n--- Final Evaluation on Test Set with Best Model ---")
    if len(test_loader_loop.dataset) > 0:
        evaluate_model(model_train, test_loader_loop, criterion_train, phase="Test")
    else:
        print("Test dataset is empty. Skipping test evaluation.")
    return model_train, best_model_path

# ---------------- RUN TRAINING ----------------
if __name__ == '__main__':
    print("--- Starting X-Ray Image Classification Training ---")
    if not os.path.isdir(ALL_DATA_DIR):
        print(f"ERROR: Base data directory '{ALL_DATA_DIR}' not found. Please check path.")
        exit()

    try:
        if DEVICE.type == 'cuda':
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            torch.cuda.empty_cache()
            print("Cleared CUDA cache before starting training.")

        print(f"\nTraining with model: {MODEL_NAME}, Image Size: {IMAGE_SIZE}")
        print(f"Splitting data from: {ALL_DATA_DIR}")
        print(f"Split Ratios: Train={TRAIN_SPLIT_RATIO*100}%, Val={VAL_SPLIT_RATIO*100}%, Test={TEST_SPLIT_RATIO*100}%")
        print(f"Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE} (Backbone LR: {LEARNING_RATE_BACKBONE})")
        print(f"Epochs: {NUM_EPOCHS}, Early Stopping Patience: {PATIENCE_EARLY_STOPPING}")
        print(f"Loss Function: FocalLoss (gamma=2.0)")
        print(f"Data Augmentation: Albumentations")
        print(f"Weighted Sampler: {USE_WEIGHTED_SAMPLER}, Focal Loss Alpha from Class Weights: {USE_CLASS_WEIGHTS_IN_LOSS}")
        print(f"Initial Backbone Freeze: {FREEZE_BACKBONE_INITIALLY}, Unfreeze after {EPOCHS_TO_TRAIN_CLASSIFIER_ONLY} epochs if applicable.")

        trained_model, saved_model_path = train_model_main(
            model, criterion, optimizer, scheduler,
            train_loader, val_loader, test_loader
        )
        print(f"\n--- Training Finished Successfully ---")
        print(f"The best model was saved to: {saved_model_path}")

    except FileNotFoundError as e:
        print(f"\n--- TRAINING HALTED: FILE NOT FOUND ---")
        print(f"Error: {e}")
    except ValueError as e:
        print(f"\n--- TRAINING HALTED: CONFIGURATION OR DATA ERROR ---")
        print(f"Error: {e}")
    except torch.cuda.OutOfMemoryError:
        print(f"\n--- TRAINING HALTED: CUDA OUT OF MEMORY ---")
        print("Try reducing `BATCH_SIZE` further or using a smaller model/image size.")
    except ImportError as e:
        print(f"\n--- TRAINING HALTED: IMPORT ERROR ---")
        print(f"Error: {e}")
        if "albumentations" in str(e).lower():
            print("Please install Albumentations: pip install albumentations")
    except Exception as e_main:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED DURING TRAINING ---")
        import traceback
        traceback.print_exc()

