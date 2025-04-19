import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path so we can import utilities
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.common_utils import set_seeds, get_configs
from utils.model_utils import train_one_epoch, validate_one_epoch

os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Set to the GPU you want to use

def get_data_loaders(data_root, resize_dim, batch_size, val_ratio=0.2, seed=42):
    """
    Prepare train and validation DataLoaders with a stratified split.
    Uses Resize+CenterCrop to match pretrained model input.
    """
    # Define transform matching pretrained ResNet50 expectations
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(resize_dim),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Load entire dataset
    full_dataset = ImageFolder(root=os.path.join(data_root, "train"), transform=transform)
    labels = full_dataset.targets
    indices = list(range(len(full_dataset)))

    # Stratified split
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=labels,
        random_state=seed
    )

    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    val_ds   = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, full_dataset.classes

def build_finetune_resnet50(num_classes, freeze_until_layer=3):
    """
    Load a pretrained ResNet50, freeze layers up to `freeze_until_layer`,
    unfreeze the rest (including the final fc), then replace fc for num_classes.
    freeze_until_layer = number of ResNet 'layerX' modules to freeze: 0..4
    """
    model = torchvision.models.resnet50(weights="IMAGENET1K_V2")

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Map layer index to attribute name
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    # Unfreeze the top blocks
    for layer_name in layer_names[freeze_until_layer:]:
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = True

    # Replace the final fc layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # Ensure classifier parameters are trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

def evaluate_on_test(model, data_root, resize_dim, batch_size, device):
    """
    Load the test set, evaluate the model, and return accuracy.
    """
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(resize_dim),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    test_dataset = ImageFolder(root=os.path.join(data_root, "test"), transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total

def main(static_config):
    model_config = static_config['model_config']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seeds(model_config['seed'])

    # Prepare data loaders
    train_loader, val_loader, class_names = get_data_loaders(
        data_root=static_config['data_root'], 
        resize_dim=model_config['resize_dim'], 
        batch_size=model_config['batch_size'], 
        val_ratio=model_config['val_ratio'], 
        seed=model_config['seed']
    )
    num_classes = len(class_names)
    print("Successfully loaded data.")
    print(f"Number of classes: {num_classes}")

    # Build model for fine‑tuning
    model = build_finetune_resnet50(num_classes, freeze_until_layer=model_config['freeze_until_layer'])
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)
    print("Successfully built model.")

    # # Optimizer: only parameters with requires_grad=True
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=model_config['learning_rate'],
    )
    criterion = nn.CrossEntropyLoss()
    print("Successfully set up optimizer and loss function.")
    
    patience = static_config["patience"]
    best_val_acc = float('-inf')
    patience_counter = 0
    os.makedirs(static_config['output_dir'], exist_ok=True)
    best_model_path = os.path.join(static_config["output_dir"], "partB_Q3_best_resnet50_old.pth")

    epochs = model_config['epochs']
    # Training loop
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = validate_one_epoch(model, val_loader,   criterion, device)
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f} | "
              f"Val Loss:   {val_loss:.3f}, Val Acc:   {val_acc:.3f}")

        # --- Early Stopping Logic ---
        if val_acc > best_val_acc:   # slight delta to avoid float issues
            best_val_acc = val_acc
            patience_counter = 0
            # Save the best model weights
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation accuracy improved to {best_val_acc:.3f}. Checkpoint saved.")
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch}. Best val_scc={best_val_acc:.3f}")
                break
    
    print(f"Training completed. Best model saved at {best_model_path}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    
    # Load the best weights before final evaluation
    print("Loading the best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))

    # Evaluate on test set
    test_acc = evaluate_on_test(model, static_config['data_root'], model_config['resize_dim'], model_config['batch_size'], device)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    config = get_configs(project_root, "configs.yaml")['part_b_configs']['solution_3_configs']
    main(config)
 