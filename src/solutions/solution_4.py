import os
import sys
import torch
import yaml
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.common_utils import set_seeds
from src.utils.model_utils import train_one_epoch, validate_one_epoch
from src.data.data_loader import load_inaturalist_train_val
from src.models.implementation import MyCNNExtended

# Best hyperparameters provided.
best_hparams = {
    "activation_fn": "mish",
    "batch_norm": True,
    "batch_size": 64,
    "data_augmentation": False,
    "dense_neurons": 256,
    "dropout_rate": 0.3,
    "epochs": 10,
    "filter_organization": "double_each_layer",
    "kernel_size": 3,
    "learning_rate": 0.0001,
    "num_filters": 16,
    "resize_dim": 352
}

def get_configs(config_filename):
    with open(os.path.join(project_root, "config", config_filename), 'r') as f:
        config = yaml.safe_load(f)
    return config

# --------------------------
# Function: Load Test Data
# --------------------------
def load_test_data(test_dir, resize_dim):
    """
    Loads the test dataset from 'test_dir' using ImageFolder.
    Applies a minimal transformation: resizing then conversion to tensor.
    
    :param test_dir: Directory containing test data (with subfolders per class).
    :param resize_dim: Desired image size (square, e.g., 352) (should be a multiple of 32).
    :return: (test_dataset, class_names)
    """
    transform = T.Compose([
        T.Resize((resize_dim, resize_dim)),
        T.ToTensor()
    ])
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
    class_names = test_dataset.classes
    return test_dataset, class_names

# --------------------------
# Function: Evaluate Model on Test Data
# --------------------------
def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on the test dataset and returns the accuracy.
    
    :param model: Trained model.
    :param test_loader: DataLoader for the test dataset.
    :param device: cuda or cpu.
    :return: Test accuracy as float.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# --------------------------
# Function: Plot Predictions Grid (10x3)
# --------------------------
def plot_predictions(model, test_dataset, class_names, device, grid_shape=(10, 3)):
    """
    Plots a creative 10×3 grid (or grid_shape rows x cols) of test sample images,
    along with their true labels and predictions. Correct predictions are shown in green,
    incorrect in red. Additional creative elements (such as a grid title and custom layout)
    are added.
    
    :param model: Trained model (in eval mode).
    :param test_dataset: Test dataset (instance of ImageFolder).
    :param class_names: List of class names.
    :param device: Device to run inference on.
    :param grid_shape: Tuple (rows, cols) for grid arrangement.
    """
    model.eval()
    num_samples = grid_shape[0] * grid_shape[1]
    # Randomly sample indices without replacement
    indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)
    samples = [test_dataset[i] for i in indices]
    
    # Create the figure
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(grid_shape[1]*3, grid_shape[0]*3))
    fig.suptitle("Creative 10×3 Grid of Test Predictions", fontsize=18, color="navy")
    
    with torch.no_grad():
        for ax, (img, true_label) in zip(axes.flatten(), samples):
            input_tensor = img.unsqueeze(0).to(device)
            output = model(input_tensor)
            _, pred_label = torch.max(output, 1)
            pred_label = pred_label.item()
            
            # Convert image tensor to numpy array for plotting
            np_img = img.cpu().numpy().transpose(1, 2, 0)
            np_img = np.clip(np_img, 0, 1)
            ax.imshow(np_img)
            ax.axis("off")
            # Title with true and predicted labels; green if correct, red if incorrect.
            title_color = "green" if pred_label == true_label else "red"
            ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", 
                         fontsize=10, color=title_color)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("test_predictions_grid.png", dpi=300)
    plt.show()

# --------------------------
# Main training function (using best hyperparameters from Q2)
# --------------------------
def train_and_evaluate_best():
    """
    This function trains the best model on training data (with validation monitoring),
    then evaluates the final model on test data and displays a creative 10×3 grid of test images.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seeds(42)
    
    # Determine activation function.
    if best_hparams["activation_fn"].lower() == "mish":
        act_fn = nn.Mish
    elif best_hparams["activation_fn"].lower() == "relu":
        act_fn = nn.ReLU
    elif best_hparams["activation_fn"].lower() == "gelu":
        act_fn = nn.GELU
    elif best_hparams["activation_fn"].lower() == "silu":
        act_fn = nn.SiLU
    else:
        act_fn = nn.ReLU
    
    # Instantiate the model with best hyperparameters.
    model = MyCNNExtended(
        in_channels=3,
        num_filters=best_hparams["num_filters"],
        kernel_size=best_hparams["kernel_size"],
        activation_fn=act_fn,
        dense_neurons=best_hparams["dense_neurons"],
        image_height=best_hparams["resize_dim"],
        image_width=best_hparams["resize_dim"],
        filter_organization=best_hparams["filter_organization"],
        batch_norm=best_hparams["batch_norm"],
        dropout_rate=best_hparams["dropout_rate"]
    )
    
    # Use multiple GPUs if available.
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Use the training data (with 20% reserved as validation) similar to Q2.
    static_config = get_configs('configs.yaml')['solution_2_configs']
    train_dir = os.path.join(static_config["data_root"], "train")
    train_dataset, val_dataset, class_names = load_inaturalist_train_val(
        data_dir=train_dir,
        val_ratio=0.2,
        seed=42,
        augment=best_hparams["data_augmentation"],
        resize_dim=best_hparams["resize_dim"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=best_hparams["batch_size"], shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=best_hparams["batch_size"], shuffle=False, num_workers=4)
    
    optimizer = optim.Adam(model.parameters(), lr=best_hparams["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    # Training loop
    for epoch in range(best_hparams["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"[Epoch {epoch+1}/{best_hparams['epochs']}] "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    
    print("Training complete.")
    
    # --------------------------
    # Evaluate on test data.
    # --------------------------
    test_dir = os.path.join(static_config["data_root"], "test")
    test_dataset, class_names = load_test_data(test_dir, best_hparams["resize_dim"])
    test_loader = DataLoader(test_dataset, batch_size=best_hparams["batch_size"], shuffle=False, num_workers=4)
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # --------------------------
    # Plot a creative 10x3 grid of predictions on test data.
    # --------------------------
    plot_predictions(model, test_dataset, class_names, device, grid_shape=(10, 3))

if __name__ == "__main__":
    train_and_evaluate_best()
