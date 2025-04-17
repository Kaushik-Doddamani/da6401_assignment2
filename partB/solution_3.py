import os
import sys
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Utility functions and configurations
from utils.common_utils import set_seeds, get_activation_fn, get_configs
from utils.model_utils import train_one_epoch, validate_one_epoch, evaluate_model_on_test_data
from data.data_loader import load_inaturalist_train_val_data, load_inaturalist_test_data

model_config = {
    "seed": 42,
    "resize_dim": 352,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std":  [0.229, 0.224, 0.225],
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 1e-4,
    "data_augmentation": False,
}

# --------------------------------------------------------------------------------
# Plot Predictions Grid (10×3)
# --------------------------------------------------------------------------------
def plot_predictions(model, test_dataset, class_names, device, output_dir, grid_shape=(10, 3)):
    """
    Plots a creative 10×3 grid of test samples with true vs predicted labels.
    Correct predictions are green; incorrect are red.

    Args:
        model: Trained PyTorch model (in eval mode).
        test_dataset: ImageFolder or Subset with test images.
        class_names: List of class names.
        device: 'cuda' or 'cpu'.
        output_dir: Directory to save the plot.
        grid_shape: Tuple (rows, cols) for layout.

    Returns:
        fig: Matplotlib figure object.
        plot_path: Filepath of the saved image.
    """
    model.eval()
    num_samples = grid_shape[0] * grid_shape[1]
    indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)
    samples = [test_dataset[i] for i in indices]

    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], 
                             figsize=(grid_shape[1]*3, grid_shape[0]*3))
    fig.suptitle("Top-Block Fine-Tuning: Test Predictions", fontsize=16, color="navy")

    with torch.no_grad():
        for ax, (img, true_label) in zip(axes.flatten(), samples):
            x = img.unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).item()

            img_np = img.cpu().numpy().transpose(1,2,0)
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)
            ax.axis('off')
            color = 'green' if pred == true_label else 'red'
            ax.set_title(f"T: {class_names[true_label]}\nP: {class_names[pred]}", color=color, fontsize=9)

    plt.tight_layout(rect=[0,0.03,1,0.95])
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'partB_q3_predictions_grid.png')
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    return fig, plot_path

# --------------------------------------------------------------------------------
# Main Fine‑Tuning Function (Top-Block Strategy)
# --------------------------------------------------------------------------------
def fine_tune_top_block(static_config):
    """
    Fine‑tune a pre‑trained ResNet50 by freezing all layers except the last block + head.
    Logs metrics and a prediction grid to W&B.
    """
    # Initialize W&B run
    run = wandb.init(
        project=static_config['wandb_project'],
        name=static_config['wandb_run_name'],
        config=model_config
    )

    run_config = run.config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seeds(42)

    # ------------------------------
    # Data loading
    # ------------------------------
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(run_config.resize_dim),
        T.ToTensor(),
        T.Normalize(mean=run_config.imagenet_mean, std=run_config.imagenet_std)
    ])

    train_ds, val_ds, class_names = load_inaturalist_train_val_data(
        data_dir=os.path.join(run_config.data_root, 'train'),
        val_ratio=0.2,
        seed=run_config.seed,
        augment=run_config.data_augmentation,
        resize_dim=run_config.resize_dim
    )
    train_loader = DataLoader(train_ds, batch_size=run_config.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=run_config.batch_size, shuffle=False, num_workers=4)

    # ------------------------------
    # Model preparation
    # ------------------------------
    # Load ResNet50 pretrained on ImageNet
    model = torchvision.models.resnet50(pretrained=True)
    # Freeze all layers
    for param in model.parameters(): param.requires_grad = False
    # Unfreeze last residual block + fc head
    for param in model.layer4.parameters(): param.requires_grad = True
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, len(class_names))
    # Move to device (with DataParallel if multi‑GPU)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # ------------------------------
    # Optimizer & loss
    # ------------------------------
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=run_config.learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    # ------------------------------
    # Training loop with early stopping
    # ------------------------------
    best_val_acc = 0.0
    no_improve = 0
    ckpt_path = os.path.join(static_config['output_dir'], 'partB_q3_best.pth')
    os.makedirs(static_config['output_dir'], exist_ok=True)

    for epoch in range(run_config.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        wandb.log({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        print(f"Epoch {epoch+1}/{run_config.epochs} | "
              f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

        # Early stopping & checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= run_config.patience:
                print("Early stopping triggered.")
                break

    # Load best checkpoint
    model.load_state_dict(torch.load(ckpt_path))

    # ------------------------------
    # Evaluate on test data
    # ------------------------------
    test_ds, _ = load_inaturalist_test_data(
        data_dir=os.path.join(run_config.data_root, 'test'),
        resize_dim=run_config.resize_dim
    )
    test_loader = DataLoader(test_ds, batch_size=run_config.batch_size, shuffle=False, num_workers=4)
    test_acc = evaluate_model_on_test_data(model, test_loader, device)
    wandb.log({'test_accuracy': test_acc})
    print(f"Test Accuracy: {test_acc:.3f}")

    # ------------------------------
    # Plot and log predictions grid
    # ------------------------------
    fig, plot_path = plot_predictions(
        model, test_ds, class_names, device,
        output_dir=static_config['output_dir'], grid_shape=(10,3)
    )
    run.log({ 'predictions_grid': wandb.Image(plot_path) })
    run.finish()

# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    # Load static config
    config = get_configs(project_root, 'configs.yaml')['part_b_configs']['solution_3_configs']
    fine_tune_top_block(config)
