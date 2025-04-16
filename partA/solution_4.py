import os
import sys
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.common_utils import set_seeds, get_activation_fn, get_configs
from utils.model_utils import train_one_epoch, validate_one_epoch, evaluate_model_on_test_data
from data.data_loader import load_inaturalist_train_val_data, load_inaturalist_test_data
from models.implementation import MyCNNExtended

# Best hyperparameters provided.
best_hparams = {
    "activation_fn": "mish",
    "batch_norm": True,
    "batch_size": 64,
    "data_augmentation": False,
    "dense_neurons": 256,
    "dropout_rate": 0.3,
    "epochs": 20,
    "filter_organization": "double_each_layer",
    "kernel_size": 3,
    "learning_rate": 0.0001,
    "num_filters": 32,
    "resize_dim": 224
}


# --------------------------
# Function: Plot Predictions Grid (10x3)
# --------------------------
def plot_predictions(model, test_dataset, class_names, device, output_dir, grid_shape=(10, 3)):
    """
    Plots a creative 10×3 grid (or grid_shape rows x cols) of test sample images,
    along with their true labels and predictions. Correct predictions are shown in green,
    incorrect in red. Additional creative elements (such as a grid title and custom layout)
    are added.
    
    :param model: Trained model (in eval mode).
    :param test_dataset: Test dataset (instance of ImageFolder).
    :param class_names: List of class names.
    :param device: Device to run inference on.
    :param output_dir: Directory to save the output figure.
    :param grid_shape: Tuple (rows, cols) for grid arrangement.
    :return: The figure object (matplotlib.figure.Figure)
    """
    model.eval()
    num_samples = grid_shape[0] * grid_shape[1]
    # Randomly sample indices without replacement
    indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)
    samples = [test_dataset[i] for i in indices]

    # Create the figure
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(grid_shape[1] * 3, grid_shape[0] * 3))
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
    # Save the figure so that we can log it with wandb
    output_filepath = f"{output_dir}/part_a_q4_test_predictions_grid.png"
    plt.savefig(output_filepath, dpi=300)
    # Optionally, you can call plt.show() if you want to display the plot interactively.
    plt.show()
    return fig, output_filepath


# --------------------------
# Main training function (using best hyperparameters from Q2)
# --------------------------
def train_and_evaluate_best(static_config):
    """
    This function trains the best model on training data (with validation monitoring),
    then evaluates the final model on test data and displays a creative 10×3 grid of test images.
    Metrics and plots are logged with wandb.
    """
    # Initialize wandb run with the best hyperparameters.
    run = wandb.init(project=static_config['wandb_project'], config=best_hparams)
    run.name = static_config['wandb_run_name']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seeds(42)

    # Determine activation function.
    best_act_fn = get_activation_fn(best_hparams["activation_fn"])

    # Instantiate the model with best hyperparameters.
    model = MyCNNExtended(
        in_channels=3,
        num_filters=best_hparams["num_filters"],
        kernel_size=best_hparams["kernel_size"],
        activation_fn=best_act_fn,
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
    train_dir = os.path.join(static_config["data_root"], "train")
    train_dataset, val_dataset, class_names = load_inaturalist_train_val_data(
        data_dir=train_dir,
        val_ratio=0.2,
        seed=42,
        augment=best_hparams["data_augmentation"],
        resize_dim=best_hparams["resize_dim"]
    )

    train_loader = DataLoader(train_dataset, batch_size=best_hparams["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=best_hparams["batch_size"], shuffle=False, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=best_hparams["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    # Training loop
    for epoch in range(best_hparams["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        # Log the metrics to wandb.
        wandb.log({
            "PartA_Q4_epoch": epoch + 1,
            "PartA_Q4_train_loss": train_loss,
            "PartA_Q4_train_accuracy": train_acc,
            "PartA_Q4_val_loss": val_loss,
            "PartA_Q4_val_accuracy": val_acc
        })
        print(f"[Epoch {epoch + 1}/{best_hparams['epochs']}] "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > 0.42:  # Early stopping condition
            print(f"Validation accuracy: {val_acc:.4f} reached 0.42. Performing early stopping to avoid overfitting.")
            break

    print("Training complete.")

    # --------------------------
    # Evaluate on test data.
    # --------------------------
    test_dir = os.path.join(static_config["data_root"], "test")
    test_dataset, class_names = load_inaturalist_test_data(test_dir, best_hparams["resize_dim"])
    test_loader = DataLoader(test_dataset, batch_size=best_hparams["batch_size"], shuffle=False, num_workers=4)
    test_accuracy = evaluate_model_on_test_data(model, test_loader, device)
    wandb.log({"PartA_Q4_test_accuracy": test_accuracy})
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # --------------------------
    # Plot a creative 10x3 grid of predictions on test data and log the plot to wandb.
    # --------------------------
    os.makedirs(static_config['output_dir'], exist_ok=True)
    # Plot predictions and save the figure.
    fig, plot_path = plot_predictions(model, test_dataset, class_names, device, static_config['output_dir'], grid_shape=(10, 3))
    wandb.log({"part_a_q4_test_predictions_grid": wandb.Image(plot_path)})

    run.finish()


if __name__ == "__main__":
    config = get_configs(project_root, 'configs.yaml')['solution_4_configs']
    train_and_evaluate_best(config)
