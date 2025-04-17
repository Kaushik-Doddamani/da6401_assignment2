import torch
import os
import sys
import torch.nn as nn
import yaml

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root to sys.path if it isn’t already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.implementation import MyCNN
from utils.data_utils import load_single_image
from utils.common_utils import extract_data_if_needed, get_configs


def test_model_with_image(model, image_tensor):
    """
    Passes a single image tensor through the model and prints output shape.
    :param model: Instance of MyCNN
    :param image_tensor: shape (1, 3, H, W)
    """
    with torch.no_grad():
        output = model(image_tensor)
    print(f"Model output shape: {output.shape} (Batch, 10)")
    print("Raw output logits:", output)


def main():
    config = get_configs(project_root, 'configs.yaml')['part_a_configs']['solution_1_configs']

    # Paths (adjust if necessary)
    DATA_ZIP_PATH = config['data_zip_path']
    EXTRACT_DIR = config['extracted_data_dir']

    # 1) Optional: Extract ZIP if needed
    extract_data_if_needed(DATA_ZIP_PATH, EXTRACT_DIR)

    # ------------------------------------------------------------------
    # The ZIP has top-level folder 'inaturalist_12K' inside it.
    # After extraction, we get:
    #   ../inaturalist_data/inaturalist_12K_extracted/
    #       inaturalist_12K/
    #           train/
    #           val/
    #           ...
    #
    # So the image path must include "inaturalist_12K".
    # ------------------------------------------------------------------

    # 2) Pick a single image path from the extracted data
    #    For example, one file from 'train/Plantae'
    sample_image_path = os.path.join(
        EXTRACT_DIR,
        "inaturalist_12K",  # top-level folder from the zip
        "train",
        "Insecta",
        "0a4a6a25d2b409ed0755097ed21fdf5b.jpg"
    )
    if not os.path.isfile(sample_image_path):
        raise FileNotFoundError(f"Sample image not found at {sample_image_path}")

    # 3) Load and transform the image
    image_tensor = load_single_image(sample_image_path, resize=True, resize_dim=(32 * 15, 32 * 15))

    # Inspect the shape to pick your image_height, image_width
    # For example, if the printed shape is [3, 480, 640], do:
    _, c, h, w = image_tensor.shape

    # 4) Create model instance
    #    Example: 16 filters each conv, kernel_size=3, dense of 128
    model = MyCNN(in_channels=3,
                  num_filters=16,
                  kernel_size=3,
                  activation_fn=nn.ReLU,
                  dense_neurons=128,
                  image_height=h,
                  image_width=w)
    print(f"Model created with input shape: (3, {h}, {w})")

    # 5) Test forward pass
    test_model_with_image(model, image_tensor)


if __name__ == "__main__":
    main()
