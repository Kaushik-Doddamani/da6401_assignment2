import os
import zipfile
import torch
import random
import numpy as np
import torch.nn as nn
import yaml


def extract_data_if_needed(zip_path, extract_dir):
    """
    Extracts the zip file into 'extract_dir' if that folder does not exist.
    """
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        print(f"Extracting {zip_path} to {extract_dir} ...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path=extract_dir)

        print("Extraction done.")
    else:
        print(f"Directory {extract_dir} already exists. Skipping extraction.")


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_activation_fn(activation_name):
    """
    Returns the activation function class based on the provided name.
    """
    act_fn = None
    if activation_name.lower() == "mish":
        act_fn = nn.Mish
    elif activation_name.lower() == "relu":
        act_fn = nn.ReLU
    elif activation_name.lower() == "gelu":
        act_fn = nn.GELU
    elif activation_name.lower() == "silu":
        act_fn = nn.SiLU
    else:
        act_fn = nn.ReLU
    return act_fn


def get_configs(project_root, config_filename):
    with open(os.path.join(project_root, "config", config_filename), 'r') as f:
        config = yaml.safe_load(f)
    return config
