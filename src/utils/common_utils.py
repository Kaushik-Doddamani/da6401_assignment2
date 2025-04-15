import os
import zipfile
import torch
import random
import numpy as np

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