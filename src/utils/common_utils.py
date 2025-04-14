import os
import zipfile

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