import os
import requests
import zipfile
from io import BytesIO

def download_and_extract():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    dest_dir = "data"
    
    print(f"Downloading dataset from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        print("Download complete. Extracting zip archive...")
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            # The zip file contains a subfolder like 'ml-latest-small/'
            # We want to extract files to 'data/' directly or keep the subfolder structure
            # Let's extract everything to 'data' directory
            zip_ref.extractall(dest_dir)
        print(f"Extraction complete. Data stored in '{dest_dir}' directory.")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

if __name__ == "__main__":
    download_and_extract()
