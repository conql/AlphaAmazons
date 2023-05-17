import os
import requests
from pathlib import Path
import zipfile


# Function to download the file
def download_file(url, local_filename):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


# Function to extract the zip file
def extract_zip(local_filename, destination_folder):
    with zipfile.ZipFile(local_filename, "r") as zip_ref:
        zip_ref.extractall(destination_folder)


def download_all_data_zip(destination_folder="data/raw"):
    # Base URL
    base_url = "https://extra.botzone.org.cn/matchpacks/Amazons-"

    Path(destination_folder).mkdir(parents=True, exist_ok=True)

    # Iterate through the years and months
    for year in range(2021, 2024):
        for month in range(1, 13):
            # Stop at 2023-05
            if year == 2023 and month > 5:
                break

            # Construct the URL and local file name
            url = f"{base_url}{year}-{month}.zip"
            local_filename = os.path.join(
                destination_folder, f"Amazons-{year}-{month}.zip"
            )

            # Download the file
            try:
                print(f"Downloading {url}...")
                download_file(url, local_filename)
                print(f"Successfully downloaded {local_filename}.")
                print(f"Extracting {url}...")
                extract_zip(
                    local_filename,
                    os.path.join(destination_folder, f"extracted/{year}-{month}"),
                )
            except requests.exceptions.HTTPError as e:
                print(f"Error downloading {url}: {e}")


download_all_data_zip()
