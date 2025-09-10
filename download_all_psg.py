import os
import requests

def download_files_from_list(file_list_path="all_files.txt", download_folder="data"):
    # Ensure the download folder exists
    os.makedirs(download_folder, exist_ok=True)

    with open(file_list_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        filename = os.path.basename(url)
        dest_path = os.path.join(download_folder, filename)
        print(f"Downloading {url} to {dest_path}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(dest_path, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    download_files_from_list()