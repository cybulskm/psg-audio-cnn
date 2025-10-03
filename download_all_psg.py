import os
import requests

def download_files_from_list(file_list_path="all_files.txt", download_folder="data"):
    # Ensure the download folder exists
    os.makedirs(download_folder, exist_ok=True)

    with open(file_list_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        #Extract filename from URL
        filename=os.path.basename(url)
        if '=' in url:
            filename = url.split('=')[-1]
            dest_path = os.path.join(download_folder, filename)
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(dest_path, "wb") as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        out_file.write(chunk)
                print(f"Downloaded: {filename}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")

def download_edf_files():
    #Download edf files that are assosiated with aready downloaded rml files
    data_bin = "data"
    rml_files = os.listdir(data_bin)
    rml_files = [f for f in rml_files if f.endswith('.rml')]
    rml_files = [f.split('=')[-1].replace('.rml', '') for f in rml_files]
    all_files_txt = "all_files.txt"
    with open(all_files_txt, "r") as f:
        all_files = [line.strip() for line in f if line.strip()]
    edf_files_to_download = []
    for f in all_files:
        if any(rml_id in f for rml_id in rml_files) and f.endswith('.edf'):
            edf_files_to_download.append(f)

    if edf_files_to_download:
        print(f"Downloading {len(edf_files_to_download)} EDF files...")
        for file in edf_files_to_download:
            filename = os.path.basename(file)
            if '=' in file:
                filename = file.split('=')[-1]
            dest_path = os.path.join(data_bin, filename)
            try:
                response = requests.get(file, stream=True)
                response.raise_for_status()
                with open(dest_path, "wb") as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        out_file.write(chunk)
                print(f"Downloaded: {filename}")
            except Exception as e:
                print(f"Failed to download {file}: {e}")
    else:
        print("All EDF files are already downloaded.")

if __name__ == "__main__":
    download_edf_files()