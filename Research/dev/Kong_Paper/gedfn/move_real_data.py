import os
import shutil

# Source and destination directories
source_dir = "Kong_Paper/data/patient_data_downloaded"  # Source directory where the folders are located
destination_dir = "Kong_Paper/data/patient_genes"  # Destination directory where folders will be copied

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Loop through each item in the source directory
for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)

    # Check if the item is a directory (folder)
    if os.path.isdir(folder_path):
        # Loop through each file in the subfolder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Check if it is a file (not a subdirectory)
                if os.path.isfile(file_path):
                    # Construct the destination file path
                    destination_path = os.path.join(destination_dir, file_name)
                    # Copy the file to the destination
                    shutil.copy(file_path, destination_path)
                    print(f"Copied file: {file_name} from {folder}")
            except Exception as e:
                print(f"Error copying {file_name} from {folder}: {e}")

print("All files copied successfully.")


    
