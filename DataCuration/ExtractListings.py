import os
import gzip
import shutil

def extract_gz_files(folder_path):
    # List all .gz files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.gz'):
            file_path = os.path.join(folder_path, filename)
            
            # Extract the gz file
            with gzip.open(file_path, 'rb') as f_in:
                output_filename = file_path[:-3]  # Remove '.gz' extension
                with open(output_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            print(f"Extracted: {filename}")

# Example usage
folder_path = './listings/metadata'  # Replace this with the path to your folder
extract_gz_files(folder_path)
