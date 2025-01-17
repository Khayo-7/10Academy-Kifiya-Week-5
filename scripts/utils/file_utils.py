import os
import shutil

def copy_and_rename_files(file_pairs):
    try:
        for old_path, new_path in file_pairs:
            
            new_dir = os.path.abspath(os.path.dirname(new_path))
            os.makedirs(new_dir, exist_ok=True)

            if not os.path.isfile(old_path):
                continue            
            shutil.copy(old_path, new_path)

    except Exception as e:
        print(f"An error occurred: {e}")
