import os
import pandas as pd
from shutil import copy2


metadata_path = "./cv-corpus-19.0-delta-2024-09-13/en/validated.tsv"  
clips_path = "./cv-corpus-19.0-delta-2024-09-13/en/clips/"  
output_dir = "./split_dataset/" 
os.makedirs(os.path.join(output_dir, "male"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "female"), exist_ok=True)


metadata = pd.read_csv(metadata_path, sep="\t")

metadata = metadata.dropna(subset=['gender'])


metadata['gender'] = metadata['gender'].replace({
    'male_masculine': 'male',
    'female_feminine': 'female'
})


valid_genders = ['male', 'female']
metadata = metadata[metadata['gender'].isin(valid_genders)]

male_files = metadata[metadata['gender'] == 'male']['path'].tolist()
female_files = metadata[metadata['gender'] == 'female']['path'].tolist()


def copy_files(file_list, target_dir):
    for file_name in file_list:
        source_path = os.path.join(clips_path, file_name)
        target_path = os.path.join(target_dir, file_name)
        try:
            copy2(source_path, target_path)
            print(f"Copied: {file_name} -> {target_dir}")
        except Exception as e:
            print(f"Error copying {file_name}: {e}")


copy_files(male_files, os.path.join(output_dir, "male"))
copy_files(female_files, os.path.join(output_dir, "female"))

print("Dataset splitting complete!")
