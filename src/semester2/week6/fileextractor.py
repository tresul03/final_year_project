import os
import shutil

# Path to the directory containing your input folders
input_directory = r"../../../data/rad_transfer_16dim/inputs"

# Path to the directory where you want to store the parameter files
output_directory = r"../../../data/rad_transfer_16dim/numbered_inputs"



# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

file_path = "../../../data/rad_transfer_16dim/numbered_inputs/README.txt"
with open(file_path, "w") as file:
    file.write("This file contains the numbered parameter files for the input folders. \n\n")

# Iterate over the input folders
for index_folder in os.listdir(input_directory):
    index_folder_path = os.path.join(input_directory, index_folder)

    if index_folder.startswith("input_"):
        # Extract index from folder name
        index = index_folder.split("_")[1]
    
    # Check if it's a directory
    if os.path.isdir(index_folder_path):
        # Assuming there's only one parameter file per folder
        parameter_file = os.path.join(index_folder_path, 'parameters.txt')
        
        # Check if the parameter file exists
        if os.path.exists(parameter_file):
            # Construct the new filename using the index of the folder
            new_filename = f'parameters{index}.txt'
            output_file_path = os.path.join(output_directory, new_filename)
            
            # Copy the parameter file to the output directory and rename it
            shutil.copy(parameter_file, output_file_path)
            print(f'Copied and renamed {parameter_file} to {output_file_path}')


# Path to the directory containing your input folders
input_directory = r"../../../data/rad_transfer_16dim/outputs"

# Path to the directory where you want to store the parameter files
output_directory = r"../../../data/rad_transfer_16dim/numbered_outputs"



# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

file_path = "../../../data/rad_transfer_16dim/numbered_inputs/README.txt"
with open(file_path, "w") as file:
    file.write("This file contains the numbered h5 files for the input folders. \n\n")

# Iterate over the input folders
for index_folder in os.listdir(input_directory):
    index_folder_path = os.path.join(input_directory, index_folder)

    if index_folder.startswith("output_"):
        # Extract index from folder name
        index = index_folder.split("_")[1]
    
    # Check if it's a directory
    if os.path.isdir(index_folder_path):
        # Assuming there's only one parameter file per folder
        parameter_file = os.path.join(index_folder_path, 'data.h5')
        
        # Check if the parameter file exists
        if os.path.exists(parameter_file):
            # Construct the new filename using the index of the folder
            new_filename = f'data{index}.h5'
            output_file_path = os.path.join(output_directory, new_filename)
            
            # Copy the parameter file to the output directory and rename it
            shutil.copy(parameter_file, output_file_path)
            print(f'Copied and renamed {parameter_file} to {output_file_path}')

