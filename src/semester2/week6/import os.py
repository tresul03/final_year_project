import os
import shutil

# Path to the directory containing your input folders
input_directory = r"C:\Users\joshu\OneDrive\Desktop\Physics_Year_3\Final_year_project\Github\project_script\data\rad_transfer_16dim\inputs"

# Path to the directory where you want to store the parameter files
output_directory = r'C:\Users\joshu\OneDrive\Desktop\Physics_Year_3\Final_year_project\Github\project_script\data\rad_transfer_16dim\INPUTS'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over the input folders
for index_folder in os.listdir(input_directory):
    index_folder_path = os.path.join(input_directory, index_folder)
    
    # Check if it's a directory
    if os.path.isdir(index_folder_path):
        # Assuming there's only one parameter file per folder
        parameter_file = os.path.join(index_folder_path, 'parameter_file.txt')
        
        # Check if the parameter file exists
        if os.path.exists(parameter_file):
            # Construct the new filename using the index of the folder
            new_filename = f'{index_folder}_parameter_file.txt'
            output_file_path = os.path.join(output_directory, new_filename)
            
            # Copy the parameter file to the output directory and rename it
            shutil.copy(parameter_file, output_file_path)
            print(f'Copied and renamed {parameter_file} to {output_file_path}')
