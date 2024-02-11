import subprocess
import os
import h5py
import numpy as np

# Define the base command and the number of runs
base_command = "./scripts/run.s"
num_runs = 50

# Create directories if they don't exist
for i in range(1, num_runs + 1):
    directory_name = f"run_ref_model{i}"
    os.makedirs(directory_name, exist_ok=True)

# Run each command in a separate screen
for i in range(1, num_runs + 1):
    directory_name = f"run_ref_model{i}"

    # Change directory
    os.chdir(directory_name)

    # Create an "output" folder if it doesn't exist
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # Run the command in a new screen
    screen_command = f"screen -dmS {directory_name} {base_command}"
    subprocess.run(screen_command, shell=True)

    # Create and save an h5 file with an index
    output_file_path = os.path.join(output_folder, f"data{i}.h5")
    with h5py.File(output_file_path, 'w') as h5_file:
        # Example: Saving a dataset with random data
        data = np.random.random((10, 10))
        h5_file.create_dataset("random_data", data=data)

    # Change back to the original directory
    os.chdir("..")
