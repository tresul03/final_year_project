from pdb import run
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchbnn as bnn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import os
from sklearn.preprocessing import StandardScaler
import time


class RadiativeTransferBNN(nn.Module):
    def __init__(
            self,
            number_of_neurones: int,
            dropout_probablity: float,
            learning_rate: float,
            output_choice: str
            ):
        super(RadiativeTransferBNN, self).__init__()

        self.shared_layer = nn.Sequential(
            bnn.BayesLinear(  # input layer
                prior_mu=0,
                prior_sigma=0.1,
                in_features=3,
                out_features=number_of_neurones
            ),
            nn.ReLU(),
            nn.Dropout(dropout_probablity),

            bnn.BayesLinear(  # 1st hidden layer
                prior_mu=0,
                prior_sigma=0.1,
                in_features=number_of_neurones,
                out_features=number_of_neurones
            ),
            nn.ReLU(),
            nn.Dropout(dropout_probablity),

            bnn.BayesLinear(  # 2nd hidden layer
                prior_mu=0,
                prior_sigma=0.1,
                in_features=number_of_neurones,
                out_features=number_of_neurones
            ),
            nn.ReLU(),
            nn.Dropout(dropout_probablity),
        )

        self.output_layer = nn.Sequential(
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=number_of_neurones,
                out_features=113
            )
        )

        cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda_available else "cpu")

        self.normalise = lambda x: (x - np.mean(x)) / np.std(x)
        self.denormalise = lambda x, mean, std: x * std + mean

        self.output_choice = output_choice
        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean')
        self.kl_weight = 0.01
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.to(self.device)  # move the model to the GPU if available
        self.df = pd.DataFrame(
            columns=[
                "log_mstar",
                "log_mdust_over_mstar",
                "theta",
                "n",
                "flux",
                "r"
                ]
            )

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def forward(self, x):
        shared = self.shared_layer(x)
        return self.output_layer(shared)

    def read_input_file(
            self,
            filename: str,
            filepath: str = '../../data/radiative_transfer/input/'
            ):
        """
        Read the parameters from the input file.

        Parameters:
        - filename (str): Name of the input file.
        - filepath (str): Path to the input file.
        Default is '../../data/radiative_transfer/input/'.

        Returns:
        - table (dict): Dictionary containing the parameters.
        """

        lines = open(filepath+filename, 'r').readlines()

        keys = []
        values = []
        for i in range(len(lines)):

            line_i = lines[i]
            line1 = line_i.split('\n')[0]
            line2 = line1.split('#')[0]
            line3 = line2.split('=')
            line4 = []
            for j in range(len(line3)):
                line4.append(line3[j].strip(' '))

            if len(line4) == 2:
                keys.append(line4[0])
                line5 = line4[1].split(', ')
                line5 = np.array(line5).astype(float)
                if len(line5) == 1 and line4[0] != 'theta':
                    line5 = line5[0]
                values.append(line5)

        table = dict(zip(keys, values))
        return table

    def read_input_dict(
            self,
            filenames: list,
            filepath: str = "../../../data/radiative_transfer/input/"
            ):
        """
        Read the parameter files.

        Parameters:
        - filenames (list): List of filenames.
        - filepath (str): Path to the parameter files.
        Default is '../../data/radiative_transfer/input/'.

        Returns:
        - list_log_mstar (np.array): Array of log of stellar mass.
        - list_log_mdust_over_mstar (np.array): Array of log of dust mass over
        stellar mass.
        - list_theta (np.array): Array of viewing angles.
        """

        list_log_mstar = np.array([])
        list_log_mdust = np.array([])
        list_theta = np.array([])

        for filename in filenames:
            table = self.read_input_file(filename, filepath)
            list_log_mstar = np.append(list_log_mstar, table['logMstar'])
            list_log_mdust = np.append(list_log_mdust, table['logMdust'])
            list_theta = np.append(list_theta, table['theta'])

        list_log_mdust_over_mstar = list_log_mdust - list_log_mstar

        return list_log_mstar, list_log_mdust_over_mstar, list_theta

    def read_output_file(
            self,
            filename: str,
            data,
            thetas,
            log_mstar,
            log_mdust_over_mstar,
            filepath: str = '../../../data/radiative_transfer/output/'
            ):
        """
        Read the output file.

        Parameters:
        - filename (str): Name of the output file.
        - data (pd.DataFrame): DataFrame containing the data.
        - thetas (list): List of viewing angles.
        - log_mstar (float): log of stellar mass.
        - log_mdust_over_mstar (float): log of dust mass over stellar mass.
        - filepath (str): Path to the output file.
        Default is '../../data/radiative_transfer/output/'.

        Returns:
        - wvl (np.array): Rest-frame wavelength [micron].
        - data (pd.DataFrame): DataFrame containing the data.
        """

        filepath += filename

        # Finding hdf keys
        hdf_keys = np.array([])
        with pd.HDFStore(filepath, 'r') as hdf:
            hdf_keys = np.append(hdf_keys, hdf.keys())

        for i in range(len(hdf_keys)):
            table = pd.read_hdf(filepath, hdf_keys[i])

            # obtain wavelength, flux, half-light radius, and Sersic index
            wvl = table['wvl'].to_numpy(dtype=np.float64)  # [micron]
            flux = table['flux'].to_numpy(dtype=np.float64)  # [Wm^-2]
            r = table['r'].to_numpy(dtype=np.float64)  # [kpc]
            n = table['n'].to_numpy(dtype=np.float64)

            self.df = pd.concat([
                self.df,
                pd.DataFrame({
                    "log_mstar": log_mstar,
                    "log_mdust_over_mstar": log_mdust_over_mstar,
                    "theta": thetas[i],
                    "n": [n],
                    "flux": [flux],
                    "r": [r]
                    })
                ], ignore_index=True)

        return wvl, self.df.reset_index(drop=True)

    def compile_dataset(
            self
            ):
        """
        Generate the dataset.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the data.
        - params (list): List of parameter files.
        - files (list): List of output files.

        Returns:
        - wavelength (np.array): Rest-frame wavelength [micron].
        - data (pd.DataFrame): DataFrame containing the data.
        """

        input_filepath = "../../../data/radiative_transfer/input/"
        output_filepath = "../../../data/radiative_transfer/output/"

        input_files = [
            file for file in os.listdir(input_filepath)
            if file.startswith("parameters")
            ]

        output_files = [
            file for file in os.listdir(output_filepath)
            if file.startswith("data")
            ]

        list_log_mstar, list_log_mdust_over_mstar, list_theta = \
            self.read_input_dict(input_files)
        list_theta = (list_theta * np.pi) / 180  # convert to radians

        for i in range(len(output_files)):
            wavelength, data = self.read_output_file(
                output_files[i],
                self.df,
                np.sin(list_theta),
                list_log_mstar[i],
                list_log_mdust_over_mstar[i]
            )

        return wavelength, data

    def preprocess_data(self):
        scaler = StandardScaler()
        X = self.df[["log_mstar", "log_mdust_over_mstar", "theta"]].copy()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        X["run_id"] = X.groupby([
            "log_mstar",
            "log_mdust_over_mstar"
            ]).ngroup()

        y = self.df[["n", "flux", "r"]].copy()
        y["run_id"] = X["run_id"]
        y = y[[self.output_choice, "run_id"]]

        run_ids = X["run_id"].unique()
        train_runs, test_runs = train_test_split(
            run_ids,
            test_size=0.2,
            random_state=42
            )

        self.X_train = X[X["run_id"].isin(train_runs)]\
            .drop(columns="run_id").reset_index(drop=True)
        self.X_test = X[X["run_id"].isin(test_runs)]\
            .drop(columns="run_id").reset_index(drop=True)
        self.y_train = y[y["run_id"].isin(train_runs)]\
            .drop(columns="run_id").reset_index(drop=True)
        self.y_test = y[y["run_id"].isin(test_runs)]\
            .drop(columns="run_id").reset_index(drop=True)

        self.X_train = self.convert_to_tensor(self.X_train)
        self.X_test = self.convert_to_tensor(self.X_test)
        self.y_train = self.convert_to_tensor(self.y_train)
        self.y_test = self.convert_to_tensor(self.y_test)

    def convert_to_tensor(self, data):
        """
        Convert the DataFrame to a tensor.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the data.

        Returns:
        - data (tensor): Tensor containing the data.
        """

        data = data.map(np.array)
        stacked_input_arrays = np.stack(
            data.apply(lambda row: np.stack(row, axis=0), axis=1).to_numpy()
            )
        data = torch.Tensor(stacked_input_arrays)

        return data

    def train_model(
            self,
            epochs: int,
            batch_size: int
            ):
        """
        Train the model using batch training.

        Parameters:
        - model_attributes (tuple): Tuple containing the initialized model,
        MSE loss function, KL loss function, KL weight, and optimizer.
        - input_train (tensor): Input tensor for training.
        - output_train (tensor): Output tensor for training.
        - epochs (int): Number of epochs to train the model.
        - batch_size (int): Batch size for training.

        Returns:
        - The trained model.
        """

        print("Training the model...")
        t0 = time.time()
        self.train()

        # Create a TensorDataset from input and output tensors
        tensor_dataset = TensorDataset(
            self.X_train,
            self.y_train
            )

        # Create a DataLoader for batch training
        data_loader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=True
            )

        for epoch in range(epochs):
            for batch_data, batch_labels in data_loader:
                self.optimizer.zero_grad()
                pred = self(batch_data)

                # Calculate cost (MSE + KL)
                mse = self.mse_loss(pred, batch_labels[:, 0, :])
                kl = self.kl_loss(self)
                cost = mse + self.kl_weight * kl

                # Backpropagation and optimization
                cost.backward()
                self.optimizer.step()

            print(f"- epoch {epoch+1}/{epochs} - cost: {cost.item():.3f}")
        print(f"- this took {time.time() - t0:.2f} seconds")

    def test_model(self):
        """
        Test the model.

        Parameters:
        - model (nn.Module): Trained model.
        - input_test (tensor): Input tensor for testing.
        - output_test (tensor): Output tensor for testing.

        Returns:
        - mean_pred_results (np.array): Mean predicted results.
        - std_pred_results (np.array): Standard deviation of predicted results.
        """

        print("Testing the model...")
        t0 = time.time()
        self.eval()

        pred = np.array([
            self(self.X_test).detach().numpy() for _ in range(500)
            ])
        mean_pred_results = np.mean(pred, axis=0)
        std_pred_results = np.std(pred, axis=0)

        # find the cost of the model
        mse = self.mse_loss(
            torch.Tensor(mean_pred_results),
            torch.Tensor(self.y_test[:, 0, :])
            )
        kl = self.kl_loss(self)
        cost = mse + self.kl_weight * kl

        print(f"- cost: {cost.item():.3f}")
        print(f"- this took {time.time() - t0:.2f} seconds")
        return mean_pred_results, std_pred_results


model = RadiativeTransferBNN(1000, 0.3, 0.01, "n")
model.compile_dataset()
model.preprocess_data()
# model.X_test.to_csv("X_test.csv")
# model.y_test.to_csv("y_test.csv")
model.train_model(10, 20)
mean_pred_results, std_pred_results = model.test_model()
