import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchbnn as bnn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import os


class RadiativeTransferBNN(nn.Module):
    def __init__(self, number_of_neurones, dropout_probablity, learning_rate):
        super(RadiativeTransferBNN, self).__init__()

        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda_available else "cpu")

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
                out_features=1
            )
        )

        # loss function
        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean')
        self.kl_weight = 0.01
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.to(device)  # move the model to the GPU if available

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

            data = pd.concat([
                data,
                pd.DataFrame({
                    "log_mstar": log_mstar,
                    "log_mdust_over_mstar": log_mdust_over_mstar,
                    "theta": thetas[i],
                    "n": [n],
                    "flux": [flux],
                    "r": [r]
                    })
                ], ignore_index=True)

        return wvl, data.reset_index(drop=True)

    def compile_dataset(
            self,
            data,
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

        X = [
            file for file in os.listdir(input_filepath)
            if file.startswith("parameters")
            ]

        Y = [
            file for file in os.listdir(output_filepath)
            if file.startswith("data")
            ]

        list_log_mstar, list_log_mdust_over_mstar, list_theta = \
            self.read_input_dict(X)

        list_theta = (list_theta * np.pi) / 180  # convert to radians

        for i in range(len(Y)):
            wavelength, data = self.read_output_file(
                Y[i],
                data,
                np.sin(list_theta),
                list_log_mstar[i],
                list_log_mdust_over_mstar[i]
            )

        return wavelength, data

    def train_model(
            self,
            input_train,
            output_train,
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

        self.train()

        # Create a TensorDataset from input and output tensors
        tensor_dataset = TensorDataset(
            input_train,
            output_train
            )

        # Create a DataLoader for batch training
        data_loader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=True
            )

        for _ in range(epochs):
            for batch_data, batch_labels in data_loader:
                self.optimizer.zero_grad()
                pred = self(batch_data)

                # Calculate cost (MSE + KL)
                mse = self.mse_loss(pred, batch_labels)
                kl = self.kl_loss(self)
                cost = mse + self.kl_weight * kl

                # Backpropagation and optimization
                cost.backward()
                self.optimizer.step()

        print(f"- cost: {cost.item():.3f}")

    def test_model(
            self,
            input_test,
            output_test
            ):
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

        self.eval()

        pred = np.array([
            self(input_test).detach().numpy() for _ in range(500)
            ])
        mean_pred_results = np.mean(pred, axis=0)
        std_pred_results = np.std(pred, axis=0)

        # find the cost of the model
        mse = self.mse_loss(
            torch.Tensor(mean_pred_results),
            torch.Tensor(output_test)
            )
        kl = self.kl_loss(self)
        cost = mse + self.kl_weight * kl

        print(f"- cost: {cost.item():.3f}")
        return mean_pred_results, std_pred_results


# parameter_files = [file for file in os.listdir("../../data/radiative_transfer/input/") if file.startswith("parameters")]
# h5_files = [file for file in os.listdir("../../data/radiative_transfer/output/") if file.startswith("data")]

# wavelength, h5_data = generate_dataset(pd.DataFrame(columns=["log_mstar", "log_mdust_over_mstar", "theta", "n", "flux", "r"]), parameter_files, h5_files)