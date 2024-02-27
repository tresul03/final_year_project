import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchbnn as bnn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import os
import time


class RadiativeTransferBNN(nn.Module):
    """
    Bayesian Neural Network for Radiative Transfer.

    Parameters:
    - number_of_neurones (int): Number of neurones in the hidden layers.
    - dropout_probablity (float): Dropout probability.
    - learning_rate (float): Learning rate.
    - output_choice (str): Choice of output.
    """
    def __init__(
            self,
            number_of_neurones: int,
            dropout_probablity: float,
            learning_rate: float,
            output_choice: str
            ):
        super(RadiativeTransferBNN, self).__init__()

        self.number_of_neurones = number_of_neurones
        self.dropout_probablity = dropout_probablity
        self.learning_rate = learning_rate
        self.output_choice = output_choice

        self.shared_layer = nn.Sequential(
            bnn.BayesLinear(  # input layer
                prior_mu=0,
                prior_sigma=0.1,
                in_features=3,
                out_features=self.number_of_neurones
            ),
            nn.BatchNorm1d(self.number_of_neurones),
            nn.ReLU(),
            nn.Dropout(self.dropout_probablity),

            bnn.BayesLinear(  # 1st hidden layer
                prior_mu=0,
                prior_sigma=0.1,
                in_features=self.number_of_neurones,
                out_features=self.number_of_neurones
            ),
            nn.BatchNorm1d(self.number_of_neurones),
            nn.ReLU(),
            nn.Dropout(self.dropout_probablity),

            bnn.BayesLinear(  # 2nd hidden layer
                prior_mu=0,
                prior_sigma=0.1,
                in_features=self.number_of_neurones,
                out_features=self.number_of_neurones
            ),
            nn.BatchNorm1d(self.number_of_neurones),
            nn.ReLU(),
            nn.Dropout(self.dropout_probablity),
        )

        self.output_layer = nn.Sequential(
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=self.number_of_neurones,
                out_features=113
            )
        )

        cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda_available else "cpu")

        self.normalise = lambda x: (x - np.mean(x)) / np.std(x)
        self.denormalise = lambda x, mean, std: x * std + mean

        self.mse_loss = nn.MSELoss().to(self.device)
        self.kl_loss = bnn.BKLLoss(reduction='mean').to(self.device)
        self.kl_weight = 0.1
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=250,
                    gamma=0.1
                    )

        self.X_train = torch.Tensor().to(self.device)
        self.X_test = torch.Tensor().to(self.device)
        self.y_train = torch.Tensor().to(self.device)
        self.y_test = torch.Tensor().to(self.device)
        self.wavelength = np.array([])

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

        self.to(self.device)  # move the model to the GPU if available

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x (tensor): Input tensor.

        Returns:
        - output_layer (tensor): Output tensor.
        """
        shared = self.shared_layer(x)
        return self.output_layer(shared)

    def read_input_file(
            self,
            filename: str,
            filepath: str = '../../data/radiative_transfer/input/'
            ):
        """
        Reads the input file and extracts parameters.

        This method reads a file from the provided path and filename, and
        parses it to extract parameters.
        Each line in the file should contain a key-value pair, separated by an
        equals sign (=).
        The key is the parameter name, and the value is the parameter value.
        If the value contains multiple items separated by commas, it is
        converted into a numpy array.
        If the value is a single item, it is converted to a float.

        Parameters:
        - filename (str): The name of the input file.
        - filepath (str, optional): The path to the directory containing the
        input file.
        Defaults to '../../data/radiative_transfer/input/'.

        Returns:
        - dict: A dictionary where the keys are parameter names and the values
        are parameter values.
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
        Reads multiple parameter files and extracts parameters.

        This method reads a list of files from the provided path, and parses
        each file to extract parameters.
        Each file should contain key-value pairs, separated by an equals sign
        (=).
        The key is the parameter name, and the value is the parameter value.
        If the value contains multiple items separated by commas, it is
        converted into a numpy array.
        If the value is a single item, it is converted to a float.
        The parameters extracted from each file are appended to corresponding
        numpy arrays.

        Parameters:
        - filenames (list): The list of names of the input files.
        - filepath (str, optional): The path to the directory containing the
        input files.
        Defaults to '../../../data/radiative_transfer/input/'.

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
        Reads the output file and extracts parameters.

        This method reads an output file from the provided path and filename,
        and parses it to extract parameters.
        Each file should contain key-value pairs, separated by an equals sign
        (=).
        The key is the parameter name, and the value is the parameter value.
        If the value contains multiple items separated by commas, it is
        converted into a numpy array.
        If the value is a single item, it is converted to a float.
        The parameters extracted from the file are appended to a DataFrame.

        Parameters:
        - filename (str): The name of the output file.
        - data (pd.DataFrame): DataFrame containing the data.
        - thetas (list): List of viewing angles.
        - log_mstar (float): log of stellar mass.
        - log_mdust_over_mstar (float): log of dust mass over stellar mass.
        - filepath (str, optional): The path to the directory containing the
        output file.
        Defaults to '../../../data/radiative_transfer/output/'.

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

            self.wavelength = wvl
            flux = np.log10(flux)
            r = np.log10(r)
            n = np.log10(n)

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
                ], ignore_index=True).reset_index(drop=True)

        return wvl, self.df.reset_index(drop=True)

    def compile_dataset(
            self
            ):
        """
        Compiles the dataset from multiple input and output files.

        This method reads a list of input files and a list of output files
        from specified directories.
        Each input file contains parameters which are extracted and stored.
        Each output file contains data which is read and appended to a
        DataFrame.
        The method returns the final compiled dataset and the rest-frame
        wavelength.

        Parameters:
        - None

        Returns:
        - wavelength (np.array): Rest-frame wavelength [micron].
        - data (pd.DataFrame): DataFrame containing the compiled data from all
        output files.
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
            wavelength, self.df = self.read_output_file(
                output_files[i],
                self.df,
                np.sin(list_theta),
                list_log_mstar[i],
                list_log_mdust_over_mstar[i]
            )

        return wavelength

    def preprocess_data(self):
        """
        Preprocesses the dataset for training and testing.

        This method scales the input features and separates the dataset into
        training and testing subsets.
        It also assigns a unique run_id to each unique combination of
        "log_mstar" and "log_mdust_over_mstar".
        The method then splits the dataset into training and testing subsets
        based on these run_ids, ensuring that
        all data from a single run is either in the training set or the
        testing set, but not both.
        Finally, it converts the training and testing subsets into PyTorch
        tensors.

        Parameters:
        - None

        Returns:
        - None, but updates the following instance variables:
            - self.X_train: Training subset of the input features.
            - self.X_test: Testing subset of the input features.
            - self.y_train: Training subset of the output features.
            - self.y_test: Testing subset of the output features.
        """

        self.compile_dataset()

        self.df["run_id"] = self.df.groupby([
            "log_mstar",
            "log_mdust_over_mstar"
            ]).ngroup()
        self.df["angle_id"] = self.df.index % 10

        X = self.df[[
            "log_mstar",
            "log_mdust_over_mstar",
            "theta",
            "run_id",
            "angle_id"
            ]].copy()

        X["log_mstar"] = self.normalise(X["log_mstar"])
        X["log_mdust_over_mstar"] = self.normalise(X["log_mdust_over_mstar"])
        X["theta"] = self.normalise(X["theta"])

        y = self.df[[self.output_choice, "run_id", "angle_id"]].copy()
        y_output_matrix = np.array(y[self.output_choice].to_list())

        for i in range(len(y_output_matrix)):
            y_output_matrix[i] = self.normalise(y_output_matrix[i])
            y.at[i, self.output_choice] = y_output_matrix[i]

        run_ids = X["run_id"].unique()
        train_runs, test_runs = train_test_split(
            run_ids,
            test_size=0.2,
            random_state=42
            )

        self.X_train = X[X["run_id"].isin(train_runs)]
        self.X_test = X[X["run_id"].isin(test_runs)]
        self.y_train = y[y["run_id"].isin(train_runs)]
        self.y_test = y[y["run_id"].isin(test_runs)]

        # order the datasets by both run_id and theta, then drop run_id
        self.X_train = self.X_train.sort_values(by=["run_id", "angle_id"])\
            .drop(columns=["run_id", "angle_id"]).reset_index(drop=True)
        self.X_test = self.X_test.sort_values(by=["run_id", "angle_id"])\
            .drop(columns=["run_id", "angle_id"]).reset_index(drop=True)
        self.y_train = self.y_train.sort_values(by=["run_id", "angle_id"])\
            .drop(columns=["run_id", "angle_id"]).reset_index(drop=True)
        self.y_test = self.y_test.sort_values(by=["run_id", "angle_id"])\
            .drop(columns=["run_id", "angle_id"]).reset_index(drop=True)

        self.X_train = self.convert_to_tensor(self.X_train)
        self.X_test = self.convert_to_tensor(self.X_test)
        self.y_train = self.convert_to_tensor(self.y_train)[:, 0, :]
        self.y_test = self.convert_to_tensor(self.y_test)[:, 0, :]

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
        Trains the Bayesian Neural Network model using batch training.

        This method trains the model using the Mean Squared Error (MSE) loss
        and the Kullback-Leibler (KL) divergence loss.
        It creates a DataLoader for batch training, and for each epoch, it
        iterates over the batches of data,
        performs forward propagation, calculates the cost (MSE + KL), performs
        backpropagation, and updates the model parameters.
        It prints the cost at the end of each epoch and the total training
        time at the end.

        Parameters:
        - epochs (int): The number of epochs to train the model.
        - batch_size (int): The batch size for training.

        Returns:
        - None, but updates the model parameters.
        """

        print("Training the model...")
        t0 = time.time()
        self.train()

        # Create a TensorDataset from input and output tensors
        tensor_dataset = TensorDataset(
            torch.Tensor(self.X_train),
            torch.Tensor(self.y_train)
            )

        for epoch in range(epochs):
            data_loader = DataLoader(
                tensor_dataset,
                batch_size=batch_size,
                shuffle=True
                )

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

            print(f"- epoch {epoch+1}/{epochs} - cost: {cost.item():.3f}, kl: \
                {kl.item():.3f}"
                )
        self.scheduler.step()
        print(f"- this took {time.time() - t0:.2f} seconds")

    def test_model(self):
        """
        Tests the Bayesian Neural Network model.

        This method sets the model to evaluation mode and performs forward
        propagation on the testing data.
        It generates predictions for the testing data multiple times and
        calculates the mean and standard deviation of these predictions.
        It then calculates the cost of the model using the Mean Squared Error
        (MSE) loss and the Kullback-Leibler (KL) divergence loss.
        Finally, it prints the cost and the time taken to test the model.

        Parameters:
        - None, but uses the following instance variables:
            - self.X_test: Testing subset of the input features.
            - self.y_test: Testing subset of the output features.

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
            torch.Tensor(self.y_test)
            )
        kl = self.kl_loss(self)
        cost = mse + self.kl_weight * kl

        print(f"- cost: {cost.item():.3f}")
        print(f"- this took {time.time() - t0:.2f} seconds")
        return mean_pred_results, std_pred_results
