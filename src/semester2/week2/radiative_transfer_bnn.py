import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchbnn as bnn
from sklearn.model_selection import train_test_split
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
                if len(line5) == 1 and line4[0]!='theta':
                    line5 = line5[0]
                values.append(line5)

        table = dict(zip(keys, values))
        return table
