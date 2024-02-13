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

        is_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean')
        self.kl_weight = 0.01
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)

    def forward(self, x):
        shared = self.shared_layer(x)
        return self.output_layer(shared)
    

    # def initialise_model(self, learning_rate):