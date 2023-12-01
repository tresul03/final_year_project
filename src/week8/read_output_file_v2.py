#!/usr/bin/env python
# coding: utf-8

# So my approach was completely wrong...
# 
# I'll try again. You'll see how differently I handle data preprocessing here.

# In[13]:


import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

#allocating datasets and model to GPU for speed's sake
is_available = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[14]:


class ResulDualOutputBNN(nn.Module):
    def __init__(self, no_of_neurones, dropout_prob):
        super(ResulDualOutputBNN, self).__init__()
        self.shared_layer = nn.Sequential( #this is the input layer
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=3, out_features=no_of_neurones),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.output_layer_y0 = nn.Sequential( #this is the output layer for y0
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=no_of_neurones, out_features=no_of_neurones),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=no_of_neurones, out_features=4)
        )
        self.output_layer_y1 = nn.Sequential( #this is the output layer for y1
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=no_of_neurones, out_features=no_of_neurones),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=no_of_neurones, out_features=4)
        )
        self.output_layer_y2 = nn.Sequential( #this is the output layer for y2
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=no_of_neurones, out_features=no_of_neurones),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=no_of_neurones, out_features=3)
        )


    def forward(self, x): #this is the forward pass, run automatically when you call the model
        shared = self.shared_layer(x)
        y0 = self.output_layer_y0(shared)
        y1 = self.output_layer_y1(shared)
        y2 = self.output_layer_y2(shared)
        return y0, y1, y2


# In[15]:


def read_params(filename: str, filepath: str = '../../data/radiative_transfer/input/'):
    """
    Reads parameters from a specified file and returns them as a dictionary.

    The function reads a text file where each line contains a parameter in the format:
    `key = value # optional comment`. The function parses these lines to extract the keys 
    and values, ignoring any text following a '#' as a comment.

    Parameters:
    - file (str, optional): The path to the file containing the parameters. 
    Default is '../../data/radiative_transfer/input/parameters.txt'.

    Returns:
    - dict: A dictionary where each key-value pair corresponds to a parameter 
    and its respective value. If a line contains a comma-separated list of values, 
    they are converted to a NumPy array. If the value is a single number (except for 
    the 'theta' parameter), it is converted to a float.

    Note:
    - This function assumes that each parameter is defined only once in the file.
    - The function is designed to handle special cases where the value is a list 
    (converted to a NumPy array) or a single float. The exception is the 'theta' 
    parameter, which is always treated as a NumPy array.
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
            line4.append( line3[j].strip(' ') )

        if len(line4) == 2:
            keys.append(line4[0])
            line5 = line4[1].split(', ')
            line5 = np.array(line5).astype(float)
            if len(line5) == 1 and line4[0]!='theta':
                line5 = line5[0]
            values.append(line5)

    table = dict(zip(keys, values) )
    return table


# In[16]:


def read_h5_file(filename: str, df, thetas, log_mstar, log_mdust_over_mstar, filepath: str = '../../data/radiative_transfer/output/'):
    """
    Reads HDF5 files and compiles data into a single DataFrame with additional parameters.

    Parameters:
    - filename (str): The name of the HDF5 file to be read.
    - thetas (array-like): An array of viewing angles corresponding to each entry in the HDF5 file.
    - log_mstar (float): Logarithmic value of stellar mass.
    - log_mdust_over_mstar (float): Logarithmic value of the dust mass over stellar mass ratio.
    - filepath (str, optional): Path to the directory containing the HDF5 file. Defaults to '../../data/radiative_transfer/output/'.

    Returns:
    - pd.DataFrame: A DataFrame containing wavelength, flux, half-light radius, Sersic index, viewing angle, logarithm of stellar mass, and logarithm of dust mass over stellar mass ratio.

    This function iterates over keys in the HDF5 file, extracts relevant data, and compiles it into a comprehensive DataFrame, adding constant parameters for stellar mass and dust mass ratios.
    """

    filepath += filename 
    print(filepath)

    # Finding hdf keys
    hdf_keys = np.array([])
    with pd.HDFStore(filepath, 'r') as hdf:
        hdf_keys = np.append(hdf_keys, hdf.keys())

    for i in range(len(hdf_keys)):

        table = pd.read_hdf(filepath, hdf_keys[i]) # Face-on view
        wvl = table['wvl'].to_numpy() # rest-frame wavelength [micron]
        flux = table['flux'].to_numpy() # flux [W/m^2]
        r = table['r'].to_numpy() # half-light radius [kpc]
        n = table['n'].to_numpy() # Sersic index

        df = pd.concat([df, pd.DataFrame({"log_mstar": log_mstar, "log_mdust_over_mstar": log_mdust_over_mstar, "theta": thetas[i], "n":[n], "flux":[flux], "r":[r]})], ignore_index=True)

    return df.reset_index(drop=True)


# In[17]:


def read_parameter_files(filenames: list, filepath: str = "../../data/radiative_transfer/input/"):
    """
    Reads multiple parameter files and extracts key information.

    Parameters:
    - filenames (list): A list of filenames for the parameter files to be read.
    - filepath (str, optional): Path to the directory containing the parameter files. Defaults to "../../data/radiative_transfer/input/".

    Returns:
    - tuple: A tuple containing three arrays - list_log_mstar, list_log_mdust_over_mstar, and list_theta. 
        - list_log_mstar (numpy.ndarray): Array of logarithmic stellar mass values.
        - list_log_mdust_over_mstar (numpy.ndarray): Array of logarithmic dust mass over stellar mass ratio values.
        - list_theta (numpy.ndarray): Array of viewing angles.

    The function iterates over each file, reads its parameters, and compiles key data into arrays for further processing.
    """

    list_log_mstar = np.array([])
    list_log_mdust = np.array([])
    list_theta = np.array([])

    for filename in filenames:
        table = read_params(filename, filepath)
        list_log_mstar = np.append(list_log_mstar, table['logMstar'])
        list_log_mdust = np.append(list_log_mdust, table['logMdust'])
        list_theta = np.append(list_theta, table['theta'])

    list_log_mdust_over_mstar = list_log_mdust - list_log_mstar

    return list_log_mstar, list_log_mdust_over_mstar, list_theta


# In[18]:


def generate_dataset(df, params, files):
    list_log_mstar, list_log_mdust_over_mstar, list_theta = read_parameter_files(params)

    for i in range(len(files)):
        df = read_h5_file(files[i], df, list_theta, list_log_mstar[i], list_log_mdust_over_mstar[i])

    return df


# In[19]:


#obtaining logs of stellar mass, and ratio of dust to stellar mass
parameter_files = [f"parameters{i}.txt" for i in range(1, 7)]
h5_files = [f"data{i}.h5" for i in range(1, 7)]


# In[20]:


df = generate_dataset(pd.DataFrame(columns=["log_mstar", "log_mdust_over_mstar", "theta", "n", "flux", "r"]), parameter_files, h5_files)
df


# In[21]:


# model = ResulDualOutputBNN(no_of_neurones=1000, dropout_prob=0.3).to(device)

# mse_loss = nn.MSELoss().to(device)
# kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False).to(device)
# kl_weight = 0.01

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# In[ ]:




