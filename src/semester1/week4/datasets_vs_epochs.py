#!/usr/bin/env python
# coding: utf-8

# In this notebook, I'm going to plot the relation between training & testing dataset costs against epochs.
# 
# Notes:
# * I have to reinitialise the model every time I am about to feed it data, otherwise the model is retraining.
# * In retraining.ipynb, my aim was to see if I could produce an accurate mean model with an extremely small dataset. Since my aim in this notebook is different, I'll use regular-sized dataset for simplicity.

# In[38]:


#importing libraries
import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import matplotlib.pyplot as plt
import pandas as pd


# In[39]:


is_available = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[40]:


x_train = torch.rand(500)*4-2
y_train = x_train.pow(5) - 10 * x_train.pow(1) + 2*torch.rand(x_train.size())

plt.scatter(x_train.data.numpy(), y_train.data.numpy())
plt.show()

x_train = torch.unsqueeze(x_train, dim=1).to(device)
y_train = torch.unsqueeze(y_train, dim=1).to(device)


# In[41]:


def clean_target(x):
    return x.pow(5) - 10 * x.pow(1)+1


def target(x):
    return x.pow(5) - 10 * x.pow(1) + 2*torch.rand(x.size())


# In[42]:


def intialise_model(no_of_neurones):
    model = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1,
                        out_features=no_of_neurones),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                        in_features=no_of_neurones, out_features=1),
    ).to(device)

    return model



# In[43]:


no_of_neurones = 2000
model = intialise_model(no_of_neurones)

mse_loss = nn.MSELoss().to(device)
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False).to(device)
kl_weight = 0.01

optimizer = optim.Adam(model.parameters(), lr=0.01)


# In[44]:


def train(model: nn.Sequential, x_train: torch.Tensor, y_train: torch.Tensor, epochs: int):
    for _ in range(epochs):
        predictions = model(x_train)
        mse = mse_loss(predictions, y_train)
        kl = kl_loss(model)
        cost = mse + kl_weight * kl

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    print(f'- With {epochs} epochs, MSE : {mse.item():2.2f}, KL : {kl.item():2.2f}, Cost: {cost.item():2.2f}')

    return model, mse, kl, cost


# Okay, everything I need is ready.
# 
# I'm testing for a range of epochs, so I'll need to call the above function quite a lot.

# In[45]:


no_of_epochs = [epochs for epochs in range(500, 5001, 500)]
cost_train_list = []
cost_test_list = []

for epochs in no_of_epochs:
    model, mse, kl, cost = train(model, x_train, y_train, epochs)
    cost_train_list.append(cost.item())

    x_test = torch.rand(1000)*4-2
    y_test = target(x_test)

    x_test = torch.unsqueeze(x_test, dim=1).to(device)
    y_test = torch.unsqueeze(y_test, dim=1).to(device)

    model, mse, kl, cost = train(model, x_test, y_test, epochs)
    cost_test_list.append(cost.item())

    model = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1,
                        out_features=no_of_neurones),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                        in_features=no_of_neurones, out_features=1),
    ).to(device)

    mse_loss = nn.MSELoss().to(device)
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False).to(device)
    kl_weight = 0.01

    optimizer = optim.Adam(model.parameters(), lr=0.01)


# In[46]:


# Plotting the results
fig = plt.figure(figsize=(10, 7))
plt.plot(
    no_of_epochs[1:], 
    cost_train_list[1:], 
    color='blue',
    marker='o',
    markersize=5,
    linestyle='None',
    label='Train Cost'
)

plt.plot(
    no_of_epochs[1:], 
    cost_test_list[1:], 
    color='green',
    marker='o',
    markersize=5,
    linestyle='None',
    label='Test Cost'
)

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Cost')

