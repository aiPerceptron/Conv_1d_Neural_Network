# the output shape of nn.flatten when the input is 3d IE (7,2,6), not 2d IE( 2,6)
# the batch size plays a role


#----
# the impact of going from a 2d input from a 3d input 
# 2d imput is (2,6) 3d imput (7,2,6)
# study impact on X and xb (the batch size plays a crucial role)
# and on the output of the convolutional layer
# First finding:
# X is 7,2,6
# batch_size = 1
# xb is 1(batch_size),2,6
# when xb used to be 2,6 convolutional out = 5,4
# when xb is 1,2,6 convolutional output = 1,5,4

# Simple CNN
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset,DataLoader,random_split

from tqdm import tqdm

#torch.manual_seed(42) # makes it so that the seed doesnt change

# Inspecting the hardware # preparation 1

isgpu = torch.cuda.is_available()
if isgpu is True:
    device = torch.device("cuda")
    print("GPU is available")
else:
    print("GPU isn't available")
    device = torch.device("cpu")

# Hyperparameters # preparation 2 you pick them, and they impact the algorithms.
C_out = 5 # how many outputs
kernal_size = 3
n_epochs = 50
batch_size = 2
lr = 0.001

# These determine the shape of the randomly generated data
N = 365
C = 2
L = 6

# Data

# Manually Created Data

"""X = torch.tensor([[ # monday 
    #3   6   9   12  3   6
    #am  am  am  pm  pm  pm
    [40, 55, 65, 70, 65, 55], # humidity
    [65, 70, 75, 80, 75, 65] # temp
    ], [ # tuesday
    [45, 60, 70, 75, 65, 45], # humidity
    [60, 65, 72, 77, 66, 60] # temp
    ], [ # wednesday
    [10, 15, 20, 25, 20, 10], # humidity
    [25, 30, 40, 45, 40, 35] # temp
    ], [ # thursday
    [15, 20, 25, 20, 25, 20], # humidity
    [30, 35, 40, 45, 40, 30] # temp
    ], [ # friday
    [20, 25, 30, 35, 30, 25], # humidity
    [32, 37, 42, 47, 42, 37] # temp
    ], [ # saturday
    [40, 57, 65, 68, 52, 40], # humidity
    [60, 65, 68, 62, 57, 51] # temp
    ], [ # sunday
    [45, 60, 70, 75, 65, 55], # humidity
    [65, 70, 72, 86, 71, 60] # temp
    ],
                 ], dtype=torch.float32)/100

y = torch.tensor([1,1,0,0,0,1,1],dtype=torch.int64)"""

# Random Data
#Good_X = 70 + torch.randn(N//2, C, L)
#Bad_X = 40 + torch.randn(N//2, C, L)
#X = torch.concat((Good_X, Bad_X)).to(torch.float32)

# Randomly Generated Data

Good_Temperature = 75 + torch.randn(N//2, L)
Good_Humidity = 55 + torch.randn(N//2, L)
Good_X = torch.stack((Good_Temperature, Good_Humidity))
Good_X = Good_X.permute(1,0,2)

Bad_Temperature = 74 + torch.randn(N//2, L)
Bad_Humidity = 56 + torch.randn(N//2, L)
Bad_X = torch.stack((Bad_Temperature, Bad_Humidity))
Bad_X = Bad_X.permute(1,0,2)
X = torch.concat((Good_X, Bad_X)) 

y = torch.concat((torch.ones(N//2),torch.zeros(N//2))).to(torch.int64)

N_classes = len(torch.unique(y))

# N = 7, C_in = 2, L = 6
N,C_in,L = X.shape
ds_Xy = TensorDataset(X,y)

ds_train, ds_test = random_split(ds_Xy, [0.9,0.1])

dl_train = DataLoader(ds_train,batch_size=batch_size,shuffle=True)
dl_test = DataLoader(ds_test,batch_size=batch_size) # maybe if i have more time create a dataloader for the data so you can feed it into your ais

# Model

model = nn.Sequential(nn.Conv1d(C_in,C_out,kernal_size, bias = False),nn.Flatten(),nn.LazyLinear(N_classes))
#print(model.weight.data)
#print(model.bias.data) 
#model.weight.data = torch.tensor([[[1,1,1],[2,2,2]]], dtype=torch.float32)

# Training

# Training #3

model.train() # prepares the model for training
loss_function = nn.CrossEntropyLoss() # object for finding the loss, loss function works well with classification
optimizer = optim.Adam(model.parameters(), lr=lr) # object for adjusting the machine learning models weights
for epoch in tqdm(range(n_epochs)): # loop to loop over the loop for a long time
    for xb,yb in dl_train: # loops over the batches
        xb,yb = xb.to(device),yb.to(device) # moves the data to the device (CPU, GPU)
        #print(xb.shape)
        ml_out = model(xb) # the models answer
        #print(ml_out.shape)
        #print(yb.shape)
        loss = loss_function(ml_out,yb) # you're comparing ml_out with y[i] here to see the loss (aka what adjustments you should make next time)
        optimizer.zero_grad() # sets gradient to 0
        loss.backward() # compute the gradient
        optimizer.step() # goes down the gradient in little baby steps (to make the loss less)

# Testing
model.eval()
ml_out = model(X) 
#print(ml_out)

#F.conv1d(X,model.weight.data,bias=None) # 

#model.weight.data.shape
#model.weight.data

# C_in L_in --> C_out,C_in,K --> C_out,L_out
# X.shape
#ml_out.shape

# the bigger number means what type of good or bad day it is

y_hat = ml_out.argmax(axis=1)
compare_y_test_y_hat = y_hat == y
#print(y_hat)
#print(compare_y_test_y_hat)
#ml_out

percent_accuracy = (sum(compare_y_test_y_hat)/len(y_hat))*100
print(percent_accuracy)

