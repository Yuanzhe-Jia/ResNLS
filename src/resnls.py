'''
This is the code of the proposed model in the paper:
ResNLS: An Improved Model for Stock Price Forecasting

Structure of code:
1. data preprocessing
2. model definition
3. model training
4. model validation
'''

##################### data preprocessing #####################

# install packages
!pip install baostock

# import packages
import math
import numpy as np
import pandas as pd
import baostock as bs
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# download CSI dataset
lg = bs.login()

fields= "date,open,high,low,close"
rs = bs.query_history_k_data("sh.000001", fields, start_date="2012-01-01", end_date="2022-12-31", frequency="d", adjustflag="2")
data_list = []
while (rs.error_code == "0") & rs.next():
    data_list.append(rs.get_row_data())
df = pd.DataFrame(data_list, columns=rs.fields)
df.index=pd.to_datetime(df.date)

bs.logout()

# check dataset
# df.info()

# create the dataset only containing close prices
data = pd.DataFrame(pd.to_numeric(df["close"]))
dataset = np.reshape(data.values, (df.shape[0], 1))

# normalise the dataset
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# define a function to get training and test data
def split_data(dataset, train_day, predict_day):
    x = []
    y = []
    for i in range(train_day, len(dataset)-predict_day+1):
        x.append(dataset[i-train_day : i, 0])
        y.append(dataset[i+predict_day-1, 0])
    return x, y

# x => data from previous days; y => data in the next day
def reshape_data(train_data, test_data, days):
    x_train, y_train = split_data(train_data, days, 1)
    x_test, y_test = split_data(test_data, days, 1)
    # convert data into numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)
    # reshape the data for neural network training
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    return x_train, y_train, x_test, y_test

# create the scaled training data set
training_data_len = math.ceil(len(dataset) * 0.9087)
train_data = scaled_data[0:training_data_len, :]
#print(data[:train_data.shape[0]].tail())

# create the scaled test data set
test_data = scaled_data[training_data_len-5: , :]

# use 5 consecutive trading days as the unit step size sliding through the stock price data
x_train_5, y_train_5, x_test_5, y_test_5 = reshape_data(train_data, test_data, 5)
print("when sequence length is 5, data shape:", x_train_5.shape, y_train_5.shape, x_test_5.shape, y_test_5.shape)


##################### model definition #####################

# import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from sklearn.metrics import accuracy_score, f1_score

# initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_input = 5; n_hidden = 64

class ResNLS(nn.Module):

    def __init__(self):
        super(ResNLS, self).__init__()

        # intialise weights of the attention mechanism
        self.weight = nn.Parameter(torch.zeros(1)).to(device)

        # intialise cnn structure
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_hidden, kernel_size=3, stride=1, padding=1), # ((5 + 1*2 - 3)/1 + 1) = 5
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden, eps=1e-5),
            nn.Dropout(0.1),

            nn.Conv1d(in_channels=n_hidden, out_channels=n_hidden, kernel_size=3, stride=1, padding=1), # ((5 + 1*2 - 3)/1 + 1) = 5
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden, eps=1e-5),

            nn.Flatten(),
            nn.Linear(n_input * n_hidden, n_input)
        )

        # intialise lstm structure
        self.lstm = nn.LSTM(n_input, n_hidden, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(n_hidden, 1)


    def forward(self, x):

        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(-1, 1, n_input)

        residuals = x + self.weight * cnn_output

        _, (h_n, _)  = self.lstm(x)
        y_hat = self.linear(h_n[0,:,:])

        return y_hat


##################### model training #####################

# prepare validation data
val_input = torch.tensor(x_test_5, dtype=torch.float).to(device)
val_target = torch.tensor(y_test_5, dtype=torch.float).to(device)

# initialization
epochs = 50; batch_size = 64

# model instance
model = ResNLS().to(device)

# loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mini-batch training
if x_train_5.shape[0] % batch_size == 0:
    batch_num = int(x_train_5.shape[0] / batch_size)
else:
    batch_num = int(x_train_5.shape[0] / batch_size) + 1

for epoch in range(epochs):
    for j in range(batch_num):

        # prepare training data
        train_input = torch.tensor(x_train_5[j * batch_size : (j+1) * batch_size], dtype=torch.float).to(device)
        train_targe = torch.tensor(y_train_5[j * batch_size : (j+1) * batch_size], dtype=torch.float).to(device)

        # training
        model.train()
        optimizer.zero_grad()
        train_output = model(train_input)
        train_loss = criterion(train_output, train_targe)
        train_loss.backward()
        optimizer.step()

    if (epoch+1) % (epochs/20) == 0:
        with torch.no_grad():
            model.eval()
            val_output = model(val_input)
            val_loss = criterion(val_output, val_target)
            print("Epoch: {:>3}, train loss: {:.4f}, val loss: {:.4f}".format(epoch+1, train_loss.item(), val_loss.item()))


##################### model validation #####################

# import packages
from sklearn import metrics

# get the model predicted price values
predictions = model(val_input)
predictions = scaler.inverse_transform(predictions.cpu().detach().numpy())
# plot the stock price
train = data[:training_data_len]
valid = data[training_data_len:]
valid["predictions"] = predictions

# calculate MSE
y = np.array(valid["close"])
y_hat = np.array(valid["predictions"])
mae = metrics.mean_absolute_error(y_hat, y)
mse = metrics.mean_squared_error(y_hat, y)
rmse = metrics.mean_squared_error(y_hat, y) ** 0.5
print("MAE:{:.2F}   MSE: {:.2f}   RMSE:{:.2F}".format(mae, mse, rmse))
