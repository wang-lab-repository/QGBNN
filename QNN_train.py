import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, ZFeatureMap, TwoLocal
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
# Additional torch-related imports

import torch.optim as optim
from Model import Net

import time

from sklearn.model_selection import train_test_split
import torch 
import pandas as pd 
from utils import mix_seed
import torch
from early_stopping import EarlyStopping
from data import get_data, get_dataloader
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
from torch import nn
from torch.autograd import Variable
import shap
from qiskit.circuit import  Parameter, ParameterVector
warnings.filterwarnings('ignore')


model = Net()
x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv.xlsx", random=9204)
train_load = get_dataloader(x_train, y_train)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"训练设备：{device}")
mix_seed(9204)
# mix_seed(77)S
epochs = 100
lr = 0.05
# lr_min = 5e-05
step_size = 35
optimizer = optim.Adam(model.parameters(), lr=lr)

# loss_func = L1Loss()
loss_func = MSELoss()
early_stopping = EarlyStopping(patience=300, delta=0)
# epochs = 10
loss_list = []
model.train()
# print(model.qnn.neural_network.output_shape)
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_load):
        # print(data.requires_grad() is True)
        # print(target.requires_grad() is True)
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad(set_to_none=True)  # Initialize gradient
        output = model(data)  # Forward pass
        # print(output._Shape)
        loss = loss_func(output, target)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        total_loss.append(loss.item())  # Store loss
    prediction_val = model(x_val)
    R2_val = 1 - torch.mean((y_val - prediction_val) ** 2) / torch.mean(
                (y_val - torch.mean(y_val)) ** 2)
    early_stopping(-R2_val)
    if early_stopping.early_stop:
        print("Early stopping")
        break    
    loss_list.append(sum(total_loss) / len(total_loss))
    print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))
    
torch.save(model.state_dict(), 'checkpoint/qnn_8_3-18.pth' )
x = np.arange(len(loss_list))
    # x = len(train_acc_list)
plt.plot(x, loss_list, label='train loss')
    
plt.xlabel("epochs")
plt.ylabel("loss")  
plt.savefig('result/the_total_train_loss_8_3-18.png') 
# plt.show()
model.qnn.neural_network.circuit.decompose().draw(output='mpl',style='iqp').savefig('result/the_model_circuit_8_3-18.tif', dpi=300)

prediction_train = model(x_train)
prediction_val = model(x_val)
prediction_test = model(x_test)

R2_train = 1 - torch.mean((y_train - prediction_train) ** 2) / torch.mean(
        (y_train - torch.mean(y_train)) ** 2)
R2_val = 1 - torch.mean((y_val - prediction_val) ** 2) / torch.mean(
        (y_val - torch.mean(y_val)) ** 2)
R2_test = 1 - torch.mean((y_test - prediction_test) ** 2) / torch.mean(
        (y_test - torch.mean(y_test)) ** 2)
print("------------------------结果------------------------")
print(f'train: R2：{R2_train.detach().numpy()}\n')
print(f'val: R2：{R2_val.detach().numpy()}\n')
print(f'test: R2：{R2_test.detach().numpy()}\n')
print(f'train: RMSE：{np.sqrt(mean_squared_error(y_train.detach().numpy(), prediction_train.detach().numpy()))}\n')
print(f'val: RMSE：{np.sqrt(mean_squared_error(y_val.detach().numpy(), prediction_val.detach().numpy()))}\n')
print(f'test: RMSE：{np.sqrt(mean_squared_error(y_test.detach().numpy(), prediction_test.detach().numpy()))}\n')

