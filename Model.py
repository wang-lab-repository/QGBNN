from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, ZFeatureMap, TwoLocal
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
# Additional torch-related imports
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import time
import random
from math import pi
import numpy as np
from torch.nn import (
    Module,
    
    Linear,
    
    L1Loss,
    MSELoss,
    
    Sequential,
    ReLU,
)

def parity(x):
    return x % 32

def create_qnn():
    feature_map = ZFeatureMap(8,reps=1)
    ansatz = TwoLocal(8,'ry', 'cx', 'linear', reps=1, insert_barriers=True)
    # print(ansatz.preferred_init_points())
    qc = QuantumCircuit(8)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)


    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True
        
    )
    return qnn


qnn = create_qnn()



class Net(Module):
    def __init__(self):
        super().__init__()
        self.inline = Linear(8,8)
        self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        # self.outline = Linear(1, 1)  # 1-dimensional output from QNN
        self.hideline1 = Linear(256,64)
        self.hideline2 = Linear(64,8)
        # self.hideline  = Linear(64,8)
        self.outline   = Linear(8,1)
        self.relu      = ReLU()
        self.initial_weights = np.random.uniform(0, 2*pi, size=qnn.num_weights)
    def forward(self, x):
        # x = self.inline(x)
        x = self.qnn(x)  # apply QNN
        # print(x)
        # x = self.outline(x)
        # return x
        x = self.hideline1(x)
        x = self.relu(x)
        x = self.hideline2(x)
        x = self.relu(x)
        # x = self.hideline(x)
        x = self.outline(x)
        return self.relu(x)
    def load(self, path):
        self.load_state_dict(torch.load(path))
        pass
    def save(self, name='samplerNet'):
        
        # prefix = 'checkpoint\\' + name +'_'
        # save_name   = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        # torch.save(self.state_dict(), save_name)
        torch.save(self.state_dict(),'checkpoint/%s_%s.pth'%('samplerNet',time.strftime('%m%d_%H:%M:%S')))
        pass
    def predict(self,x):
        return self.forward(x)