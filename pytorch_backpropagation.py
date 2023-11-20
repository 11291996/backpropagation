import torch 
import torch.nn as nn 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time 
#set m1 gpu
device = torch.device('mps')
#loading data
data_ready = np.loadtxt('./training.txt')
data = [i for i in range(2)]
data[0] = torch.from_numpy(data_ready[:, :2]).float()
data[1] = torch.from_numpy(data_ready[:, 2:3]).float()
#setting hyperparameters
learning_rate = 1e-3
batch_size = 1000
epochs = 5
layers = (2,4,5,4,5,6)
#creating torch dataset class and instance
class dataset(torch.utils.data.Dataset):
    i = 0
    def __init__(self, data):
        super().__init__()
        self.data = data
    def __getitem__(self, index):
        return self.data[0][index].float(), self.data[1][index].float()
    def __len__(self):
        return self.data[0].shape[0]

dataset = dataset(data)

#making dataloader by this
dataloader = DataLoader(dataset, batch_size, shuffle = True)

first_batch = dataloader.__iter__().__next__()
print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15s} | {:<25s} | {}'.format('Num of Batch', '', len(dataloader)))
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape))
print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))


#creating model
layers = (2,) + layers + (1,)
linear_add = [nn.Linear(layers[i], layers[i + 1]) for i in range((len(layers) - 1))] #(2, 2, 4, 5, 1)
print(linear_add[2])
model = nn.Sequential()
j = 0; k = 1
for i in range(len(layers) - 1):
    model.add_module('linear{}'.format(i), linear_add[i])
    model.add_module('sigmoid{}'.format(i), nn.Sigmoid())
print(model)
#defining optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) 
model.train() #setting model status
#sending model to device
model = model.to(device) 
#loss function
criterion = nn.MSELoss()
#training loop 
i = 0

for epoch in range(epochs):
    start = time.time()
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print("Train Step : {}\tLoss : {:3f}".format(i, loss.item()))
        i += 1
    end = time.time()
    print(end - start)

