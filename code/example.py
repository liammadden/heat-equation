import numpy as np

from crank_nicolson import make_data
from lstm_model import LSTMModel

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data

n_x = 10
n_t = 10
c_x = 10
c_t = 10
r = .001
u_min = 0
u_max = 1
num_samples = 100
print(num_samples*(n_t-1))

np.random.seed(0)
u = make_data(n_x, n_t, c_x, c_t, r, u_min, u_max, num_samples) # n_t by num_samples by n_x
u = torch.tensor(u).to(torch.float32)

m = 20
num_epochs = 1000

# model = nn.LSTM(input_size=n_x, hidden_size=n_x)
# model = nn.LSTM(input_size=n_x, hidden_size=m, proj_size=n_x)
model = LSTMModel(input_size=n_x, hidden_size=m, output_size=n_x)
model_num_params = sum(p.numel() for p in model.parameters())
print(model_num_params)

criterion = nn.MSELoss()
step_size = .001
optimizer = optim.Adam(model.parameters(), lr=step_size)
trainloader = torch_data.DataLoader(u, batch_size=num_samples, shuffle=False)

for epoch in range(num_epochs):
    for _, batch in enumerate(trainloader):
        optimizer.zero_grad()
        # output, _ = model(batch[:-1, :, :])
        output = model(batch[:-1, :, :])
        loss = criterion(output, batch[1:, :, :])
        loss.backward()
        optimizer.step()
    
    if epoch == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")