# FederatedLearning.py
import torch
import torch.nn as nn
import torch.optim as optim

class FederatedLearning:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)

    def train(self, epochs):
        for epoch in range(epochs):
            for data in self.data_loader:
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                self.optimizer.step()


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FederatedModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FederatedModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
