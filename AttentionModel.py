# author:Luinage ~ 2024
import torch
import torch.nn as nn

class CustomAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomAttentionModel, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        energy = torch.tanh(self.linear(x))
        attention_weights = torch.softmax(self.attn(energy), dim=1)
        context = attention_weights * x
        return context

class CustomModelWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModelWithAttention, self).__init__()
        self.attention = CustomAttentionModel(input_size, hidden_size)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        context = self.attention(x)
        output = self.fc(context)
        return output
