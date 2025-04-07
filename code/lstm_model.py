import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, input_size, lstm_size, fnn_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_size)
        self.first_layer = nn.Linear(lstm_size, fnn_size)
        self.act = nn.GELU()
        self.second_layer = nn.Linear(fnn_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.first_layer(out)
        out = self.act(out)
        out = self.second_layer(out)
        return out
