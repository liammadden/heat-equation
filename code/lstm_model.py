import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, input_size, lstm_size, fnn_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_size)
        self.first_layer = nn.Linear(lstm_size, fnn_size)
        self.activation = nn.GELU()
        self.second_layer = nn.Linear(fnn_size, output_size)

    def forward(self, x):
        """
        x: (sequence_length, batch_size, input_size)
        """

        out, _ = self.lstm(x)  # (sequence_length, batch_size, lstm_size)
        out = self.first_layer(out)  # (sequence_length, batch_size, fnn_size)
        out = self.activation(out)
        out = self.second_layer(out)  # (sequence_length, batch_size, output_size)
        return out  # (sequence_length, batch_size, output_size)
