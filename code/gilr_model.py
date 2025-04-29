import torch
import torch.nn as nn


class GILRModel(nn.Module):

    def __init__(self, input_size, lstm_size, fnn_size, output_size):
        super(GILRModel, self).__init__()
        self.lstm = GILRNet(input_size, lstm_size)
        self.first_layer = nn.Linear(lstm_size, fnn_size)
        self.activation = nn.GELU()
        self.second_layer = nn.Linear(fnn_size, output_size)

    def forward(self, x):
        """
        x: (sequence_length, batch_size, input_size)
        """

        out = self.lstm(x)  # (sequence_length, batch_size, lstm_size)
        out = self.first_layer(out)  # (sequence_length, batch_size, fnn_size)
        out = self.activation(out)
        out = self.second_layer(out)  # (sequence_length, batch_size, output_size)
        return out  # (sequence_length, batch_size, lstm_size)


class GILRNet(nn.Module):

    def __init__(self, input_size, lstm_size):
        super(GILRNet, self).__init__()
        self.lstm_size = lstm_size
        self.lstm = GILRCell(input_size, lstm_size)

    def forward(self, x):
        """
        x: (sequence_length, batch_size, input_size)
        """

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        h = []
        h.append(torch.zeros(batch_size, self.lstm_size))  # start with h_old = 0
        # compute h sequentially
        for i in range(sequence_length):
            h_old = h[-1]
            x_new = x[i, :, :]
            h_new = self.lstm(h_old, x_new)
            h.append(h_new)
        h = torch.stack(h)[1:]

        return h  # (sequence_length, batch_size, lstm_size)


class GILRCell(nn.Module):

    def __init__(self, input_size, lstm_size):
        super(GILRCell, self).__init__()
        self.linear_g = nn.Linear(input_size, lstm_size)
        self.linear_i = nn.Linear(input_size, lstm_size)

    def forward(self, h_old, x_new):
        """
        h_old: (batch_size, lstm_size) - old hidden state
        x_new: (batch_size, input_size) - new input state
        """

        g_new = torch.sigmoid(self.linear_g(x_new))  # (batch_size, lstm_size)
        i_new = torch.tanh(self.linear_i(x_new))  # (batch_size, lstm_size)
        h_new = g_new * h_old + (1 - g_new) * i_new  # (batch_size, lstm_size)

        return h_new  # (batch_size, lstm_size)
