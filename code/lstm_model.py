import torch
import torch.nn as nn

class GILRModel(nn.Module):

    def __init__(self, input_size, lstm_size, fnn_size, output_size):
        super(GILRModel, self).__init__()
        self.lstm = GILRNet(input_size, lstm_size)
        self.first_layer = nn.Linear(lstm_size, fnn_size)
        self.act = nn.GELU()
        self.second_layer = nn.Linear(fnn_size, output_size)

    def forward(self, x):
        out = self.lstm(x)
        out = self.first_layer(out)
        out = self.act(out)
        out = self.second_layer(out)
        return out

class GILRNet(nn.Module):

    def __init__(self, input_size, lstm_size):
        super(GILRNet, self).__init__()
        self.lstm_size = lstm_size
        self.lstm = GILRCell(input_size, lstm_size)

    def forward(self, x):
        """
        x: (input_length, batch_size, input_size)
        """

        input_length = x.shape[0]
        batch_size = x.shape[1]

        h = []
        h.append(torch.zeros(batch_size, self.lstm_size))
        for i in range(input_length):
            pre_h = h[-1]
            x_t = x[i, :, :]
            h_t = self.lstm(pre_h, x_t)
            h.append(h_t)
        h = torch.stack(h)[1:]

        return h # (input_length, batch_size, lstm_size)

class GILRCell(nn.Module):

    def __init__(self, input_size, lstm_size):
        super(GILRCell, self).__init__()
        self.linear_g = nn.Linear(input_size, lstm_size)
        self.linear_i = nn.Linear(input_size, lstm_size)

    def forward(self, pre_h, x_t):
        """
        pre_h: (batch_size, lstm_size) - previous hidden state
        x_t: (batch_size, input_size) - input at time step t
        """

        g_t = torch.sigmoid(self.linear_g(x_t))  # (batch_size, lstm_size)
        i_t = torch.tanh(self.linear_i(x_t))  # (batch_size, lstm_size)
        h_t = g_t * pre_h + (1 - g_t) * i_t  # (batch_size, lstm_size)

        return h_t  # (batch_size, lstm_size)

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

class MinLSTMModel(nn.Module):

    def __init__(self, input_size, lstm_size, fnn_size, output_size):
        super(MinLSTMModel, self).__init__()
        self.lstm = MinLSTM(input_size, lstm_size)
        self.first_layer = nn.Linear(lstm_size, fnn_size)
        self.act = nn.GELU()
        self.second_layer = nn.Linear(fnn_size, output_size)

    def forward(self, x):
        out = self.lstm(x)
        out = self.first_layer(out)
        out = self.act(out)
        out = self.second_layer(out)
        return out

class MinLSTM(nn.Module):

    def __init__(self, input_size, lstm_size):
        super(MinLSTM, self).__init__()
        self.lstm_size = lstm_size
        self.lstm = MinLSTMCell(input_size, lstm_size)

    def forward(self, x):
        """
        Args:
            (input_length, batch_size, input_size)

        output:
            (input_length, batch_size, lstm_size)

        """
        input_length = x.shape[0]
        batch_size = x.shape[1]

        # Initialize the hidden state, only the h needs to be initialized
        h = []
        h.append(torch.zeros(batch_size, self.lstm_size))

        # Pass the entire sequence through the LSTM
        for i in range(input_length):
            pre_h = h[-1]
            x_t = x[i, :, :]
            h_t = self.lstm(pre_h, x_t)
            h.append(h_t)
        h = torch.stack(h)[1:]

        return h

# Adapted from YecanLee/min-LSTM-torch
class MinLSTMCell(nn.Module):

    def __init__(self, input_size, lstm_size):
        super(MinLSTMCell, self).__init__()
        self.linear_f = nn.Linear(input_size, lstm_size)
        self.linear_i = nn.Linear(input_size, lstm_size)
        self.linear_h = nn.Linear(input_size, lstm_size)

    def forward(self, pre_h, x_t):
        """
        pre_h: (batch_size, lstm_size) - previous hidden state
        x_t: (batch_size, input_size) - input at time step t
        """

        # Forget gate: f_t = sigmoid(W_f * x_t)
        f_t = torch.sigmoid(self.linear_f(x_t))  # (batch_size, lstm_size)

        # Input gate: i_t = sigmoid(W_i * x_t)
        i_t = torch.sigmoid(self.linear_i(x_t))  # (batch_size, lstm_size)

        # Hidden state: tilde_h_t = W_h * x_t
        tilde_h_t = self.linear_h(x_t)  # (batch_size, lstm_size)

        # Normalize the gates
        sum_f_i = f_t + i_t
        f_prime_t = f_t / sum_f_i  # (batch_size, lstm_size)
        i_prime_t = i_t / sum_f_i  # (batch_size, lstm_size)

        # New hidden state: h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t
        h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t  # (batch_size, lstm_size)

        return h_t  # (batch_size, lstm_size)