import math

import torch
import torch.nn as nn


class AttentionModel(nn.Module):

    def __init__(
        self, input_size, attn_size, fnn_size, output_size, max_length, device
    ):
        super(AttentionModel, self).__init__()
        self.position = PositionalEncoding(input_size, max_length)
        self.attention = Attention(input_size, attn_size, device)
        self.first_layer = nn.Linear(attn_size, fnn_size)
        self.activation = nn.GELU()
        self.second_layer = nn.Linear(fnn_size, output_size)

    def forward(self, x):
        """
        x: (sequence_length, batch_size, input_size)
        """

        out = self.position(x)
        out = self.attention(out)  # (sequence_length, batch_size, attn_size)
        out = self.first_layer(out)  # (sequence_length, batch_size, fnn_size)
        out = self.activation(out)
        out = self.second_layer(out)
        return out  # (sequence_length, batch_size, output_size)


class PositionalEncoding(nn.Module):

    def __init__(self, input_size, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.input_size = input_size
        pe = torch.zeros(max_length, input_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_size, 2).float() * (-math.log(10000.0) / input_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        x: (sequence_length, batch_size, input_size)
        """

        sequence_length = x.shape[0]
        out = x + self.pe[:, :sequence_length, : self.input_size].transpose(0, 1)
        return out  # (sequence_length, batch_size, input_size)


class Attention(nn.Module):

    def __init__(self, input_size, attn_size, device):
        super(Attention, self).__init__()
        self.attn_size = attn_size
        self.device = device
        self.W_q = nn.Linear(input_size, attn_size)
        self.W_k = nn.Linear(input_size, attn_size)
        self.W_v = nn.Linear(input_size, attn_size)

    def scaled_dot_product_attention(self, Q, K, V):
        """
        Q, K, V: (..., sequence_length, attn_size)
        """

        sequence_length = Q.shape[-2]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.attn_size
        )  # (..., sequence_length, sequence_length)
        attn_scores = attn_scores.masked_fill(
            torch.tril(torch.ones(sequence_length, sequence_length).to(self.device))
            == 0,
            float("-inf"),
        )
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output  # (..., sequence_length, attn_size)

    def forward(self, x):
        """
        x: (sequence_length, batch_size, input_size)
        """

        x = x.transpose(0, 1)  # (batch_size, sequence_length, input_size)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        # Q, K, V: (batch_size, sequence_length, attn_size)
        output = self.scaled_dot_product_attention(
            Q, K, V
        )  # (batch_size, sequence_length, attn_size)
        output = output.transpose(0, 1)
        return output  # (sequence_length, batch_size, attn_size)
