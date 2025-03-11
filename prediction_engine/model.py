import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import wandb
from tqdm import tqdm
from datetime import datetime

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):

    def __init__(self, n_items, num_layers=2, input_size=5, item_index=3, embedding_size=16, hidden_size=16, dropout_p=0.1, bidirectional=True):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.item_index = item_index

        self.embedding = nn.Embedding(n_items, embedding_size)
        self.rnn = nn.GRU(input_size + embedding_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, X, lengths):
        item_ids = X[:, :, self.item_index].long()
        X = torch.cat([X[:, :, :self.item_index], X[:, :, self.item_index + 1:]], dim=2)
        item_embeddings = self.dropout(self.embedding(item_ids))

        X = torch.cat([X, item_embeddings], dim=2)
        X_packed = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)

        output_packed, hidden = self.rnn(X_packed)

        return output_packed, hidden


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=True, dropout_p=0.1):
        super(Decoder, self).__init__()
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout_p)
        self.projection = nn.Linear(output_size, 1)

    def forward(self, encoder_outputs, encoder_hidden):
        output_packed, _ = self.rnn(encoder_outputs, encoder_hidden)

        output, _ = pad_packed_sequence(output_packed, batch_first=True)
        output = self.projection(output)

        return output


class AuctionRNN(nn.Module):
    
    def __init__(self, n_items, num_layers=2, input_size=5, encoder_hidden_size=16, decoder_hidden_size=16, item_index=3, embedding_size=16, dropout_p=0.1, bidirectional=True):
        super(AuctionRNN, self).__init__()
        decoder_input_size = encoder_hidden_size * 2 if bidirectional else encoder_hidden_size
        self.encoder = Encoder(n_items, input_size=input_size, item_index=item_index, embedding_size=embedding_size, hidden_size=encoder_hidden_size, dropout_p=dropout_p, num_layers=num_layers, bidirectional=bidirectional)
        self.decoder = Decoder(decoder_input_size, decoder_hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout_p=dropout_p)

    def forward(self, X, lengths):
        encoder_outputs, encoder_hidden = self.encoder(X, lengths)
        decoder_outputs = self.decoder(encoder_outputs, encoder_hidden)
        return decoder_outputs
