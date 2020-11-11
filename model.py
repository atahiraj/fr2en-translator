"""
INFO2049-1: Practical project: Machine translation 1
Authors: Folon Nora, Tahiraj Arian
Parts of the code are inspired from:
    - Title: Torchtext translation tutorial
      Authors: Seith Weidman
      Availability: https://github.com/pytorch/tutorials/blob/master/beginner_source/torchtext_translation_tutorial.py
    - Title: Pytorch Seq2Seq
      Authors: Ben Trevett
      Availability: https://github.com/bentrevett/pytorch-seq2seq
"""

import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from torchtext.vocab import Vectors


class Encoder(nn.Module):
    """ Encoder class representing the many to one encoding for the translation

    Coutains 3 layers:
    - 1 embedding layer
    - 1 LSTM
    - 1 dropout layer
    """

    def __init__(
            self,
            emb_dim: int,
            emb_vectors: Vectors,
            input_dim: int,
            hid_dim: int,
            n_layers: int,
            dropout: float):
        """ Create an encoder object

        :param emb_dim: Embedding dimension
        :type emb_dim: int
        :param emb_vectors: Embedding vectors
        :type emb_vectors: torchtext.vocab.Vectors
        :param input_dim: Input dimension of the LSTM
        :type input_dim: int
        :param hid_dim: Size of hidden states of the LSTM (of c and h, not
            concatenated)
        :type hid_dim: int
        :param n_layers: Number of layers of the LSTM
        :type n_layers: int
        :param dropout: Probability of an element to be zeroed for the dropout
            and LSTM layer
        :type dropout: float

        :rtype: Encoder
        """
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding.weight.data.copy_(emb_vectors)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor):
        """ Forwards and element through the layers

        :param src: Input tensor
        :type src: torch.Tensor

        :return: Returns a tuple containing the outputs and the hidden states
            of the LSTM
            (see https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
        :rtype: tuple
        """
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return outputs, hidden


class Decoder(nn.Module):
    """ Encoder class representing the one to many decoding for the translation

    Coutains 3 layers:
    - 1 embedding layer
    - 1 LSTM
    - 1 fully connected layer
    - 1 dropout layer
    """

    def __init__(
            self,
            emb_dim: int,
            emb_vectors: int,
            output_dim: int,
            hid_dim: int,
            n_layers: int,
            dropout: float):
        """ Create a decoder object

        :param emb_dim: Embedding dimension
        :type emb_dim: int
        :param emb_vectors: Embedding vectors
        :type emb_vectors: torchtext.vocab.Vectors
        :param output_dim: Input dimension of the LSTM and output dimension of
            the fully connected layer
        :type output_dim: int
        :param hid_dim: Size of hidden states of the LSTM (of c and h, not
            concatenated)
        :type hid_dim: int
        :param n_layers: Number of layers of the LSTM
        :type n_layers: int
        :param dropout: Probability of an element to be zeroed for the dropout
            and LSTM layer
        :type dropout: float

        :rtype: Decoder
        """
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.embedding.weight.data.copy_(emb_vectors)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor, encoder_outputs: Tensor):
        """ Forwards and element through the layers

        :param src: Input tensor
        :type src: torch.Tensor
        :param encoder_outputs: Hidden states of the encoder
        :type encoder_outputs: Tensor

        :return: Returns a tuple containing the outputs and the hidden states
            of the LSTM
            (see https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
        :rtype: tuple
        """
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, hidden = self.rnn(embedded, encoder_outputs)

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    """ Seq2Seq class implementing a machine translation

    Coutains 3 layers:
    - 1 encoder
    - 1 decoder
    """

    def __init__(self,
                 emb_dim: int,
                 src_vectors: Vectors,
                 trg_vectors: Vectors,
                 hid_dim: int,
                 n_layers: int,
                 input_dim: int,
                 enc_dropout: float,
                 output_dim: int,
                 dec_dropout: float,
                 device: torch.device):
        """ Create a decoder object

        :param emb_dim: Embedding dimension
        :type emb_dim: int
        :param src_vectors: Embedding vectors of the source language
        :type src_vectors: torchtext.vocab.Vectors
        :param trg_vectors: Embedding vectors of the target language
        :type trg_vectors: torchtext.vocab.Vectors
        :param hid_dim: Size of hidden states of the LSTMs (of c and h, not
            concatenated)
        :type hid_dim: int
        :param n_layers: Number of layers of the LSTMs
        :type n_layers: int
        :param input_dim: Input dimension of the encoder's LSTM
        :type input_dim: int
        :param enc_dropout: Probability of an element to be zeroed for the
            dropout layers of the ecoder
        :type enc_dropout: float
        :param output_dim: Input dimension of the decoder's LSTM and output
            dimension of the decoder's fully connected layer
        :type output_dim: int
        :param dec_dropout: Probability of an element to be zeroed for the
            dropout layers of the decoder
        :type dec_dropout: float
        :param device: Device on which the model will run
        :type device: torch.device

        :rtype: Seq2Seq
        """

        super().__init__()

        self.encoder = Encoder(
            emb_dim,
            src_vectors,
            input_dim,
            hid_dim,
            n_layers,
            enc_dropout)
        self.decoder = Decoder(
            emb_dim,
            trg_vectors,
            output_dim,
            hid_dim,
            n_layers,
            dec_dropout)
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """ Forwards and element through the layers

        :param src: Input tensor
        :type src: torch.Tensor
        :param trg: Target Sentence (for training)
        :type trg: torch.Tensor
        :param teacher_forcing_ratio: Probability to use teacher forcing
            e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75%
            of the time
        :type teacher_forcing_ratio: float

        :return: The translated sentence
        :rtype: Tensor
        """
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(
            trg_len,
            batch_size,
            trg_vocab_size).to(
            self.device)

        # last hidden state of the encoder is used as the initial hidden state
        # of the decoder
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell
            # states
            output, hidden = self.decoder(input, hidden)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
