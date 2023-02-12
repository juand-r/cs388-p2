# models.py

import numpy as np
import random
import torch.nn as nn
from torch import optim
import torch

from transformer import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, position_style):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.d_internal = d_internal # dimension of keys and queries
        self.num_classes = num_classes
        self.num_layers = num_layers

        # later
        self.d_value = d_internal # just for now ... 20
        self.d_ff = 2048#1024 #480 #PREV 250

        num_heads = 3 # 6##3 4

        # only one layer for now!
        #if self.num_layers !=1:
        #    raise NotImplementedError("TODO: more than one transformer layer")

        self.tl1 = TransformerLayer(self.d_model, self.d_internal, self.d_value, self.d_ff)
        self.log_softmax = nn.LogSoftmax(dim=1)#TODO check this is right dim
        self.W = nn.Linear(d_model, num_classes) # last linear layer before softmax

        #USE A RANDOM VECTOR EMBEDDING FOR EACH CHARACTER OR JUST ONE-HOT?
        self.emb = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.num_positions)

        self.position_style = position_style

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=num_heads,
                                                        dim_feedforward=self.d_ff,
                                                        batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """

        # input embedding + positional encoding:
        #E = torch.nn.functional.one_hot(torch.LongTensor(indices))
        # 20 x 27 ....
        seqlen = len(indices)

        E = self.emb(indices) # size T x d_model   [T = len of sequence]

        if self.position_style == 'learned':
            # POSITIONAL ENCODING -- this will add E to the positional encoding
            E2 = self.positional_encoding(E)
        if self.position_style == 'none':
            E2 = E

#################################
#       #NOTE here is my version, without multiple attention heads
#        # pass through transformer layer N times (once for now)
#        attention, V = self.tl1(E2)
#        attn_maps = [attention]

        #NOTE: easier to use the pytorch implementation of TransformerEncoderLayer

        mask = torch.triu(torch.ones(seqlen, seqlen) * float('-inf'),  diagonal=1)
        V = self.transformer_encoder(E2.unsqueeze(dim=0), mask=mask)
        V = V.squeeze(0)

        # final linear layer  and  softmax
        log_probs = self.log_softmax(self.W( V  )  )

        return log_probs
#        return log_probs, attn_maps


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, d_value, d_ff):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries should both be of this length.
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.d_value = d_value
        self.d_ff = d_ff

        self.softmax = nn.Softmax(dim=1) # want columns to sum to 1

        self.WQ = nn.init.xavier_uniform_(torch.empty(d_model, d_internal))
        self.WK = nn.init.xavier_uniform_(torch.empty(d_model, d_internal))

        self.WV = nn.init.xavier_uniform_(torch.empty(d_model, d_value))

        self.WO = nn.Linear(d_value, d_model)

        self.FF1 = nn.Linear(d_model, d_ff)
        self.FF2 = nn.Linear(d_ff, d_model)
        self.g = nn.ReLU()


    def forward(self, input_vecs):
        #raise Exception("Implement me")

        #NOTE input_vecs is a matrix (T rows x d_model )
        seqlen = input_vecs.shape[0]

        Q = torch.matmul(input_vecs, self.WQ)

        K = torch.matmul(input_vecs, self.WK)

        V = torch.matmul(input_vecs, self.WV)

        #TODO more than one attention head
        # self-attention
        mask = torch.triu(torch.ones(seqlen, seqlen) * float('-inf'),  diagonal=1)
        SA = self.softmax(mask + torch.matmul(Q, torch.transpose(K,0,1))/np.sqrt(self.d_internal) )

        # do weighted average, map back to dim d_model using welf.WO, and residual connection
        attouts = input_vecs + self.WO( torch.matmul(SA, V) )

        #feed forward.. use same weight matrix for each position
        output_vecs = attouts  +  self.FF2( self.g( self.FF1( attouts  ) ))

        #input_vecs

        #output_vecs = attouts

        return SA, output_vecs


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_index, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, position_style):

        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.d_internal = d_internal # dimension of keys and queries
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.vocab_index = vocab_index

        # later
        self.d_value = d_internal # just for now ... 20
        self.d_ff = 1000#480 #PREV 250

        self.model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, position_style)

    def get_next_char_log_probs(self, context):
        self.model.eval()

        #REMEMBER, BOS is represented by " "
        context = " "+context
        ex = torch.LongTensor([self.vocab_index.index_of(i) for i in context])
#        (log_probs, attn_maps) = self.model.forward(ex)
        log_probs = self.model.forward(ex)

        return log_probs[-1].detach().numpy()
        #:param context: the string context that the LM conditions on
        #:return: A numpy vector log P(y | context) where y ranges over the output vocabulary.


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """

    chunk_len = 20
    bos = ' '

    train_chunks = [train_text[i:i+chunk_len] for i in range(0,len(train_text),chunk_len)]
    train_chunks_shifted = [bos+x[:-1] for x in train_chunks]

    train_chunks_indexed = [torch.LongTensor(np.array([vocab_index.index_of(ci) for ci in chunk]))   for chunk in train_chunks]

    train_chunks_shifted_indexed = [torch.LongTensor(np.array([vocab_index.index_of(ci) for ci in chunk])) for chunk in train_chunks_shifted]


    dev_chunks = [dev_text[i:i+chunk_len] for i in range(0,len(dev_text),chunk_len)]
    dev_chunks_shifted = [bos+x[:-1] for x in dev_chunks]

    dev_chunks_indexed = [torch.LongTensor(np.array([vocab_index.index_of(ci) for ci in chunk])) for chunk in dev_chunks]

    dev_chunks_shifted_indexed = [torch.LongTensor(np.array([vocab_index.index_of(ci) for ci in chunk])) for chunk in dev_chunks_shifted]

    vocab_size = len(vocab_index)
    num_positions = len(train_chunks_indexed[0])


    learning_rate = 1e-3
    d_model = 120 #512 ##120 #original transformer uses 512
    d_internal = 15 #64##15 #original transformer uses 64
    num_classes = vocab_size
    num_layers = 1
    position_style = 'learned'

    lm = NeuralLanguageModel(vocab_index, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, position_style)

    lm.model.zero_grad()
    optimizer = optim.Adam(lm.model.parameters(), lr=learning_rate)

    num_epochs = 10#10

    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train_chunks_indexed))]
        random.shuffle(ex_idxs)

        loss_fcn = nn.NLLLoss()

        for ind in ex_idxs:

            #ex = train[ind]
            ex_input = train_chunks_shifted_indexed[ind]
            ex_output = train_chunks_indexed[ind]

            lm.model.zero_grad()

            # input of size 20
#            log_probs, attn_maps = lm.model.forward(ex_input)
            log_probs = lm.model.forward(ex_input)

            loss = loss_fcn(log_probs, ex_output)

            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print("Training loss on EPOCH: "+str(t)+": ", loss_this_epoch)

    lm.model.eval()
    return lm
