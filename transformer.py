# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *

from tqdm import tqdm

# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, position_style, num_heads):
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
        self.position_style = position_style
        self.num_heads = num_heads

        # later
        self.d_value = d_internal # just for now ... 20
        self.d_ff = 512 ##480 #1024#480 #PREV 250

        # only one layer for now!
        #if self.num_layers !=1:
        #    raise NotImplementedError("TODO: more than one transformer layer")

        self.tl1 = TransformerLayer(self.d_model, self.d_internal, self.d_value, self.d_ff, self.position_style, self.num_heads)
        self.log_softmax = nn.LogSoftmax(dim=1)#TODO check this is right dim
        self.W = nn.Linear(d_model, num_classes) # last linear layer before softmax

        #self.E2 = nn.Linear()
        #USE A RANDOM VECTOR EMBEDDING FOR EACH CHARACTER OR JUST ONE-HOT?
        self.emb = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.num_positions)
        self.sinusoid_encoding = SinusoidEncoding(self.d_model)

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """

        # input embedding + positional encoding:
        #E = torch.nn.functional.one_hot(torch.LongTensor(indices))
        # 20 x 27 ....


        V = self.emb(indices) # size T x d_model   [T = len of sequence]

        if self.position_style == 'learned':
            # POSITIONAL ENCODING -- this will add E to the positional encoding
            V = self.positional_encoding(V)
#        if self.position_style in ['none', 'alibi']:
#            V = E
        if self.position_style == 'sinusoid':
            #P = torch.FloatTensor(getPositionEncoding(seq_len=len(indices), d=self.d_model, n=10000))
            #V = P+E
            V = self.sinusoid_encoding(V)

        # pass through transformer layer N times (once for now)
        attn_maps = []
        for ii in range(self.num_layers):
            attention, V = self.tl1(V)
            attn_maps.extend(attention)
            #attn_maps.append(attention)
#            attn_maps = [attention]

        # linear  and  softmax
        log_probs = self.log_softmax(self.W( V  )  )

#        assert log_probs.shape == (20,3)

#        for atm in attn_maps:
#            if atm.shape != (20,20):
#                print(atm.shape)
#                raise ValueError("attention map is wrong shape!")

        return log_probs, attn_maps

# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, d_value, d_ff, position_style, num_heads):
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
        self.position_style = position_style
        self.num_heads = num_heads

        #self.softmax = nn.Softmax(dim=1) # want columns to sum to 1
        self.softmax = nn.Softmax(dim=-1) # want columns to sum to 1

#        self.WQs = []
#        self.WKs = []
#        self.WVs = []
#        for ii in range(num_heads):
#        self.WQ = nn.init.xavier_uniform_(torch.empty(d_model, d_internal))
#        self.WK = nn.init.xavier_uniform_(torch.empty(d_model, d_internal))
#        self.WV = nn.init.xavier_uniform_(torch.empty(d_model, d_value))

#            self.WQs.append(nn.init.xavier_uniform_(torch.empty(d_model, d_internal)))
#            self.WKs.append(nn.init.xavier_uniform_(torch.empty(d_model, d_internal)))
#            self.WVs.append(nn.init.xavier_uniform_(torch.empty(d_model, d_value)))

        #NOTE need to put num_head dim first!
        self.WQs =  nn.init.xavier_uniform_(torch.empty(num_heads, d_model, d_internal))
        self.WKs =  nn.init.xavier_uniform_(torch.empty(num_heads, d_model, d_internal))
        self.WVs =  nn.init.xavier_uniform_(torch.empty(num_heads, d_model, d_value))

#        self.WO = nn.Linear(d_value, d_model)
        self.WO = nn.Linear(d_value*num_heads, d_model)

        self.FF1 = nn.Linear(d_model, d_ff)
        self.FF2 = nn.Linear(d_ff, d_model)
        self.g = nn.ReLU()


    def forward(self, input_vecs):
        #raise Exception("Implement me")
        seqlen = input_vecs.shape[0]
        #NOTE input_vecs is a matrix (T rows x d_model )

        zouts = []
        attentionmaps = []

        Qs = torch.matmul(input_vecs, self.WQs) # will multiply for each head.
        Ks = torch.matmul(input_vecs, self.WKs)
        Vs = torch.matmul(input_vecs, self.WVs)

        mask = torch.triu(torch.ones(seqlen, seqlen) * float('-inf'),  diagonal=1)

        # note: first dim is for heads, so need to transpose (1,2) instead of (0,1)
        scores = torch.matmul(Qs, torch.transpose(Ks,1,2)) # /np.sqrt(self.d_internal)


        if self.position_style == 'alibi':
            slopes = [ (2**(-8.0/self.num_heads))**i  for i in range(1,self.num_heads+1)]
            linear_biases = torch.FloatTensor([[[ -max(0,i-j)*slope   for j in range(seqlen)] for i in range(seqlen)] for slope in slopes])
            SA = self.softmax(mask + linear_biases + (scores/np.sqrt(self.d_internal) )  )
        else:
            SA = self.softmax(mask + scores/np.sqrt(self.d_internal) ) # this is numheads x len x len

#        for ii in range(self.num_heads):
#            Q = torch.matmul(input_vecs, self.WQs[ii])
#            K = torch.matmul(input_vecs, self.WKs[ii])
#            V = torch.matmul(input_vecs, self.WVs[ii])

#            mask = torch.triu(torch.ones(seqlen, seqlen) * float('-inf'),  diagonal=1)

            # self-attention
#            if self.position_style == 'alibi':
#                linear_bias = slopes[ii]*torch.FloatTensor([[-max(0,i-j)   for j in range(seqlen)] for i in range(seqlen)])
#                SA = self.softmax(mask + linear_bias + (torch.matmul(Q, torch.transpose(K,0,1)) )*1/np.sqrt(self.d_internal) )
#            else:
#                SA = self.softmax(mask + torch.matmul(Q, torch.transpose(K,0,1)) /np.sqrt(self.d_internal) )

#            attentionmaps.append(SA)
#            zouts.append(torch.matmul(SA, V))
        zouts = torch.matmul(SA, Vs)

        #NOTE there's got to be a better way to do this...
        zconcat = torch.hstack([zouts[i,:,:] for i in range(self.num_heads)])


#        zconcat = torch.hstack(zouts)
        attouts = input_vecs + self.WO(zconcat)



        # do weighted average, map back to dim d_model using welf.WO, and residual connection
        #attouts = input_vecs + self.WO( torch.matmul(SA, V) )

        #feed forward.. use same weight matrix for each position
        output_vecs = attouts  +  self.FF2( self.g( self.FF1( attouts  ) ))

        #input_vecs

        #output_vecs = attouts

        SA = [SA[i] for i in range(self.num_heads)]

        return SA, output_vecs


class SinusoidEncoding(nn.Module):
    def __init__(self, d_model: int, batched=False):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        input_size = x.shape[-2]
        P = torch.Tensor(getPositionEncoding(seq_len=len(x), d=self.d_model, n=10000))
        return x + P

# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev,
                     num_layers = 1,
                     position_style = 'none',
                     num_heads=8,
                     num_epochs=3):
    """
    train, dev : list of LetterCountingExample
    """
#    raise Exception("Not fully implemented yet")

    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:

    ##train = train[:2000]

    num_positions = len(train[0].input_tensor)
    #assert num_positions == 20

    #NOTE ideally don't hard-code this
    vocab_size = 27
    num_classes = 3 # predicting {0,1,2}

    #num_heads = 1
    d_model = 128 ##120 #PREV 100  experiment with it here..
    d_internal = 16 #PREV 20  # same as d_k
    num_layers = num_layers#1 # try 2 later

    learning_rate = 1e-4#5e-4#1e-4#1e-3 #1e-4#1e-3
    #position_style = position_style#'learned'#'none' #'learned' # ALSO: 'none', ''

    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, position_style, num_heads)

    #model.train()

    model.zero_grad()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #num_epochs = 30

    accs_over_epochs = []
    for t in range(0, num_epochs):
        model.train() #NOTE: need this because I used model.eval() below
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)

        loss_fcn = nn.NLLLoss()

        for ind in ex_idxs:

            ex = train[ind]
            model.zero_grad()

            # input of size 20
            log_probs, attn_maps = model.forward(ex.input_tensor)

#            loss = COMPUTE LOSS HERE
            loss = loss_fcn(log_probs, ex.output_tensor)

            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print("Training loss on EPOCH: "+str(t)+": ", loss_this_epoch)

        #SEE HOW FAST IT CONVERGES
        model.eval()
        with torch.no_grad():
            train_acc = decode(model, train)#dev)
            acc = decode(model, dev)
        accs_over_epochs.append(acc)
        print("TRAIN ACC for EPOCH: "+ str(t)+": ", train_acc)
        print("  DEV ACC for EPOCH: "+ str(t)+": ", acc)

    model.eval()
    return model, accs_over_epochs


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False, do_attention_normalization_test=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
        do_attention_normalization_test = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
#                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
#                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_xticks(np.arange(len(ex.input)))#, labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)))#, labels=ex.input)
                ax.set_xticklabels(ex.input)
                ax.set_yticklabels(ex.input)

                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        if do_attention_normalization_test:
            normalizes = attention_normalization_test(attn_maps)
            print("%s normalization test on attention maps" % ("Passed" if normalizes else "Failed"))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
#    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    return float(num_correct) / num_total

def attention_normalization_test(attn_maps):
    """
    Tests that the attention maps sum to one over rows
    :param attn_maps: the list of attention maps
    :return:
    """
    for attn_map in attn_maps:
        total_prob_over_rows = torch.sum(attn_map, dim=1)
        if torch.any(total_prob_over_rows < 0.99).item() or torch.any(total_prob_over_rows > 1.01).item():
            print("Failed normalization test: probabilities not sum to 1.0 over rows")
            print("Total probability over rows:", total_prob_over_rows)
            return False
    return True
