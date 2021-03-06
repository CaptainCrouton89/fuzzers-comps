# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# Following this guide: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
#

from __future__ import absolute_import, unicode_literals, print_function, division
import logging
import random
import itertools
import os
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_pipeline

torch.manual_seed(1)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# # Assembling Vocabulary, Formatting Input
# All text must be converted to numbers that can be embedded into vectors for the model."

# %%
# Vocabulary Class
# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
# APP_NAME_token = 3
# DIGITS_token = 4
# USERNAME_token = 5
# URL_TOKEN = 6
# EMAIL_token = 7

# Batching Data
# In order to take advantage of the GPU, we need to send data in batches. These batches need to be of same length, however, and our sentences are not of all the same length, so they need to get padded with extra space so they all take up the same size.

def indexesFromSentence(voc, sentence):
    """Returns an array of indices corresponding to tokens in an input string.

    Keyword arguments:
    voc -- the voc object corresponding to the vocabulary of our langauge
    sentence -- the input string to be indiced

    Returns:
    An array of indices
    """
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    """Returns a 2 dimensional array of indices corresponding to an array of sentences.

    Since all arrays must be of same length, it adds zeros to the ends of sentences
    that are not as long as the our max length.

    Keyword arguments:
    l -- the array of sentences
    voc -- the voc object corresponding to the vocabulary of our langauge

    Returns:
    A 2d array of indices
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    """

    Keyword arguments:
    l -- the array of sentences
    voc -- the voc object corresponding to the vocabulary of our langauge

    Returns:
    A 2d array of indices
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch, category_indices):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch, meta_data = [], [], []
    for pair in pair_batch:
        input_batch.extend([pair[i]
                           for i in category_indices["encoder_inputs"]])
        output_batch.extend([pair[i] for i in category_indices["target"]])
        meta_data.append([pair[i] for i in category_indices["static_inputs"]])

    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len, meta_data


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        packed = packed.to(device)
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(
                self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat(
            (hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    """Our attention decoder network. 

    See torch documentation on details. Our implementation follows the guide found at
    https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
    """
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, batchsize=64, dropout=0.1, meta_data_size=0):
        super(LuongAttnDecoderRNN, self).__init__()
        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size + meta_data_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batchsize = batchsize
        self.dropout = dropout
        self.meta_data_size = meta_data_size
        logging.debug(f"Meta data size: {self.meta_data_size}")

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        logging.debug(f"Decoder hidden size ={self.hidden_size}")
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # logging.debug(f"Hidden layer size: {last_hidden.size()}")
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        # logging.debug(f"Embedded layer size{embedded.size()}")
        # Maybe we add some zeros to the end of embedded
        if (self.meta_data_size > 0):
            embedded = torch.cat(
                (embedded, torch.zeros(1, self.batchsize, self.meta_data_size).to(device)), 2)
        # logging.debug(f"gru = {self.gru.input_size}, {self.gru.proj_size}, {self.gru.hidden_size}")
        # logging.debug(f"meta_data_size: {self.meta_data_size}")
        # logging.debug(f"embedded shape: {embedded.shape}")
        # logging.debug(f"last_hidden shape: {last_hidden.shape}")
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

def maskNLLLoss(inp, target, mask):
    """Loss function used by the network for variable length sentences.

    The network has to learn to match words to the target sentence, but also should
    not be penalized too harshly if it gives a good reply but is of a different total
    length. This function uses the mask array to "cancel out" the loss from mismatching
    sentence lengths.
    """

    nTotal = mask.sum()
    crossEntropy = - \
        torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

# Training Code
def train(input_variable, lengths, target_variable, mask, max_target_len, meta_data, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # We get the total amount of metadata that will be concatenated
    meta_data_size = len(meta_data[0])
    encoder_outputs = torch.cat(
        (encoder_outputs, torch.zeros(encoder_outputs.size()[0], batch_size, meta_data_size).to(device)), 2)
    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # logging.debug(f"hidden layer size before meta_data [seq_len, batch_size, features]: {encoder_hidden.size()}")
    
    # Concatonating other embeddings to hidden layer
    meta_data_tensor = torch.FloatTensor(
        [[meta_data_list for meta_data_list in meta_data] for _ in range(2*encoder.n_layers)]).to(device)

    # logging.debug(f"hidden layer size [seq_len, batch_size, features]: {meta_data_tensor.size()}")
    # logging.debug(f"meta_data: {meta_data}")
    # logging.debug(f"meta_data_tensor size: {meta_data_tensor.size()}")
    # logging.debug(f"meta_data_tensor: {meta_data_tensor}")
    # logging.debug(f"encoder_hidden size: {encoder_hidden.size()}")

    first_hidden = torch.cat((encoder_hidden, meta_data_tensor.to(device)), 2)
    # logging.debug(f"first_hidden size [seq_len, batch_size, features]: {first_hidden.size()}")

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = first_hidden[:decoder.n_layers]
    meta_data_tensor = meta_data_tensor[:decoder.n_layers]
    # Adjusted input for static variables
    # first_hidden = torch.cat((encoder_hidden, meta_data_tensor), 2)
    # decoder_hidden = first_hidden

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_hidden = torch.narrow(decoder_hidden, 2, 0, encoder.hidden_size)
            decoder_hidden = torch.cat((decoder_hidden, meta_data_tensor), 2)
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(
                decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_hidden = torch.narrow(decoder_hidden, 2, 0, encoder.hidden_size)
            decoder_hidden = torch.cat((decoder_hidden, meta_data_tensor), 2)
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(
                decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


# %%
def trainIters(model_name, voc, pairs, category_indices, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, config, loadFilename=None, checkpoint=None):
    """Iteratively trains the network, saving at preconfigured checkpoints."""

    min_loss = np.inf
    iter_since_min_loss = 0

    # Set up sub-configs
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    # Initialize from config
    encoder_n_layers = model_config["encoder_n_layers"]
    decoder_n_layers = model_config["decoder_n_layers"]
    hidden_size = model_config["hidden_size"]
    clip = model_config["clip"]
    teacher_forcing_ratio = model_config["teacher_forcing_ratio"]
    batch_size = model_config["batch_size"]

    network_save_path = data_config["network_save_path"]
    corpus_name = data_config["corpus_name"]

    n_iteration = training_config["n_iteration"]
    print_every = training_config["print_every"]
    save_every = training_config["save_every"]

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)], category_indices)
                        for _ in range(n_iteration)]

    # Initializations
    logging.info('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    logging.info("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len, meta_data = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, meta_data, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio)
        print_loss += loss

        if loss < min_loss:
            min_loss = loss
            iter_since_min_loss = 0
        else:
            iter_since_min_loss += 1

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            logging.info("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0 or iter_since_min_loss > training_config["learning_stop_count"] or iteration == n_iteration):
            directory = os.path.join(network_save_path, model_name, corpus_name, '{}-{}_{}'.format(
                encoder_n_layers, decoder_n_layers, hidden_size+len(meta_data[0])))
            save_path = os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint'))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'encoder_opt': encoder_optimizer.state_dict(),
                'decoder_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, save_path)
            logging.info("Saving at " + save_path)
            if iter_since_min_loss > training_config["learning_stop_count"]:
                return
