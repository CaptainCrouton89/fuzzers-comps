import data_pipeline
import gru_attention_network
import argparse
import json
import random
import torch
import torch.nn as nn
from torch import optim

torch.manual_seed(1)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def call_data_pipeline(config):
    corpus_name = config["corpus_name"]
    data_path = config["data_path"]
    vocab, pairs = data_pipeline.loadPrepareData(corpus_name, data_path)

    print("\nsample pairs:")
    for pair in pairs[:5]:
        print(pair)

    return vocab, pairs

def create_network(config, vocab, pairs):

    """
    1. Create new columns using inp, out, func_list
    2. During batching, apply other normalization
    3. Onehot encode everything necessary
    """
    # Example for validation
    small_batch_size = 5
    batches = gru_attention_network.batch2TrainData(vocab, [random.choice(pairs)
                                    for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len, meta_data = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)
    print("meta_data:", meta_data)

    # Configure models
    hidden_size = config['hidden_size']
    encoder_n_layers = config['encoder_n_layers']
    dropout = config['dropout']
    hidden_size = config['hidden_size']
    decoder_hidden_size = hidden_size + 2 # We add 2 because the hidden layer now includes 
    model_name = config['model_name']
    attn_model = config['attn_model']
    encoder_n_layers = config["encoder_n_layers"]
    decoder_n_layers = config["encoder_n_layers"]

    # Configuring optimizer
    learning_rate = config["learning_rate"]
    decoder_learning_ratio = config["decoder_learning_ratio"]

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(vocab.num_words, hidden_size)

    # Initialize encoder & decoder models
    encoder = gru_attention_network.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = gru_attention_network.LuongAttnDecoderRNN(
        attn_model, embedding, decoder_hidden_size, vocab.num_words, decoder_n_layers, dropout)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # If you have cuda, configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Run training iterations
    print("Starting Training!")
    gru_attention_network.trainIters(model_name, vocab, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, config)
    return

def main():
    parser = argparse.ArgumentParser(
            description='Enables testing of neural network.') 
    parser.add_argument("-c", "--config", 
                            help="config file for network_deployable. Should correspond to model.", 
                            default="configs/config_basic.json")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    vocab, pairs = call_data_pipeline(config)
    create_network(config, vocab, pairs)

if __name__ == "__main__":
    main()
