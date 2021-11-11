import data_pipeline
import gru_attention_network
import argparse
import os
import json
import random
import torch
from pipeline_functions.sentiment_analysis import get_sentiment
import torch.nn as nn
from torch import optim

torch.manual_seed(1)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def create_network(config, vocab, pairs, verbosity):

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

    if verbosity > 0:
        print("input_variable:", input_variable)
        print("lengths:", lengths)
        print("target_variable:", target_variable)
        print("mask:", mask)
        print("max_target_len:", max_target_len)
        print("meta_data:", meta_data)

    # Configure models
    data_config = config["data"]
    model_config = config["model"]

    model_name = data_config['model_name']

    hidden_size = model_config['hidden_size']
    encoder_n_layers = model_config['encoder_n_layers']
    dropout = model_config['dropout']
    hidden_size = model_config['hidden_size']
    decoder_hidden_size = hidden_size 
    # decoder_hidden_size = hidden_size + 2 # We add 2 because the hidden layer now includes 
    attn_model = model_config['attn_model']
    encoder_n_layers = model_config["encoder_n_layers"]
    decoder_n_layers = model_config["encoder_n_layers"]

    # Configuring optimizer
    learning_rate = model_config["learning_rate"]
    decoder_learning_ratio = model_config["decoder_learning_ratio"]

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
                            help="config file for network_deployable. Should correspond to model.")
    parser.add_argument("-v", "--verbose", 
                            help="how much verbosity to include :)", 
                            action="count",
                            default=0)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    # Load sub-configuration dicts (narrower scope)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    # Build file save path
    if not os.path.exists(data_config["network_save_path"]):
        os.mkdir(data_config["network_save_path"])

    # Set function mapping
    function_mapping = [
        (get_sentiment, "content", "sentiment_content", "static_inputs"),
    ]

    # Build data pairs
    vocab, pairs = data_pipeline.loadPrepareData(data_config, function_mapping)

    # Print sample pairs
    if args.verbose > 0:
        print("\nsample pairs:")
        for pair in pairs[:5]:
            print(pair)

    # Build network
    create_network(config, vocab, pairs, args.verbose)

if __name__ == "__main__":
    main()
