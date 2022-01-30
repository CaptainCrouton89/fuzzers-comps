import os
import sys
import json
import random
import logging
import argparse
import torch
import torch.nn as nn
from torch import optim
import data_pipeline
import gru_attention_network
from custom_logger import init_logger
from pipeline_functions.sentiment_analysis import get_sentiment
from pipeline_functions.reddit_replace import replace_user_and_subreddit
from pipeline_functions.normalize import get_normal
from pipeline_functions.string_normalize import get_normal_string

torch.manual_seed(1)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def create_network(config, vocab, pairs, category_indices):
    """
    Creates and trains the network using parameters from config.

    Keyword arguments:
    config -- the config dictionary
    vocab -- the voc object containing the entire vocabulary
    pairs -- all sentence comment-reply pairs
    category_indices -- an array of indices in the table at which to draw meta data values from

    Returns:
    null
    """

    # Configure models
    data_config = config["data"]
    model_config = config["model"]

    model_name = data_config['model_name']
    hidden_size = model_config['hidden_size']
    encoder_n_layers = model_config['encoder_n_layers']
    dropout = model_config['dropout']
    hidden_size = model_config['hidden_size']
    attn_model = model_config['attn_model']
    encoder_n_layers = model_config["encoder_n_layers"]
    decoder_n_layers = model_config["encoder_n_layers"]
    meta_data_size = len(category_indices["static_inputs"])

    # Configuring optimizer
    learning_rate = model_config["learning_rate"]
    decoder_learning_ratio = model_config["decoder_learning_ratio"]

    # Initialize encoder & decoder models
    # Initialize word embeddings
    logging.debug('Instantiating encoder, decoder, and embedding ...')
    embedding = nn.Embedding(vocab.num_words, hidden_size)
    encoder = gru_attention_network.EncoderRNN(
        hidden_size, embedding, encoder_n_layers, dropout)
    decoder = gru_attention_network.LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, vocab.num_words, decoder_n_layers, model_config["batch_size"], dropout, meta_data_size)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    logging.debug('Instantiating and initializing optimizers ...')
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
    logging.info("Starting Training!")
    gru_attention_network.trainIters(model_name, vocab, pairs, category_indices, encoder, decoder, encoder_optimizer, decoder_optimizer,
                                     embedding, config)
    return


def main():
    # Parse input args
    parser = argparse.ArgumentParser(
        description='Enables testing of neural network.')
    parser.add_argument("-c", "--config",
                        help="config file for network_deployable. Should correspond to model.")
    parser.add_argument("-l", "--loglevel",
                        help="Level at which to log events.",
                        default="INFO")
    args = parser.parse_args()

    # Get Configs
    with open(args.config) as f:
        config = json.load(f)
    data_config = config["data"]
    model_config = config["model"]
    corpus = data_config["corpus_name"]

    # initialize logger

    init_logger(os.path.join("logs", corpus),
                args.loglevel, config_path=config)

    # Build file save path
    os.makedirs(data_config["network_save_path"], exist_ok=True)

    # Build data pairs
    vocab, pairs, category_indices = data_pipeline.load_prepare_data(config, use_processed=False)

    # Build network
    create_network(config, vocab, pairs, category_indices)


if __name__ == "__main__":
    main()
