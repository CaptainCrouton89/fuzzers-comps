import data_pipeline
import gru_attention_network
import argparse
import os
from datetime import datetime
import sys
import json
import random
import logging
import torch
from pipeline_functions.sentiment_analysis import get_sentiment
from pipeline_functions.normalize import getNormal
import torch.nn as nn
from torch import optim

torch.manual_seed(1)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def create_network(config, vocab, pairs, category_indices, verbosity):
    """
    1. Create new columns using inp, out, func_list
    2. During batching, apply other normalization
    3. Onehot encode everything necessary
    """
    # Example for validation
    small_batch_size = 5
    batches = gru_attention_network.batch2TrainData(vocab, [random.choice(pairs)
                                                            for _ in range(small_batch_size)], category_indices)
    input_variable, lengths, target_variable, mask, max_target_len, meta_data = batches

    
    logging.debug(f"input_variable: {input_variable}", )
    logging.debug(f"lengths: {lengths}")
    logging.debug(f"target_variable: {target_variable}")
    logging.debug(f"mask: {mask}")
    logging.debug(f"max_target_len: {max_target_len}")
    logging.debug(f"meta_data: {meta_data}")

    # Configure models
    data_config = config["data"]
    model_config = config["model"]

    model_name = data_config['model_name']

    hidden_size = model_config['hidden_size']
    encoder_n_layers = model_config['encoder_n_layers']
    dropout = model_config['dropout']
    hidden_size = model_config['hidden_size']
    # decoder_hidden_size = hidden_size
    # decoder_hidden_size = hidden_size + 2 # We add 2 because the hidden layer now includes
    attn_model = model_config['attn_model']
    encoder_n_layers = model_config["encoder_n_layers"]
    decoder_n_layers = model_config["encoder_n_layers"]

    meta_data_size = len(category_indices["static_inputs"])

    # Configuring optimizer
    learning_rate = model_config["learning_rate"]
    decoder_learning_ratio = model_config["decoder_learning_ratio"]

    logging.info('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(vocab.num_words, hidden_size)

    # Initialize encoder & decoder models
    encoder = gru_attention_network.EncoderRNN(
        hidden_size, embedding, encoder_n_layers, dropout)
    decoder = gru_attention_network.LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, vocab.num_words, decoder_n_layers, dropout, meta_data_size)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    logging.info('Models built and ready to go!')

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    logging.info('Building optimizers ...')
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
    training_config = config["training"]
    corpus = data_config["corpus_name"]

    # Configure logger
    os.makedirs(os.path.join("logs", corpus), exist_ok=True)

    width = os.get_terminal_size().columns

    formatter = logging.Formatter('%(levelname)s: %(message)s \t\t@ %(filename)s: %(funcName)s: %(lineno)d', datefmt='%m/%d/%Y %I:%M:%S',)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(getattr(logging, args.loglevel.upper(), None))

    file_handler = logging.FileHandler(filename=f'logs/{corpus}/{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log', mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logging.basicConfig(
        handlers=[file_handler, stdout_handler])

    logging.info(f"Running pipeline with {args.config}")

    # Build file save path
    os.makedirs(data_config["network_save_path"], exist_ok=True)

    # Set function mapping
    function_mapping = [
        (get_sentiment, "parent_body", "sentiment_content", "static_inputs"),
        (getNormal, "delay", "delay", "static_inputs"),
    ]

    # Build data pairs
    vocab, pairs, category_indices = data_pipeline.load_prepare_data(
        data_config, function_mapping, use_processed=False)

    # Build network
    create_network(config, vocab, pairs, category_indices, args.verbose)


if __name__ == "__main__":
    main()
