import logging
import json
import os
import argparse
import torch
from torch import nn
from custom_logger import init_logger
from gru_attention_network import EncoderRNN, LuongAttnDecoderRNN, indexesFromSentence, SOS_token, EOS_token, batch2TrainData
from data_pipeline import Voc
from pipeline_functions.string_normalize import normalize_one_string
import random
import pandas as pd
from function_mapping_handler import apply_mappings_testing, get_num_added_columns
import testing
import data_pipeline
import gru_attention_network
from utils import get_model_path

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def main():
    parser = argparse.ArgumentParser(
        description='Enables testing of neural network.')
    parser.add_argument("-c", "--config",
                        help="config file for running model. Should correspond to model.",
                        default="configs/reddit.json")
    parser.add_argument("-l", "--loglevel",
                        help="Level at which to log events.",
                        default="INFO")
    args = parser.parse_args()

    with open(str(args.config)) as f:
        config = json.load(f)

    test_config = config['testing']
    data_config = config['data']
    model_config = config['model']

    network_save_path = data_config['network_save_path']
    corpus_name = data_config['corpus_name']
    model_name = data_config['model_name']
    static_inputs = data_config['static_inputs']
    encoder_inputs = data_config['encoder_inputs']
    max_length = data_config['max_len']

    meta_data_size = len(static_inputs)
    meta_data_size += get_num_added_columns(data_config)


    checkpoint = test_config['checkpoint']
    top_n = test_config['top_n']
    threshold = test_config['threshold']

    encoder_n_layers = model_config['encoder_n_layers']
    dropout = model_config['dropout']
    attn_model = model_config['attn_model']
    decoder_n_layers = model_config['decoder_n_layers']
    hidden_size = model_config['hidden_size']

    model_features = str(encoder_n_layers) + "-" + \
        str(decoder_n_layers) + "_" + str(hidden_size+meta_data_size)
    model_path = os.path.join(
        network_save_path, model_name, corpus_name, model_features, checkpoint)


    init_logger(os.path.join("logs", "testing", corpus_name), args.loglevel, args.config)
    logging.debug(f"Using model at {model_path}")
    # If loading on same machine the model was trained on
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    voc = Voc(corpus_name)
    encoder_sd = checkpoint['encoder']
    decoder_sd = checkpoint['decoder']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

    # encoder_optimizer_sd = checkpoint['encoder_opt']
    # decoder_optimizer_sd = checkpoint['decoder_opt']
    # I don't think these are needed but i'm leaving them here commented out just in case.

    embedding = nn.Embedding(voc.num_words, hidden_size)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, 1, dropout, meta_data_size) # We use batchsize of 1 since we are testing only one item

    embedding.load_state_dict(embedding_sd)
    encoder.load_state_dict(encoder_sd, strict=False)
    decoder.load_state_dict(decoder_sd, strict=False)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = testing.GreedySearchDecoder(encoder, decoder, top_n, threshold)

    pairs = json.load(open(os.path.join(get_model_path(config, True), "test_data.json"), "r"))

    f1 = open(os.path.join(get_model_path(config, True), "responses.txt"), "w")
    f2 = open(os.path.join(get_model_path(config, True), "responsesWithInput.txt"), "w")

    real_response = [pair[1] for pair in pairs]
    pairs = [pair[0:1] + pair[2:] for pair in pairs]

    for i, pair in enumerate(pairs):
        try:
            response = testing.evaluate(searcher, voc, pair, max_length)

            f2.write("input: " + str(pair[0]) + "\n")
            f1.write("response: " + response + "\n")
            f2.write("response: " + response + "\n")
        except KeyError as e:
            print(f"Key {e} not found")
    f1.close()
    f2.close()

if __name__ == "__main__":
    main()
