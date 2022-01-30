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
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = testing.GreedySearchDecoder(encoder, decoder, top_n, threshold)

    _, pairs, category_indices = data_pipeline.load_prepare_data(config, use_processed=False)
    pairs = random.sample(pairs, len(pairs)//50)

    pairs = [pair[0:1] + pair[2:] for pair in pairs]

    f = open("mass_test_output.txt", "w+")
    for pair in pairs:
        try:
            output_words = testing.evaluate(encoder, decoder, searcher, voc, pair, max_length)

            output_words[:] = [x for x in output_words if not (
                        x == 'EOS' or x == 'PAD')]
            f.write(' '.join(output_words) + "\n")
        except KeyError as e:
            print(f"Key {e} not found")
    f.close()

if __name__ == "__main__":
    main()