import json

import pandas
import praw
import redditcleaner
import logging
import json
import os
import argparse
from sqlalchemy import null
import torch
from torch import nn
import sys
import testing
from custom_logger import init_logger
from gru_attention_network import EncoderRNN, LuongAttnDecoderRNN
from data_pipeline import Voc, filterPairs, preprocess
from function_mapping_handler import get_num_added_columns
from pipeline_functions.sentiment_analysis import get_sentiment_single_input
from pipeline_functions.string_normalize import normalize_one_string
from function_mapping_handler import apply_mappings

credentials = json.load(open("credentials.json", "r"))
reddit = praw.Reddit(
    client_id=credentials["client_id"],
    client_secret=credentials["client_secret"],
    user_agent=credentials["user_agent"],
    redirect_uri=credentials["redirect_uri"],
    refresh_token=credentials["refresh_token"]
)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def new_posts_from_subreddit_at_index(sub_name, i):
    # This assumes you have a global "reddit" object.
    # You may prefer to pass the "reddit" object in as a
    # parameter to this function.
    subreddit = reddit.subreddit(sub_name)
    # The default for the 'top' function is "top of all time".
    return list(subreddit.new())[i]


def get_best_comment_at_index(submission, i):
    # Set comment sort to best before retrieving comments
    submission.comment_sort = 'best'
    # Limit to, at most, 5 top level comments
    submission.comment_limit = i
    # Fetch the comments and print each comment body
    print(submission.comments)
    # This must be done _after_ the above lines or they won't take affect.
    return submission.comments[0]


def main(config=None, subreddit="pics"):
    parser = argparse.ArgumentParser(
        description='Enables testing of neural network.')
    parser.add_argument("-c", "--config",
                        help="config file for running model. Should correspond to model.",
                        default="configs/reddit.json")
    parser.add_argument("-l", "--loglevel",
                        help="Level at which to log events.",
                        default="INFO")
    args = parser.parse_args()

    if not config:
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

    init_logger(os.path.join("logs", "testing", corpus_name),
                args.loglevel, args.config)
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
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, 1, dropout, meta_data_size)  # We use batchsize of 1 since we are testing only one item

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

    # Get 2nd highest comment from 20th most recent post on subreddit
    comment = get_best_comment_at_index(
        new_posts_from_subreddit_at_index(subreddit, 40), 2)
    cleaned_comment = redditcleaner.clean(comment.body)
    print("Input:", cleaned_comment)

    # df = pandas.DataFrame(
    #     {"parent_body": [cleaned_comment], "body": ["body text that meets the requirements of having lots of characters?"], "parent_score": [0]})
    # df = preprocess(df, data_config)
    # added_cols = apply_mappings(df, config)
    # for col, cat in added_cols:
    #     data_config[cat].append(col)
    # pairs = df.to_numpy().tolist()
    # category_indices = {"encoder_inputs": [df.columns.get_loc(col_name) for col_name in data_config["encoder_inputs"]],
    #                     "target": [df.columns.get_loc(col_name) for col_name in data_config["target"]],
    #                     "static_inputs": [df.columns.get_loc(col_name) for col_name in data_config["static_inputs"]]
    #                     }
    # pairs = filterPairs(
    #     pairs, data_config["max_len"], category_indices["encoder_inputs"] + category_indices["target"])

    cleaned_comment = normalize_one_string(cleaned_comment)
    sent = get_sentiment_single_input(cleaned_comment)
    pair = [cleaned_comment, sent, 0.6]
    print(pair)

    response = testing.evaluate(
        searcher, voc, pair, max_length)

    print("Response:", response)
    # Post response
    response += "\n\nI am a bot trained using a recurrent neural network on many gigabytes of 2015 reddit comment history. If you have any questions or concerns, please shoot me a DM!"
    comment.reply(response)


if __name__ == "__main__":
    main()
