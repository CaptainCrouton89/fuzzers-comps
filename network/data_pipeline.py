import logging
import pandas as pd
import unicodedata
import re
import warnings
import os
import torch
import json

from function_mapping_handler import apply_mappings

# %% [markdown]
# # Assembling Vocabulary, Formatting Input
# All text must be converted to numbers that can be embedded into vectors for the model."


# %%
# Vocabulary Class
# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
APP_NAME_token = 3
DIGITS_token = 4
USERNAME_token = 5
URL_TOKEN = 6
EMAIL_token = 7


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS",
                           APP_NAME_token: "ANT", DIGITS_token: "DGT", USERNAME_token: "UNT",
                           URL_TOKEN: "URT", EMAIL_token: "EMT"}
        # <appname>, <digits>, <username>, <url>, <email>
        self.num_words = 8  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        logging.info('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(
                keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS",
                           APP_NAME_token: "ANT", DIGITS_token: "DGT", USERNAME_token: "UNT",
                           URL_TOKEN: "URT", EMAIL_token: "EMT"}
        # <appname>, <digits>, <username>, <url>, <email>
        self.num_words = 8  # Count default tokens

        for word in keep_words:
            self.addWord(word)

# %% [markdown]
# ## Assembling Vocabulary and Formatting Pairs


# %%
def validate(df):
    return
    # This shit needs to get significantly changed.
    assert df.dtypes[0] == "object", "Column 1 must be of type string, and should be input content"
    assert df.dtypes[1] == "object", "Column 2 must be of type string, and should be output content"
    content = list(df)[0]
    replyContent = list(df)[1]
    if content != "content":
        warnings.warn("First column in dataframe should be input_text. \
The current column is named {}".format(content))
    if replyContent != "replyContent":
        warnings.warn("Second column in dataframe should be target text. \
It should be named `replyContent`. The current column is named {}".format(replyContent))


def preprocess(df, data_config):
    df = df[data_config["encoder_inputs"] +
            data_config["target"] + data_config["static_inputs"]]
    return df

# Returns True if both sentences in a pair 'p' are under the max_len threshold
def filterPair(p, max_len, indices):
    # Input sequences need to preserve the last word for EOS token
    for index in indices:
        if len(p[index].split(' ')) >= max_len:
            return False
    return True

# Filter pairs using filterPair condition


def filterPairs(pairs, max_len, indices):
    return [pair for pair in pairs if filterPair(pair, max_len, indices)]

# [[func, inp_col, out_col], [func2, inp_col, out_col]]

# Using the functions defined above, return a populated voc object and pairs list
# function_mapping is dict with format {"column_name": [map_func1, map_func2], column_name2...}


def load_prepare_data(config, function_mapping=[], use_processed=True):
    data_config = config['data']
    logging.info("Start preparing training data ...")

    format = data_config["data_format"]
    path = data_config["data_path"]
    if use_processed:
        path.replace(f".{format}", f"_processed.{format}")

    if format == "ft":
        df = pd.read_feather(path)
    elif format == "json":
        df = pd.read_json(path, orient="split")

    df = preprocess(df, data_config)
    validate(df)

    # Add additional columns, if necessary
    if not use_processed:
        apply_mappings(df, config)
        # for func, inp_col, out_col, category in function_mapping:
        #     df[out_col], constants_to_save = func(df, inp_col)
        #     if inp_col != out_col:
        #         data_config[category].append(out_col)
        #     if constants_to_save != None:
        #         directory = os.path.join(data_config["network_save_path"], data_config["model_name"], data_config["corpus_name"], '{}-{}_{}'.format(
        #             model_config["encoder_n_layers"], model_config["decoder_n_layers"], model_config["hidden_size"]+len(data_config["static_inputs"])), func.__name__)
        #         save_path = os.path.join(
        #             directory, '{}.json'.format(inp_col))
        #         if not os.path.exists(directory):
        #             os.makedirs(directory)
        #         with open(save_path, 'w') as out_file:
        #             json.dump(constants_to_save, out_file)

        # Save file to <path>_processed for future use
        path = path.replace(f".{format}", f"_processed.{format}")
        if format == "ft":
            df.to_feather(path)
        elif format == "json":
            df.to_json(path, orient="split")

    pairs = df.to_numpy().tolist()
    logging.info("Read {!s} sentence pairs".format(len(pairs)))

    category_indices = {"encoder_inputs": [df.columns.get_loc(col_name) for col_name in data_config["encoder_inputs"]],
                        "target": [df.columns.get_loc(col_name) for col_name in data_config["target"]],
                        "static_inputs": [df.columns.get_loc(col_name) for col_name in data_config["static_inputs"]]
                        }

    pairs = filterPairs(
        pairs, data_config["max_len"], category_indices["encoder_inputs"] + category_indices["target"])
    logging.info("Trimmed to {!s} sentence pairs".format(len(pairs)))

    logging.info(f"\n{df.head()}")
    logging.debug(f"category_indices: {str(category_indices)}")

    # Building vocabulary
    logging.info("Counting words...")
    voc = Voc(data_config["corpus_name"])
    # Add all text from input and output columns
    for pair in pairs:
        # Likely only a single input column: `content`
        for col in [df.columns.get_loc(col_name) for col_name in data_config["encoder_inputs"]]:
            voc.addSentence(pair[col])
        # Likely only a single output column: `replyContent`
        for col in [df.columns.get_loc(col_name) for col_name in data_config["target"]]:
            voc.addSentence(pair[col])
    logging.info(f"Counted words: {voc.num_words}")
    return voc, pairs, category_indices
