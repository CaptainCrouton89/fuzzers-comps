import logging
import pandas as pd
import os
import json
import random
from utils import get_model_path

from function_mapping_handler import apply_mappings
from function_mapping_handler import get_num_added_columns

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        # <appname>, <digits>, <username>, <url>, <email>
        self.num_words = 3  # Count SOS, EOS, PAD

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
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        # <appname>, <digits>, <username>, <url>, <email>
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)

'''
Trim words used under the MIN_COUNT from the voc,
and remove pairs containing those words.
'''
def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    logging.info("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

'''
Filters out columns of the data frame that are not 
relevant to the model.
'''
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

def filterPairs(pairs, max_len, indices):
    return [pair for pair in pairs if filterPair(pair, max_len, indices)]

# Using the functions defined above, return a populated voc object and pairs list
'''
Loads, prepares, (and processes) data as per the config.
Pairs is a misnomer - a single 'pair' includes input, output, and metadata.
returns: voc, pairs, category_indices
'''
def load_prepare_data(config, use_processed=True):
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

    # Process data and add additional columns, if necessary
    if not use_processed:
        # Mappings are defined in function_mapping_handler.py
        added_cols = apply_mappings(df, config)
        for col, cat in added_cols:
            data_config[cat].append(col)
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
        # Likely only a single input column: `parent_body`
        for col in [df.columns.get_loc(col_name) for col_name in data_config["encoder_inputs"]]:
            voc.addSentence(pair[col])
        # Likely only a single output column: `body`
        for col in [df.columns.get_loc(col_name) for col_name in data_config["target"]]:
            voc.addSentence(pair[col])
    logging.info(f"Pre-trim counted words: {voc.num_words}")
    all_words_counts = list(voc.word2count.values())
    all_words_counts.sort()
    min_count = all_words_counts[-10000] if len(all_words_counts)>10000 else 3
    pairs = trimRareWords(voc, pairs, min_count)

    count = len(pairs)
    random.shuffle(pairs)
    train = pairs[:int(count * .9)]
    test = pairs[int(count * .9):]

    model_path = os.path.join(get_model_path(config, False), "test_data.json")

    json.dump(test, open(model_path, 'w'))

    logging.info(f"Post-trim counted words: {voc.num_words}")
    return voc, train, category_indices
