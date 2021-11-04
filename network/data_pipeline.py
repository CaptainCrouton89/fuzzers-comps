import pandas as pd
import unicodedata
import re
import pandas.api.types as ptypes

config = {
    "input_column": "content",
    "input_associated_columns": [
        "score",
        "thumbsUpCount"
    ],
    "response_column": "replyContent",
    "response_associated_columns": []
}

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

        print('keep_words {} / {} = {:.4f}'.format(
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
MAX_LENGTH = 50  # Maximum sentence length to consider
# TODO: read from config

def validate(df):
    assert ptypes.is_string_dtype(list(df)[0]), "Column 1 must be of type string, and should be input content"
    assert ptypes.is_string_dtype(list(df)[1]), "Column 2 must be of type string, and should be output content"

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?<>]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def readVocs(df, corpus_name):
    print("Reading lines...")
    pairs = df.to_numpy().tolist()
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    for i in range(len(p)):
        if len(p[i].split(' ')) >= MAX_LENGTH:
            return False
    return True

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# [[func, inp_col, out_col], [func2, inp_col, out_col]]

# Using the functions defined above, return a populated voc object and pairs list
# function_mapping is dict with format {"column_name": [map_func1, map_func2], column_name2...}
def loadPrepareData(corpus_name, data_path, function_mapping=[]):
    df = pd.read_json(data_path, orient="split")
    validate(df)
    for func, inp_col, out_col in function_mapping:
        df[out_col] = func(inp_col)
    print("Start preparing training data ...")
    voc, pairs = readVocs(df, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        # This definitely doesn't work, I think.
        # How should new data be added to vocab/ should it?
        # i.e. how to add star rating corresponding with sentence?
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs
