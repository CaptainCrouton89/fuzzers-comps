import pandas as pd
import unicodedata
import re
import warnings

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
def validate(df):
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

# Returns True if both sentences in a pair 'p' are under the max_len threshold
def filterPair(p, max_len):
    # Input sequences need to preserve the last word for EOS token
    for i in range(len(p)):
        if len(p[i].split(' ')) >= max_len:
            return False
    return True

# Filter pairs using filterPair condition
def filterPairs(pairs, max_len):
    return [pair for pair in pairs if filterPair(pair, max_len)]

# [[func, inp_col, out_col], [func2, inp_col, out_col]]

# Using the functions defined above, return a populated voc object and pairs list
# function_mapping is dict with format {"column_name": [map_func1, map_func2], column_name2...}
def loadPrepareData(corpus_name, data_path, max_len, function_mapping=[]):
    df = pd.read_json(data_path, orient="split")
    validate(df)
    for func, inp_col, out_col in function_mapping:
        df[out_col] = func(inp_col)
    print("Start preparing training data ...")
    voc, pairs = readVocs(df, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs, max_len)
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
