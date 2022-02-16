import pandas
import unicodedata
import re

from torch import norm


def get_normal_string(dataFrame, inputColumn: str, learn, data_config, model_config):
    string_to_normalize = dataFrame[inputColumn]
    string_to_normalize = [normalize_one_string(
        s) for s in string_to_normalize]
    return pandas.Series(string_to_normalize), None


def normalize_one_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", r' <url> ', s)
    s = re.sub(r"([.!?])\1+", r" \1 ", s)
    s = re.sub(r"([.!?])", r" \1 ", s) # no, this isn't redundant 
    s = re.sub(r"(\'s)", r" \1 ", s)
    s = re.sub(r"(n\'t)", r" \1 ", s)
    s = re.sub(r"(\'ll)", r" \1 ", s)
    s = re.sub(r"(\'re)", r" \1 ", s)
    s = re.sub(r"(\'d)", r" \1 ", s)
    s = re.sub(r"(\'ve)", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z.!?><']+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

if __name__ == '__main__':
    strings = [
        "Hi.",
        "he.llo.",
        "Hi there :) I hope you're doing gr8 my dude!!",
        "Hi. My name is james.",
        "The dude's stuff was taken",
        "Wouldn't it be nice if they'd help they'll you're couldn't won't bobby's thing's stuff",
        ":Lk9827)*#&$T(!2l3j o2 p2oi3 hoPO*#H ;2j4i3 ;",
        "Check my mom out @ www.thiccmoms.com ",
        "I !!!! love !!! my !!!??? punctuation ...???!!!"
    ]

    for string in strings:
        print(normalize_one_string(string))
