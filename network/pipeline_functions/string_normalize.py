import pandas
import unicodedata
import re


def get_normal_string(dataFrame, inputColumn: str):
    string_to_normalize = dataFrame[inputColumn]
    string_to_normalize = [normalize_one_string(
        s) for s in string_to_normalize]
    return pandas.Series(string_to_normalize)


def normalize_one_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?<>]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
