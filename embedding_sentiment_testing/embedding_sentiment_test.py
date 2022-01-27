import nltk.sentiment
from embedding_to_vectors import *
import stanza
import numpy
# import difflib

nltk.download(["vader_lexicon"])
stanza.download('en', processors='tokenize')

sentiment_analyzer_nltk = nltk.sentiment.SentimentIntensityAnalyzer()
tokenizer = stanza.Pipeline('en', processors='tokenize')\

test_string_list = ["I am climbing a big mountain.",
                    "These are tall cliffs.", "I love this book.", "This book is my favorite."]

test_dictionary = {}
old_dictionary = {}
alt_dictionary = {}
nsq_dictionary = {}
pre_dictionary = {}
pre_nyj_dictionary = {}

# for sentence in test_string_list:
#     for comparison_sentence in test_string_list:
#         sim = 0
#         sentence_length = 0
#         for token in tokenizer(sentence).sentences[0].tokens:
#             if token.text in embedding_dictionary:
#                 sentence_length += 1
#                 current_best = -1
#                 for comp_token in tokenizer(comparison_sentence).sentences[0].tokens:
#                     if comp_token.text in embedding_dictionary:
#                         dot_product = numpy.dot(
#                             dictionary_pre_square_scaling[token.text], dictionary_pre_square_scaling[comp_token.text])
#                         norm_a = numpy.linalg.norm(
#                             dictionary_pre_square_scaling[token.text])
#                         norm_b = numpy.linalg.norm(
#                             dictionary_pre_square_scaling[comp_token.text])
#                         current_best = max(
#                             current_best, dot_product/(norm_a*norm_b))
#                 sim += current_best
#         if sentence_length > 0:
#             print("dist between \"", sentence, "\" and \"",
#                   comparison_sentence, "\" is", sim/sentence_length)

for sentence in test_string_list:
    sentence_length = 0
    sentence_value = numpy.zeros(100)
    old_sentence_value = numpy.zeros(100)
    alt_sentence_value = numpy.zeros(100)
    nsq_sentence_value = numpy.zeros(100)
    pre_sentence_value = numpy.zeros(100)
    pre_nyj_sentence_value = numpy.zeros(100)
    nltk_result = sentiment_analyzer_nltk.polarity_scores(sentence)[
        'compound']
    for token in tokenizer(sentence).sentences[0].tokens:
        # print(token.text)
        if token.text in embedding_dictionary:
            sentence_length += 1
            sentence_value += embedding_dictionary[token.text]
            old_sentence_value += dictionary_pre_scaling[token.text]
            alt_sentence_value += dictionary_alt_scaling[token.text]
            nsq_sentence_value += dictionary_no_square_scaling[token.text]
            pre_sentence_value += dictionary_pre_square_scaling[token.text]
            pre_nyj_sentence_value += dictionary_pre_sq_no_yj[token.text]
    if sentence_length > 0:
        sentence_value /= sentence_length
        old_sentence_value /= sentence_length
        alt_sentence_value /= sentence_length
        nsq_sentence_value /= sentence_length
        pre_sentence_value /= sentence_length
        pre_nyj_sentence_value /= sentence_length
    # print(sentence_value)
    test_dictionary[sentence] = sentence_value
    old_dictionary[sentence] = old_sentence_value
    alt_dictionary[sentence] = alt_sentence_value
    nsq_dictionary[sentence] = nsq_sentence_value
    pre_dictionary[sentence] = pre_sentence_value
    pre_nyj_dictionary[sentence] = pre_nyj_sentence_value

# for sentence in test_string_list:
#     for comparison_sentence in test_string_list:
#         dot_product = numpy.dot(
#             test_dictionary[sentence], test_dictionary[comparison_sentence])
#         norm_a = numpy.linalg.norm(test_dictionary[sentence])
#         norm_b = numpy.linalg.norm(test_dictionary[comparison_sentence])
#         print("dist between \"", sentence, "\" and \"",
#               comparison_sentence, "\" is", dot_product/(norm_a*norm_b))

# print("\nBEGIN DIFFLIB VERSION\n")

# for sentence in test_string_list:
#     for comparison_sentence in test_string_list:
#         sim = difflib.SequenceMatcher(
#             None, sentence, comparison_sentence).quick_ratio()
#         print("dist between \"", sentence, "\" and \"",
#               comparison_sentence, "\" is", sim)

print("\nBEGIN PRE-SQUARE VERSION\n")

for sentence in test_string_list:
    for comparison_sentence in test_string_list:
        if sentence < comparison_sentence:
            continue
        dot_product = numpy.dot(
            pre_dictionary[sentence], pre_dictionary[comparison_sentence])
        norm_a = numpy.linalg.norm(pre_dictionary[sentence])
        norm_b = numpy.linalg.norm(pre_dictionary[comparison_sentence])
        print("dist between \"", sentence, "\" and \"",
              comparison_sentence, "\" is", ((norm_a*norm_b)/dot_product)**2)

# print("\nBEGIN PRE-SQUARE NO-YJ VERSION\n")

# for sentence in test_string_list:
#     for comparison_sentence in test_string_list:
#         dot_product = numpy.dot(
#             pre_nyj_dictionary[sentence], pre_nyj_dictionary[comparison_sentence])
#         norm_a = numpy.linalg.norm(pre_nyj_dictionary[sentence])
#         norm_b = numpy.linalg.norm(pre_nyj_dictionary[comparison_sentence])
#         print("dist between \"", sentence, "\" and \"",
#               comparison_sentence, "\" is", dot_product/(norm_a*norm_b))

# print("\nBEGIN NO-SQUARE VERSION\n")

# for sentence in test_string_list:
#     for comparison_sentence in test_string_list:
#         dot_product = numpy.dot(
#             nsq_dictionary[sentence], nsq_dictionary[comparison_sentence])
#         norm_a = numpy.linalg.norm(nsq_dictionary[sentence])
#         norm_b = numpy.linalg.norm(nsq_dictionary[comparison_sentence])
#         print("dist between \"", sentence, "\" and \"",
#               comparison_sentence, "\" is", dot_product/(norm_a*norm_b))

# print("\nBEGIN OLD VERSION\n")

# for sentence in test_string_list:
#     for comparison_sentence in test_string_list:
#         dot_product = numpy.dot(
#             old_dictionary[sentence], old_dictionary[comparison_sentence])
#         norm_a = numpy.linalg.norm(old_dictionary[sentence])
#         norm_b = numpy.linalg.norm(old_dictionary[comparison_sentence])
#         print("dist between \"", sentence, "\" and \"",
#               comparison_sentence, "\" is", dot_product/(norm_a*norm_b))
