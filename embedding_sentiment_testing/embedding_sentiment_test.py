import nltk.sentiment
from embedding_to_vectors import embedding_dictionary, dictionary_pre_scaling, dictionary_alt_scaling
import stanza
import numpy

nltk.download(["vader_lexicon"])
stanza.download('en', processors='tokenize')

sentiment_analyzer_nltk = nltk.sentiment.SentimentIntensityAnalyzer()
tokenizer = stanza.Pipeline('en', processors='tokenize')

test_string_list = ["I am climbing a big mountain.",
                    "These are tall cliffs.", "Reading books is fun.", "I enjoy telling stories."]

test_dictionary = {}
old_dictionary = {}
alt_dictionary = {}

for sentence in test_string_list:
    sentence_length = 0
    sentence_value = numpy.zeros(100)
    old_sentence_value = numpy.zeros(100)
    alt_sentence_value = numpy.zeros(100)
    nltk_result = sentiment_analyzer_nltk.polarity_scores(sentence)[
        'compound']
    for token in tokenizer(sentence).sentences[0].tokens:
        # print(token.text)
        if token.text in embedding_dictionary:
            sentence_length += 1
            sentence_value += embedding_dictionary[token.text]
            old_sentence_value += dictionary_pre_scaling[token.text]
            alt_sentence_value += dictionary_alt_scaling[token.text]
    if sentence_length > 0:
        sentence_value /= sentence_length
        old_sentence_value /= sentence_length
        alt_sentence_value /= sentence_length
    # print(sentence_value)
    test_dictionary[sentence] = sentence_value
    old_dictionary[sentence] = old_sentence_value
    alt_dictionary[sentence] = alt_sentence_value

for sentence in test_string_list:
    for comparison_sentence in test_string_list:
        dot_product = numpy.dot(
            test_dictionary[sentence], test_dictionary[comparison_sentence])
        norm_a = numpy.linalg.norm(test_dictionary[sentence])
        norm_b = numpy.linalg.norm(test_dictionary[comparison_sentence])
        print("dist between \"", sentence, "\" and \"",
              comparison_sentence, "\" is", dot_product/(norm_a*norm_b))

print("\nBEGIN OLD VERSION\n")

for sentence in test_string_list:
    for comparison_sentence in test_string_list:
        dot_product = numpy.dot(
            old_dictionary[sentence], old_dictionary[comparison_sentence])
        norm_a = numpy.linalg.norm(old_dictionary[sentence])
        norm_b = numpy.linalg.norm(old_dictionary[comparison_sentence])
        print("dist between \"", sentence, "\" and \"",
              comparison_sentence, "\" is", dot_product/(norm_a*norm_b))

print("\nBEGIN ALT VERSION\n")

for sentence in test_string_list:
    for comparison_sentence in test_string_list:
        dot_product = numpy.dot(
            alt_dictionary[sentence], alt_dictionary[comparison_sentence])
        norm_a = numpy.linalg.norm(alt_dictionary[sentence])
        norm_b = numpy.linalg.norm(alt_dictionary[comparison_sentence])
        print("dist between \"", sentence, "\" and \"",
              comparison_sentence, "\" is", dot_product/(norm_a*norm_b))
