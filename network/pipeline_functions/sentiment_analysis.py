import stanza
import nltk.sentiment
import pandas

nltk.downloader.download(["vader_lexicon"])
stanza.download('en', processors='tokenize,sentiment')


def get_sentiment(dataFrame, inputColumn: str):
    sentiment_analyzer_stanza = stanza.Pipeline(
        'en', processors='tokenize,sentiment')
    sentiment_analyzer_nltk = nltk.sentiment.SentimentIntensityAnalyzer()
    sentiment_list = []
    for i, sentence in enumerate(dataFrame[inputColumn]):
        nltk_result = sentiment_analyzer_nltk.polarity_scores(sentence)[
            'compound']
        stanza_result = sentiment_analyzer_stanza(
            sentence).sentences[0].sentiment-1
        if nltk_result*stanza_result < 0:
            nltk_result = nltk_result/2
        sentiment_list.append(nltk_result)
        if i % 500 == 0:
            print("line:", i)
    return pandas.Series(sentiment_list)
