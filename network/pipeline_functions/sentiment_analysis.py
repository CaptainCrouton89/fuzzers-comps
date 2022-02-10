import nltk.sentiment
import pandas
nltk.downloader.download(["vader_lexicon"])

def get_sentiment(dataFrame, inputColumn: str, learn, data_config, model_config):
    sentiment_analyzer_nltk = nltk.sentiment.SentimentIntensityAnalyzer()
    sentiment_list = []
    for i, sentence in enumerate(dataFrame[inputColumn]):
        nltk_result = sentiment_analyzer_nltk.polarity_scores(sentence)[
            'compound']
        sentiment_list.append(nltk_result)
    return pandas.Series(sentiment_list), None
