import pandas as pd
import re


def replace_user_and_subreddit(dataFrame, inputColumn: str, learn, data_config, model_config):
    series = pd.Series(dataFrame[inputColumn])
    user_regex = '/u/[A-Za-z0-9_-]+'
    subreddit_regex = '/r/[A-Za-z0-9_-]+'
    url_regex = '\[(.*?)\]\(.*?\)'
    result = series.map(lambda x: re.sub(subreddit_regex, '<subreddit>', re.sub(
        user_regex, '<user>', re.sub(url_regex, r'\1', x))))
    return pd.Series(result), None
