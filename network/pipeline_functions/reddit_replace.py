import pandas as pd
import re

def replace_user_and_subreddit(dataFrame, inputColumn: str):
    series = pd.Series(dataFrame[inputColumn])
    user_regex = '/u/[A-Za-z0-9_-]+'
    subreddit_regex = '/r/[A-Za-z0-9_-]+'
    result = series.map(lambda x: re.sub(subreddit_regex, '<subreddit>', re.sub(user_regex, '<user>', x)))
    print(result.head())
    return pd.Series(result)


# check = pd.DataFrame({'text': ['neither', 'has a /u/user', 'has a /r/subreddit', '/u/has /r/both']})
# print(check.info())
# out = replace_user_and_subreddit(check, "text")
# print(out.info())
