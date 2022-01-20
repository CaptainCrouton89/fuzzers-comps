from pipeline_functions.sentiment_analysis import get_sentiment
from pipeline_functions.reddit_replace import replace_user_and_subreddit
from pipeline_functions.normalize import get_normal
from pipeline_functions.string_normalize import get_normal_string

function_mapping = [
    (replace_user_and_subreddit, "parent_body", "parent_body", "encoder_inputs"),
    (replace_user_and_subreddit, "body", "body", "target"),
    (get_sentiment, "parent_body", "sentiment_content", "static_inputs"),
    (get_normal, "delay", "delay", "static_inputs"),
    (get_normal, "gilded", "gilded", "static_inputs"),
    (get_normal_string, "body", "body", "target"),
    (get_normal_string, "parent_body", "parent_body", "encoder_inputs")
]

'''
Returns list of added categories
'''
def apply_mappings(df):
    new_cols = []
    for func, inp_col, out_col, category in function_mapping:
        df[out_col] = func(df, inp_col)
        if inp_col != out_col:
            new_cols.append((out_col, category))
    return new_cols

'''
Returns list of added categories
'''
def apply_mappings_testing(df):
    new_cols = []
    for func, inp_col, out_col, category in function_mapping:
        if inp_col == 'body':
            continue
        df[out_col] = func(df, inp_col, False)
        if inp_col != out_col:
            new_cols.append((out_col, category))
    return new_cols
