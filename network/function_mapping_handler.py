from torch import functional
from pipeline_functions.sentiment_analysis import get_sentiment
from pipeline_functions.reddit_replace import replace_user_and_subreddit
from pipeline_functions.normalize import get_normal
from pipeline_functions.string_normalize import get_normal_string
import os
import json
import logging

function_mapping = [
    (replace_user_and_subreddit, "parent_body", "parent_body", "encoder_inputs"),
    (replace_user_and_subreddit, "body", "body", "target"),
    (get_sentiment, "parent_body", "sentiment_content", "static_inputs"),
    (get_normal, "delay", "delay", "static_inputs"),
    (get_normal, "gilded", "gilded", "static_inputs"),
    (get_normal_string, "body", "body", "target"),
    (get_normal_string, "parent_body", "parent_body", "encoder_inputs"),
    (get_normal, "score", "score", "static_inputs"),
    (get_normal, "parent_score", "parent_score", "static_inputs"),
    (get_normal_string, "content", "content", "encoder_inputs"),
    (get_sentiment, "content", "sentiment_content", "static_inputs"),
    (get_normal, "thumbsUpContent", "thumbsUpContent", "static_inputs"),
    (get_normal, "score", "score", "static_inputs"),
    (get_normal_string, "replyContent", "replyContent", "target")
]

'''
Returns list of added categories
'''
def apply_mappings(df, config):
    data_config = config['data']
    model_config = config['model']
    function_indices = data_config['function_indices']
    allowed_functions = [function_mapping[i] for i in function_indices]
    logging.debug(f"allowed_functions: {allowed_functions}")
    new_cols = []
    for func, inp_col, out_col, category in allowed_functions:
        df[out_col], constants_to_save = func(df, inp_col, True, data_config, model_config)
        if inp_col != out_col:
            new_cols.append((out_col, category))
        if constants_to_save != None:
            directory = os.path.join(data_config["network_save_path"], data_config["model_name"], data_config["corpus_name"], '{}-{}_{}'.format(
                model_config["encoder_n_layers"], model_config["decoder_n_layers"], model_config["hidden_size"]+len(data_config["static_inputs"])+get_num_added_columns(config)), func.__name__)
            save_path = os.path.join(
                directory, '{}.json'.format(inp_col))
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(save_path, 'w') as out_file:
                json.dump(constants_to_save, out_file)
    return new_cols



'''
Returns list of added categories
'''
def apply_mappings_testing(df, config):
    data_config = config['data']
    model_config = config['model']
    function_indices = data_config['function_indices']
    allowed_functions = [function_mapping[i] for i in function_indices]
    new_cols = []
    for func, inp_col, out_col, category in allowed_functions:
        if inp_col == 'body':
            continue
        df[out_col], constants_to_save = func(df, inp_col, False, data_config, model_config)
        logging.debug(f"Just ran function {func}")
        if inp_col != out_col:
            new_cols.append((out_col, category))
    return new_cols


'''
Returns number of added categories
'''
def get_num_added_columns(config):
    function_indices = config['data']['function_indices']
    allowed_functions = [function_mapping[i] for i in function_indices]
    count = 0
    for func, inp_col, out_col, category in allowed_functions:
        if inp_col != out_col:
            count += 1
    return count
