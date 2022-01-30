import scipy.stats
import pandas
import math
import os
import logging
import json

def get_normal(dataFrame, inputColumn: str, learn, data_config, model_config):
    if learn:
        series, constants = get_normal_series(dataFrame, inputColumn, data_config, model_config)
        return series, constants
    else:
        return get_normal_single(dataFrame, inputColumn, data_config, model_config), None

def get_normal_series(dataFrame, inputColumn: str, data_config, model_config):
    data_to_normalize = dataFrame[inputColumn]
    yj_data, lmbda = scipy.stats.yeojohnson(data_to_normalize)
    normalized_yj_data = (yj_data - yj_data.min()) / \
        (yj_data.max() - yj_data.min())
    return pandas.Series(normalized_yj_data), (lmbda, yj_data.min(), yj_data.max())


# def get_normal(lmbda, min, max, input):
#     pre_scale_value = input
#     if input >= 0:
#         if lmbda == 0:
#             pre_scale_value = ((input + 1)**lmbda - 1) / lmbda
#         else:
#             pre_scale_value = math.log(input + 1)
#     else:
#         if lmbda == 2:
#             pre_scale_value = -((-input + 1)**(2 - lmbda) - 1) / (2 - lmbda)
#         else:
#             pre_scale_value = -math.log(-input + 1)
#     value = (pre_scale_value - min) / (max - min)
#     return value

def get_normal_single(dataFrame, inputColumn, data_config, model_config):
    from function_mapping_handler import get_num_added_columns
    directory = os.path.join(data_config["network_save_path"], data_config["model_name"], data_config["corpus_name"], '{}-{}_{}'.format(
        model_config["encoder_n_layers"], model_config["decoder_n_layers"], model_config["hidden_size"]+len(data_config["static_inputs"])+get_num_added_columns(data_config)), 'get_normal')
    save_path = os.path.join(directory, '{}.json'.format(inputColumn))
    if not os.path.exists(directory):
        logging.ERROR("normalize data non existent.")
    with open(save_path, 'r') as f:
        constants = json.load(f)

    lmbda, min, max = constants
    input = int(dataFrame[inputColumn][0])
    if input >= 0:
        if lmbda == 0:
            return (((input + 1)**lmbda - 1) / lmbda - min) / (max - min)
        return (math.log(input + 1) - min) / (max - min)
    if lmbda == 2:
        return (-((-input + 1)**(2 - lmbda) - 1) / (2 - lmbda) - min) / (max - min)
    return (-math.log(-input + 1) - min) / (max - min)
