import scipy.stats
import pandas
import math


def get_normal(dataFrame, inputColumn: str, learn=True):
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


def get_normal_single(constants, input):
    lmbda, min, max = constants
    if input >= 0:
        if lmbda == 0:
            return (((input + 1)**lmbda - 1) / lmbda - min) / (max - min)
        return (math.log(input + 1) - min) / (max - min)
    if lmbda == 2:
        return (-((-input + 1)**(2 - lmbda) - 1) / (2 - lmbda) - min) / (max - min)
    return (-math.log(-input + 1) - min) / (max - min)
