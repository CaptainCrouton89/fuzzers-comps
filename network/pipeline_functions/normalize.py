import scipy.stats
import pandas


def get_normal(dataFrame, inputColumn: str, learn=True):
    data_to_normalize = dataFrame[inputColumn]
    yj_data, _ = scipy.stats.yeojohnson(data_to_normalize)
    normalized_yj_data = (yj_data - yj_data.min()) / \
        (yj_data.max() - yj_data.min())
    return pandas.Series(normalized_yj_data)
