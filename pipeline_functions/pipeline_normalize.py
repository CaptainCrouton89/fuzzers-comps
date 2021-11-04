import scipy.stats
import pandas


def getNormal(dataFrame, inputColumn: str):
    data_to_normalize = dataFrame[inputColumn]
    yj_data, _ = scipy.stats.yeojohnson(data_to_normalize)
    normalized_yj_data = (yj_data - yj_data.min()) / \
        (yj_data.max() - yj_data.min())
    return pandas.Series(normalized_yj_data)
