from embedding_dictionary import embedding_dictionary, first_50_list
import scipy.stats
import numpy
import copy

weight_list = []
for _ in range(100):
    weight_list.append(0)
for item in first_50_list:
    for pos in range(100):
        weight_list[pos] += abs(item[pos])

weight_array = numpy.array(weight_list)
no_yj_filter_map = 1-((weight_array - weight_array.min()) /
                      (weight_array.max() - weight_array.min()))**2

dictionary_alt_scaling = copy.deepcopy(embedding_dictionary)

for key in embedding_dictionary:
    dictionary_alt_scaling[key] *= no_yj_filter_map

yj_data, _ = scipy.stats.yeojohnson(weight_list)
filter_map = numpy.array(1-((yj_data - yj_data.min()) /
                            (yj_data.max() - yj_data.min()))**2)

dictionary_pre_scaling = copy.deepcopy(embedding_dictionary)

for key in embedding_dictionary:
    embedding_dictionary[key] *= filter_map
