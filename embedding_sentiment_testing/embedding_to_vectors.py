from embedding_dictionary import embedding_dictionary, first_50_list
import scipy.stats
import numpy
import copy

weight_list = numpy.zeros(100)
sqr_weights = numpy.zeros(100)
for item in first_50_list:
    for pos in range(100):
        weight_list[pos] += abs(item[pos])
        sqr_weights[pos] += abs(item[pos])**2

no_yj_filter_map = 1-((weight_list - weight_list.min()) /
                      (weight_list.max() - weight_list.min()))**2

pre_sq_no_yj_filter_map = 1-((weight_list - weight_list.min()) /
                             (weight_list.max() - weight_list.min()))

dictionary_alt_scaling = copy.deepcopy(embedding_dictionary)
dictionary_pre_scaling = copy.deepcopy(embedding_dictionary)
dictionary_no_square_scaling = copy.deepcopy(embedding_dictionary)
dictionary_pre_square_scaling = copy.deepcopy(embedding_dictionary)
dictionary_pre_sq_no_yj = copy.deepcopy(embedding_dictionary)

yj_data, _ = scipy.stats.yeojohnson(weight_list)
sqr_yj_data, _ = scipy.stats.yeojohnson(sqr_weights)

filter_map = numpy.array(1-((yj_data - yj_data.min()) /
                            (yj_data.max() - yj_data.min()))**2)

no_square_filter_map = numpy.array(1-((yj_data - yj_data.min()) /
                                      (yj_data.max() - yj_data.min())))

pre_square_filter_map = numpy.array(1-((sqr_yj_data - sqr_yj_data.min()) /
                                       (sqr_yj_data.max() - sqr_yj_data.min())))

for key in embedding_dictionary:
    embedding_dictionary[key] *= filter_map
    dictionary_alt_scaling[key] *= no_yj_filter_map
    dictionary_no_square_scaling[key] *= no_square_filter_map
    dictionary_pre_square_scaling[key] *= pre_square_filter_map
    dictionary_pre_sq_no_yj[key] *= pre_sq_no_yj_filter_map
