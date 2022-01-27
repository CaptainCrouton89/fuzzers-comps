import numpy

embedding_dictionary = {}
first_50_list = []
with open('../embeddings/glove.6B.100d.txt') as embedding_file:
    for _ in range(50):
        (token, embedding_string) = embedding_file.readline().split(' ', 1)
        embedding_vector = numpy.array(
            [float(i) for i in embedding_string.split()])
        first_50_list.append(embedding_vector)
    count = 0
    for line in embedding_file:
        (token, embedding_string) = line.split(' ', 1)
        embedding_vector = numpy.array(
            [float(i) for i in embedding_string.split()])
        embedding_dictionary[token] = embedding_vector
