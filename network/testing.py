import logging
import json
import os
import argparse
import torch
from torch import nn
from custom_logger import init_logger
from gru_attention_network import EncoderRNN, LuongAttnDecoderRNN, indexesFromSentence, SOS_token, EOS_token, batch2TrainData
from data_pipeline import Voc
from pipeline_functions.string_normalize import normalize_one_string
import random
import pandas as pd
from function_mapping_handler import apply_mappings_testing, get_num_added_columns
import re

# %%

'''
Defines a greedy search decoder to query the neural network.
'''
class GreedySearchDecoder(nn.Module):
    '''
    :param encoder: the trained encoder to use
    :param decoder: the trained decoder to use
    :param top_n: how many options to consider
    :param threshold: minimum score to be considered
    :returns: none
    '''
    def __init__(self, encoder, decoder, top_n, threshold):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.top_n = top_n
        self.threshold = threshold

    '''
    :param input_seq: the seed content, converted into indices in the vocab
    :param metadata: metadata
    :param input_length: length of the input 'string'
    :param max_length: longest sentence to output
    :returns: list of words (index form) and their scores
    '''
    def forward(self, input_seq, metadata, input_length, max_length):
        # Forward input through encoder model
        # logging.debug(f"input seq: {input_seq}")
        # logging.debug(f"metadata: {metadata}")
        # logging.debug(f"input length: {input_length}")
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # logging.debug(f"encoder_output shape:{encoder_outputs.shape}")
        # logging.debug(f"encoder_hidden shape:{encoder_hidden.shape}")

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # logging.debug(f"decoder hidden shape:{decoder_hidden.shape}")

        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(
            1, 1, device=device, dtype=torch.long) * SOS_token
        # logging.debug(f"decoder input shape:{decoder_input.shape}")

        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for i in range(max_length):
            # Forward pass through decoder
            # logging.debug(f"decoder input shape:{decoder_input.shape}")
            # logging.debug(f"decoder hidden shape:{decoder_hidden.shape}")
            # logging.debug(f"encoder output shape:{encoder_outputs.shape}")

            # Create a tensor containing the meta data information
            # this next chunk definitely won't scale to other network sizes
            meta_data_tensor = torch.LongTensor(
                [[[meta_data_list for meta_data_list in metadata]] for _ in range(self.decoder.n_layers)])
            meta_data_tensor.to(device)
            # logging.debug(f"meta_data_tensor shape:{meta_data_tensor.shape}")
            # logging.debug(f"decoder input shape:{decoder_input.shape}")

            # logging.debug(f"decoder_hidden shape:{decoder_hidden.shape}")
            decoder_hidden = torch.narrow(decoder_hidden, 2, 0, self.encoder.hidden_size)
            decoder_hidden = torch.cat((decoder_hidden, meta_data_tensor), 2)
            encoder_outputs = decoder_hidden

            # logging.debug(f"decoder_input shape:{decoder_input.shape}")
            # logging.debug(f"decoder_hidden shape:{decoder_hidden.shape}")
            # logging.debug(f"encoder_outputs shape:{encoder_outputs.shape}")
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            # Obtain most likely word tokens and their softmax scores
            scores, indexs = torch.topk(decoder_output, self.top_n+1, dim=1)
            r = self.pickIndex(scores, indexs, decoder_input, i)

            decoder_score = torch.transpose(scores, 0, 1)[r]
            decoder_input = torch.transpose(indexs, 0, 1)[r]

            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_score), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
            if decoder_input[0].item() == EOS_token:
                break
        # Return collections of word tokens and scores
        return all_tokens, all_scores

    '''
    Given an array of scores, randomly pick the assosciated index.
    Ensures you don't duplicate the previous token.
    '''
    def pickIndex(self, scores, indexs, previous, length):
        probs = scores[0].tolist()
        opts = list(range(len(indexs[0])))

        # Decrease chance to predict EOS early
        if 2 in opts and length < 10:
            probs[opts.index(2)] /= (10/(length+1))

        # Remove least likely element
        min, ind = 1, 0
        for i in range(len(probs)):
            if probs[i] < min:
                min = probs[i]
                ind = i
        probs[ind] = 0

        probs = [p if p >= self.threshold else 0 for p in probs]
        if probs[0] == 0:
            probs[0] = 1

        r = random.choices(opts, probs)[0]
        # logging.debug(indexs[0][r].item())
        if (indexs[0][r] == previous[0].item() and len(probs) != 1):
            probs[r] = 0
            r = random.choices(opts, probs)[0]

        return r


def denormalize(s):
    s = re.sub(r" i ", r" I ", s)
    s = re.sub(r" ([.!?])", r"\1", s)
    s = re.sub(r" (\'s)", r"\1", s)
    s = re.sub(r" (n\'t)", r"\1", s)
    s = re.sub(r" (\'ll)", r"\1", s)
    s = re.sub(r" (\'re)", r"\1", s)
    s = re.sub(r" (\'d)", r"\1", s)

    for c in '.!?':
        temp = s.split(c+' ')
        for i in range(len(temp)):
            if len(temp[i]) == 0:
                continue
            elif len(temp[i]) == 1:
                temp[i] = temp[i][0].upper()
            else:
                temp[i] = temp[i][0].upper() + temp[i][1:]
        s = (c+' ').join(temp)
    return s

def evaluate(searcher, voc, content, max_length, do_denormalize=True):
    # Format input content as a batch
    # words -> indexes
    sentence = [indexesFromSentence(voc, content[0])]
    metadata = list(map(int, content[1:]))
    # logging.debug(f"sentence: {sentence}")
    # logging.debug(f"metadata: {metadata}")


    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in sentence])
    # Transpose dimensions of batch to match models' expectations
    # logging.debug(f"lengths: {lengths}")
    # logging.debug(f"sentence last: {sentence}")
    input_batch = torch.LongTensor(sentence).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    # logging.debug(f"input_batch : {input_batch}")
    lengths = lengths.to("cpu")


    # Decode sentence with searcher
    tokens, _ = searcher.forward(input_batch, metadata, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    # logging.debug(list(zip(tokens, decoded_words)))

    decoded_words[:] = [x for x in decoded_words if not (
        x == 'EOS' or x == 'PAD')]
    decoded_sentence = ' '.join(decoded_words)
    if do_denormalize:
        decoded_sentence = denormalize(decoded_sentence)
    return decoded_sentence


def evaluateInput(config, searcher, voc, max_length, static_inputs, encoder_inputs):
    while True:
        try:
            # Get input sentence
            content = input('content> ')
            # Check if it is quit case
            if content == 'q' or content == 'quit':
                break
            elif content == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            content = [normalize_one_string(content)]

            for field in static_inputs:
                content.append(input(field + "> "))

            data_df = pd.DataFrame(columns=encoder_inputs + static_inputs)
            data_df.loc[0] = content
            logging.debug(f"data_df:\n{data_df}")
            _ = apply_mappings_testing(data_df, config)
            # logging.info(f"data_df after mappings:\n{data_df}")
            content = data_df.values.tolist()[0]
            logging.debug(f"content:\n{content}")
            # Parse the sentence into a tuple representing the content and

            # Evaluate sentence
            response = evaluate(searcher, voc, content, max_length)
            # Format and print response sentence
            print('Bot:', response)

        except KeyError as e:
            logging.warning(f"Error: Encountered unknown word {e}.")

def main():
    parser = argparse.ArgumentParser(
        description='Enables testing of neural network.')
    parser.add_argument("-c", "--config",
                        help="config file for running model. Should correspond to model.",
                        default="configs/reddit.json")
    parser.add_argument("-l", "--loglevel",
                        help="Level at which to log events.",
                        default="INFO")
    args = parser.parse_args()

    with open(str(args.config)) as f:
        config = json.load(f)

    test_config = config['testing']
    data_config = config['data']
    model_config = config['model']

    network_save_path = data_config['network_save_path']
    corpus_name = data_config['corpus_name']
    model_name = data_config['model_name']
    static_inputs = data_config['static_inputs']
    encoder_inputs = data_config['encoder_inputs']
    max_length = data_config['max_len']

    meta_data_size = len(static_inputs)
    meta_data_size += get_num_added_columns(data_config)


    checkpoint = test_config['checkpoint']
    top_n = test_config['top_n']
    threshold = test_config['threshold']

    encoder_n_layers = model_config['encoder_n_layers']
    dropout = model_config['dropout']
    attn_model = model_config['attn_model']
    decoder_n_layers = model_config['decoder_n_layers']
    hidden_size = model_config['hidden_size']

    model_features = str(encoder_n_layers) + "-" + \
        str(decoder_n_layers) + "_" + str(hidden_size+meta_data_size)
    model_path = os.path.join(
        network_save_path, model_name, corpus_name, model_features, checkpoint)


    init_logger(os.path.join("logs", "testing", corpus_name), args.loglevel, args.config)
    logging.debug(f"Using model at {model_path}")
    # If loading on same machine the model was trained on
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    voc = Voc(corpus_name)
    encoder_sd = checkpoint['encoder'] # change to 'en' for old_app
    decoder_sd = checkpoint['decoder'] # change to 'de' for old_app
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

    embedding = nn.Embedding(voc.num_words, hidden_size)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, 1, dropout, meta_data_size) 
        # We use batchsize of 1 since we are testing only one item

    embedding.load_state_dict(embedding_sd)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder, top_n, threshold)

    evaluateInput(config, searcher, voc, max_length, static_inputs, encoder_inputs)


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
if __name__ == "__main__":
    main()

# %%
