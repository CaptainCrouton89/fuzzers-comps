import logging
import json
import os
import argparse
import torch
from torch import nn
from gru_attention_network import EncoderRNN, LuongAttnDecoderRNN, indexesFromSentence, SOS_token, EOS_token
from data_pipeline import load_prepare_data, Voc
from pipeline_functions.string_normalize import normalize_one_string
import random

# %%


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, top_n):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.top_n = top_n

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        print(f"input seq: {input_seq}")
        print(f"input length: {input_length}")
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(
            1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            print(decoder_input.shape)
            print(decoder_hidden.shape)
            print(encoder_outputs.shape)
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            # decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            scores, indexs = torch.topk(decoder_output, self.top_n, dim=1)
            r = self.pickOption(scores, indexs, decoder_input)

            decoder_scores = torch.transpose(scores, 0, 1)[r]
            decoder_input = torch.transpose(indexs, 0, 1)[r]

            # print("score: %f, index: %d" %(decoder_scores, decoder_input))
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
            if decoder_input[0].item() == EOS_token:
                break
        # Return collections of word tokens and scores
        return all_tokens, all_scores

    def pickOption(self, scores, indexs, previous):
        print(list(zip(scores[0].tolist(), indexs[0].tolist())))
        probs = scores[0].tolist()
        opts = list(range(len(indexs[0])))
        r = random.choices(opts, probs)
        r = r[0]
        print(indexs[0][r].item())
        if (indexs[0][r] == previous[0].item()):
            print('fuck ' + str(r))
            return self.pickOption(scores, indexs, previous)
        print('yay ' + str(r))

        return r


def evaluate(encoder, decoder, searcher, voc, sentence, max_length):
    # Format input sentence as a batch
    # words -> indexes
    # indexes_batch = [indexesFromSentence(voc, sentence)]
    indexes_batch = [indexesFromSentence(voc, sentence[0])]
    print(indexes_batch)
    testing = []
    for i in range(len(indexes_batch[0])):
        testing.append([indexes_batch[0][i]])
    indexes_batch = testing
    for i in range(1, len(sentence)):
        indexes_batch.append([int(sentence[i])])

    # indexes_batch.extend(sentence[1:])
    print(indexes_batch)

    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    print(lengths)
    print(indexes_batch)
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    print(list(zip(tokens, decoded_words)))
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc, max_length, static_inputs):
    while(1):
        try:
            # Get input sentence
            content = input('content> ')
            # Check if it is quit case
            if content == 'q' or content == 'quit':
                break
            content = [normalize_one_string(content)]

            for field in static_inputs:
                content.append(input(field + "> "))

            # Parse the sentence into a tuple representing the content and
            # input metadata ~somehow~

            # Evaluate sentence
            output_words = evaluate(
                encoder, decoder, searcher, voc, content, max_length)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (
                x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def main():
    meta_data_size = 6
    parser = argparse.ArgumentParser(
        description='Enables testing of neural network.')
    parser.add_argument("-c", "--config",
                        help="config file for running model. Should correspond to model.",
                        default="configs/reddit.json")
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
    max_length = data_config['max_len']

    checkpoint = test_config['checkpoint']
    top_n = test_config['top_n']

    encoder_n_layers = model_config['encoder_n_layers']
    dropout = model_config['dropout']
    attn_model = model_config['attn_model']
    decoder_n_layers = model_config['decoder_n_layers']
    hidden_size = model_config['hidden_size']

    model_features = str(encoder_n_layers) + "-" + \
        str(decoder_n_layers) + "_" + str(hidden_size+meta_data_size)
    model_path = os.path.join(
        network_save_path, model_name, corpus_name, model_features, checkpoint)

    # If loading on same machine the model was trained on
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    voc = Voc(corpus_name)
    encoder_sd = checkpoint['encoder']
    decoder_sd = checkpoint['decoder']
    encoder_optimizer_sd = checkpoint['encoder_opt']
    decoder_optimizer_sd = checkpoint['decoder_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

    embedding = nn.Embedding(voc.num_words, hidden_size)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size + meta_data_size, voc.num_words, decoder_n_layers, dropout)

    embedding.load_state_dict(embedding_sd)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder, top_n)

    evaluateInput(encoder, decoder, searcher, voc, max_length, static_inputs)


if __name__ == "__main__":
    main()

# %%
