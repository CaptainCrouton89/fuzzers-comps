import json
import os
import argparse
import torch
from torch import nn
from gru_attention_network import EncoderRNN, LuongAttnDecoderRNN, indexesFromSentence, SOS_token
from data_pipeline import loadPrepareData, normalizeString

# %%
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
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
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, voc, sentence, max_length):
    ### Format input sentence as a batch
    # words -> indexes
    # indexes_batch = [indexesFromSentence(voc, sentence)]
    indexes_batch = [indexesFromSentence(voc, sentence[0])]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluateInput(encoder, decoder, searcher, voc, max_length, static_inputs):
    while(1):
        try:
            # Get input sentence
            content = input('content> ')
            # Check if it is quit case
            if content == 'q' or content == 'quit': 
                break
            content = [normalizeString(content)]

            for field in static_inputs:
                content.append(input(field + "> "))

            # Parse the sentence into a tuple representing the content and 
            # input metadata ~somehow~

            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, content, max_length)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def main():
    parser = argparse.ArgumentParser(description='Enables testing of neural network.')
    parser.add_argument("-c", "--config", 
                            help="config file for running model. Should correspond to model.", 
                            default="configs/config_basic.json")
    args = parser.parse_args()



    with open(str(args.config)) as f:
        config = json.load(f)

    test_config = config['testing']
    model_config = config['model']

    network_saves_path = test_config['network_saves_path']
    corpus_name = test_config['corpus_name']
    model_name = test_config['model_name']
    checkpoint = test_config['checkpoint']
    data_path = test_config['data_path']
    static_inputs = test_config['static_inputs']
    max_length = test_config['max_len']

    model_path = os.path.join(network_saves_path, model_name, corpus_name, checkpoint)

    # voc, pairs = loadPrepareData(corpus_name, data_path)
    voc, pairs = loadPrepareData(config["data"], )

    # If loading on same machine the model was trained on
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    encoder_sd = checkpoint['encoder']
    decoder_sd = checkpoint['decoder']
    encoder_optimizer_sd = checkpoint['encoder_opt']
    decoder_optimizer_sd = checkpoint['decoder_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

    hidden_size = model_config['hidden_size']
    encoder_n_layers = model_config['encoder_n_layers']
    dropout = model_config['dropout']
    hidden_size = model_config['hidden_size']
    attn_model = model_config['attn_model']
    decoder_n_layers = model_config['decoder_n_layers']

    embedding = nn.Embedding(voc.num_words, hidden_size)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

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
    searcher = GreedySearchDecoder(encoder, decoder)

    evaluateInput(encoder, decoder, searcher, voc, max_length, static_inputs)

if __name__ == "__main__":
    main()
