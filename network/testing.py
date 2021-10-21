import json
import argparse
import torch
from torch import nn
from network_deployable import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder, loadPrepareData, indexesFromSentence, normalizeString

def evaluate(encoder, decoder, searcher, voc, sentence, max_length):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
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


def evaluateInput(encoder, decoder, searcher, voc, max_length):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence, max_length)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# Get args
parser = argparse.ArgumentParser(description='Enables testing of neural network.')
parser.add_argument("-m", "--model_checkpoint", help="uses model for testing responses", 
                        default="../data/network_saves/cb_model/AppReviewsResponses/2-2_500_local/4000_checkpoint.tar")
parser.add_argument("-c", "--config", help="config file for running model. Should correspond to model.", 
                        default="configs/config_basic.json")
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

with open('configs/config_basic.json') as f:
    config = json.load(f)

if not args.model_checkpoint:
    print("No model given. Use `-m <model name>` to give model")
    exit()

pairs, voc = loadPrepareData(config['corpus_name'], config["data_path"])

# If loading on same machine the model was trained on
checkpoint = torch.load(args.model_checkpoint)
# If loading a model trained on GPU to CPU
#checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
voc.__dict__ = checkpoint['voc_dict']

hidden_size = config['hidden_size']
encoder_n_layers = config['encoder_n_layers']
dropout = config['dropout']
hidden_size = config['hidden_size']

embedding = nn.Embedding(voc.num_words, hidden_size)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(
    config['attn_model'], embedding, hidden_size, voc.num_words, config['decoder_n_layers'], dropout)

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

evaluateInput(encoder, decoder, searcher, voc, config['MAX_LENGTH'])