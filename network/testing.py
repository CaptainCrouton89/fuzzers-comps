import json
import argparse
import torch
from torch import nn
from network_deployable import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder, loadPrepareData

# Get args
parser = argparse.ArgumentParser(description='Enables testing of neural network.')
parser.add_argument("-m", "--model_checkpoint", help="uses model for testing responses")
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

with open('config_basic.json') as f:
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