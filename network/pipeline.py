import data_pipeline
import network_deployable
import argparse
import json
import random
import torch

def call_data_pipeline(config):
    corpus_name = config["corpus_name"]
    data_path = config["data_path"]
    vocab, pairs = data_pipeline.loadPrepareData(corpus_name, data_path)

    print("\nsample pairs:")
    for pair in pairs[:5]:
        print(pair)

    return vocab, pairs

def call_network_deployable(config, vocab, pairs):

    """
    1. Create new columns using inp, out, func_list
    2. During batching, apply other normalization
    3. Onehot encode everything necessary
    """
    # Example for validation
    small_batch_size = 5
    batches = network_deployable.batch2TrainData(vocab, [random.choice(pairs)
                                    for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len, meta_data = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)
    print("meta_data:", meta_data)

    # Configure models
    hidden_size = config['hidden_size']
    encoder_n_layers = config['encoder_n_layers']
    dropout = config['dropout']
    hidden_size = config['hidden_size']
    network_save_path = config["network_save_path"]
    model_name = config['model_name']
    attn_model = config['attn_model']
    encoder_n_layers = config["encoder_n_layers"]
    decoder_n_layers = config["encoder_n_layers"]

    # Configuring optimizer
    learning_rate = config["learning_rate"]
    decoder_learning_ratio = config["decoder_learning_ratio"]

    # Set checkpoint to load from; set to None if starting from scratch
    if args.resume:
        checkpoint_iter = config["checkpoint_iter"]
        loadFilename = os.path.join(network_save_path, args.model_name, args.corpus_name,
                                    '{}-{}_{}'.format(encoder_n_layers,
                                                      decoder_n_layers, hidden_size),
                                    '{}_checkpoint.tar'.format(checkpoint_iter))
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        vocab.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if args.resume:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if args.resume:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if args.resume:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # If you have cuda, configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Run training iterations
    print("Starting Training!")
    trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, config)
    return

def main():
    parser = argparse.ArgumentParser(
            description='Enables testing of neural network.') 
    parser.add_argument("-c", "--config", 
                            help="config file for network_deployable. Should correspond to model.", 
                            default="configs/config_basic.json")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    vocab, pairs = call_network_deployable(config)
    call_data_pipeline(config, vocab, pairs)

if __name__ == "__main__":
    main()
