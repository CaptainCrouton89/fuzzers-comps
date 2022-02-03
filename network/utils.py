import os
from function_mapping_handler import get_num_added_columns

def get_model_path(config, add_new_static_cols):
    data_config = config['data']
    model_config = config['model']
    network_save_path = data_config['network_save_path']
    corpus_name = data_config['corpus_name']
    model_name = data_config['model_name']
    static_inputs = data_config['static_inputs']
    meta_data_size = len(static_inputs)
    if add_new_static_cols:
        meta_data_size += get_num_added_columns(config["data"])
    encoder_n_layers = model_config['encoder_n_layers']
    decoder_n_layers = model_config['decoder_n_layers']
    hidden_size = model_config['hidden_size']

    model_features = str(encoder_n_layers) + "-" + \
        str(decoder_n_layers) + "_" + str(hidden_size+meta_data_size)
    return os.path.join(
        network_save_path, model_name, corpus_name, model_features)

if __name__ == '__main__':
    import json
    with open("configs/testing_wc.json") as f:
        config = json.load(f)
    print(get_model_path(config))