from fvcore.nn import FlopCountAnalysis, flop_count_table

import pickle
import logging
import os

import torch

# Function to load the pretrained model
def display_method_info(model, data, device):
    # _, max_embedding_len, concatenated_embedding_len = input_shape
    # dummy_input = torch.ones(1, max_embedding_len, concatenated_embedding_len).to(device)
    # dummy_text_len_tensor = torch.ones(1).to(device)

    dummy_data = data
    for k, v in data.items():
        if k not in ["utterance_texts"]:
            dummy_data[k] = v.to(device)
    

    # print the flops
    dash_line = "-" * 80 + "\n"
    info = model.__repr__()
    # dummy_data = {
    #     "text_len_tensor": dummy_text_len_tensor,
    #     "input_tensor": dummy_input,
    #     }
    flops = FlopCountAnalysis(model, dummy_data)
    flops = flop_count_table(flops)
    print(flops)
    # print_log('Model info:\n' + info + '\n' + flops+ '\n' + dash_line)

# Function to print the number of trainable parameters in the model
def print_trainability(model, indent=""):
    dash_line = "-" * 80 + "\n"
    for name, param in model.named_parameters():
        trainable = param.requires_grad
        print(dash_line)
        print(f"{indent}Layer: {name}, Trainable: {trainable}")
        
    for name, sub_module in model.named_children():
        print(f"{indent}Module: {name}")
        print_trainability(sub_module, indent + "  ")

# Function to print log 
def print_log(message):
    print(message)
    logging.info(message)

# Load the pickle file
def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def check_dir(path):
    """
    Check if the dir exists if not create it
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True