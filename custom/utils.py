from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import profile, clever_format

import pickle
import logging
import os
import os.path as osp
import tempfile
import re
import shutil
import sys
import ast
from importlib import import_module

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
    # flop = FlopCountAnalysis(model, dummy_data)
    # flops = flop_count_table(flop)
    flops, params = profile(model, inputs=(dummy_data,))
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params
    # print_log('Model info:\n' + info + '\n' + flops+ '\n' + dash_line)

# Function to print the number of trainable parameters in the model
def print_trainability(model, indent=""):
    dash_line = "-" * 80 + "\n"
    for name, param in model.named_parameters():
        trainable = param.requires_grad
        print_log(dash_line)
        print_log(f"{indent}Layer: {name}, Trainable: {trainable}")
        
    for name, sub_module in model.named_children():
        print_log(f"{indent}Module: {name}")
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


# Function to update the config file
def update_config(args, config, exclude_keys=list()):
    """update the args dict with a new config"""
    assert isinstance(args, dict) and isinstance(config, dict)
    for k in config.keys():
        if args.get(k, False):
            if args[k] != config[k] and k not in exclude_keys and args[k] is not None:
                print(f"overwrite config key -- {k}: {args[k]} -> {config[k]}")
                args[k] = config[k]
            else:
                continue
                # args[k] = config[k]
        else:
            args[k] = config[k]
    return args

def load_config(filename:str = None):
    """load and print config"""
    print('loading config from ' + filename + ' ...')
    try:
        configfile = Config(filename=filename)
        config = configfile._cfg_dict
    except (FileNotFoundError, IOError):
        config = dict()
        print('warning: fail to load the config!')
    return config

'''
Thanks the code from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py wrote by Open-MMLab.
The `Config` class here uses some parts of this reference.
'''

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


class Config:
    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')

        if filename is not None:
            cfg_dict = self._file2dict(filename, True)
            filename = filename

        super(Config, self).__setattr__('_cfg_dict', cfg_dict)
        super(Config, self).__setattr__('_filename', filename)
    
    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, 'r') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    @staticmethod
    def _substitute_predefined_vars(filename, temp_config_name):
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        with open(filename, 'r') as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w') as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _file2dict(filename, use_predefined_variables=True):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.py']:
            raise IOError('Only py type are supported now!')

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)
            temp_config_name = osp.basename(temp_config_file.name)

            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename,
                                                   temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)

            if filename.endswith('.py'):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules[temp_module_name]
            # close temp file
            temp_config_file.close()
        return cfg_dict

    @staticmethod
    def fromfile(filename, use_predefined_variables=True):
        cfg_dict = Config._file2dict(filename, use_predefined_variables)
        return Config(cfg_dict, filename=filename)


