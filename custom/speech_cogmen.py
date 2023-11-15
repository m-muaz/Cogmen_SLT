# The idea is to train the underlying cogmen model but with speech modality only by using a pretrained 
# vision-audio model (frozen) and use contrastive learning.

import os
import os.path as osp
import sys
import argparse
import logging
import time
from pathlib import Path

# Add the COGMEN directory to the path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import cogmen
from custom.utils import (load_pkl, print_log, print_trainability, display_method_info, check_dir)

# create a dictionary of paths
PATH = {"ckpt_path":"./model_checkpoints/"}

class SpeechCOGMEN(nn.Module):
    def __init__(
        self,
        args
    ):
        super(SpeechCOGMEN, self).__init__()
        self.args = args
        self.device = self.args.device
        self._rank=0
        self._teacher_model = None


        self._preparation()
        self._load_pretrained_model()
        
        

    def _preparation(self):
        """Preparation of basic experiment setups"""
        # log and checkpoing
        base_dir = self.args.res_dir if self.args.res_dir is not None else 'work_dirs'
        self.path = osp.join(base_dir, self.args.ex_name if not self.args.ex_name.startswith(self.args.res_dir) \
            else self.args.ex_name.split(self.args.res_dir+'/')[-1])
        self.checkpoints_path = osp.join(self.path, 'checkpoints')

        if self._rank == 0:
            print("Rank:", self._rank)
            check_dir(self.path)
            check_dir(self.checkpoints_path)

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())   
            prefix = 'train' if not self.args.exp_mode else self.args.exp_mode
            logging.basicConfig(level=logging.INFO,
                            filename=osp.join(self.path, '{}_{}.log'.format(prefix, timestamp)),
                            filemode='a', format='%(asctime)s - %(message)s'
            )
        
        # set random seeds
        cogmen.utils.set_seed(self.args.seed)


    # Load the pretrained teacher model (vision-audio) model
    def _load_pretrained_model(self):
        # load the model and look at the model architecture
        av_model = torch.load(
            PATH["ckpt_path"]+
            "model"+
            ".pt"
            )
        stored_args = av_model["args"]
        self._teacher_model = av_model["state_dict"]

        # Load the dataset
        dataset = osp.join(self.args.data_dir_path, self.args.dataset, "data_" + self.args.dataset + ".pkl")
        data = load_pkl(dataset)
        dataset_obj = cogmen.Dataset(data["train"], stored_args)
        
        print(">"*50)
        # print("Length of train data: ", len(data["train"]))
        print("Testset: ", len(dataset_obj))

        # get a batch of data from the dataset
        data = dataset_obj[0]
        # create a dummy model
        dummy_model = cogmen.COGMEN(stored_args).to(stored_args.device)
        dummy_model.load_state_dict(self._teacher_model)
        # get shape of the input_data [Batch, max_embedding_len, concatenated_embedding_len]
        # display_method_info(dummy_model, data, stored_args.device)

        print_log(str(dummy_model))
        # delete the dummy model
        del dummy_model



    

def main(args):
    speech_cogmen = SpeechCOGMEN(args=args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="speech_cogmen.py")
    parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    default="iemocap_4",
    choices=["iemocap", "iemocap_4", "mosei"],
    help="Dataset name.",
)

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )


    # Modalities    
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in seqcontext.py"""
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        # required=True,
        choices=["a", "at", "atv"],
        help="Modalities",
    )

    # Training parameters
    parser.add_argument(
        "--from_begin", action="store_true", help="Training from begin.", default=False
    )
    parser.add_argument("--device", type=str, default="cuda", help="Computing device.")
    parser.add_argument(
        "--epochs", default=1, type=int, help="Number of training epochs."
    )

    # Experiment details
    parser.add_argument("--res_dir", type=str, default="./work_dirs", help="Directory to store results")
    parser.add_argument("--ex_name", type=str, default="speech_cogmen", help="Experiment name")
    parser.add_argument("--exp_mode", type=str, default="train", help="Experiment mode: train/eval")
    # others
    parser.add_argument("--seed", type=int, default=43, help="Random seed.")



    args = parser.parse_args()
    main(args)
