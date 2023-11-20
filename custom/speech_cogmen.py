# The idea is to train the underlying cogmen model but with speech modality only by using a pretrained
# vision-audio model (frozen) and use contrastive learning.
import warnings

warnings.filterwarnings("ignore")

import os
import os.path as osp
import sys
import argparse
import logging
import time
from tqdm import tqdm
from pathlib import Path

# Add the COGMEN directory to the path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

# * Computation library imports
import numpy as np
import matplotlib.pyplot as plt

# * Pytorch imports
import torch
import torch.nn as nn

# * Cogmen imports
import cogmen

# * Custom imports
import custom
from custom.utils import (
    load_pkl,
    print_log,
    print_trainability,
    display_method_info,
    check_dir,
    update_config,
    load_config,
)

# * Viz library imports
import matplotlib.pyplot as plt

# * Tensorboard for logging
import tensorboard as tb

""" -------------------------------------------------------------------- """

# create a dictionary of paths
PATH = {"ckpt_path": "./model_checkpoints/"}


class SpeechCOGMEN(nn.Module):
    def __init__(self, args):
        super(SpeechCOGMEN, self).__init__()
        self.args = args
        self.device = self.args.device
        self._rank = 0
        self._teacher_model = None
        self._teacher_args = None
        self._speech_cogmen = None

        self._preparation()
        self._load_pretrained_model()
        self._model()

    def _preparation(self):
        """Preparation of basic experiment setups"""
        # log and checkpoing
        base_dir = self.args.res_dir if self.args.res_dir is not None else "work_dirs"
        self.path = osp.join(
            base_dir,
            self.args.ex_name
            if not self.args.ex_name.startswith(self.args.res_dir)
            else self.args.ex_name.split(self.args.res_dir + "/")[-1],
        )
        self.checkpoints_path = osp.join(self.path, "checkpoints")

        if self._rank == 0:
            print("Rank:", self._rank)
            check_dir(self.path)
            check_dir(self.checkpoints_path)

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            prefix = "train" if not self.args.exp_mode else self.args.exp_mode
            logging.basicConfig(
                level=logging.INFO,
                filename=osp.join(self.path, "{}_{}.log".format(prefix, timestamp)),
                filemode="a",
                format="%(asctime)s - %(message)s",
            )

        # set random seeds
        cogmen.utils.set_seed(self.args.seed)

    # Load the pretrained teacher model (vision-audio) model
    def _load_pretrained_model(self):
        dash_line = "-" * 80 + "\n"
        # load the model and look at the model architecture
        try:
            print(">"*70)
            print("Loading teacher model")
            av_model = torch.load(PATH["ckpt_path"] + self.args.teacher_model + ".pt")
            ckpt_path = PATH["ckpt_path"] + self.args.teacher_model + ".pt"
            print(f"Teacher model: {ckpt_path} Loaded")
            print(">"*50)
        except:
            raise Exception("Teacher model not found. Please set the correct path or name of teacher model")


        self._teacher_args = av_model["args"]
        self._teacher_model = cogmen.COGMEN(self._teacher_args)
        self._teacher_model.load_state_dict(av_model["state_dict"])

        # * remove the classifier head from the teacher COGMEN_AV model
        # self._teacher_model = nn.Sequential(*list(self._teacher_model.children())[:-1])
        # self._teacher_model.clf = nn.Identity()
        self._teacher_model.to(self.device)

        # Load the dataset
        dataset = osp.join(
            self.args.data_dir_path,
            self.args.dataset,
            "data_" + self.args.dataset + ".pkl",
        )
        data = load_pkl(dataset)
        dataset_obj = cogmen.Dataset(data["train"], self._teacher_args)

        print(">" * 50)
        # get a batch of data from the dataset
        data = dataset_obj[0]
        # create a dummy model
        dummy_model = cogmen.COGMEN(self._teacher_args).to(self._teacher_args.device)
        # get shape of the input_data [Batch, max_embedding_len, concatenated_embedding_len]
        # display_method_info(dummy_model, data, stored_args.device)

        flops, params = display_method_info(
            dummy_model, data, self._teacher_args.device
        )
        print_log(dash_line + "FLOPS(G) :" + flops + "\n")
        print_log(dash_line + "Parameters (M) :" + params + "\n")
        print_log(dash_line + str(self._teacher_model) + "\n")

        # delete the dummy model aka clean
        del dummy_model, data

    # TODO: create a function to load the pretrained model and extract features from the generated graph
    def extract_gcn_features(self, batch_data):
        # * step 1: the pretrained graph is stored in self._teacher_model
        # * step 2: pass the data through the model and extract the graph features
        # * step 3: concatenate the features and return the features

        intermediate_output = {}
        teacher_embeddings = []

        def hook_fn(module, input, output, name):
            intermediate_output[name] = output.cpu().detach().numpy()

        target_layers = ["gcn"]
        hooks = []
        for layer_name, layer in self._teacher_model.named_children():
            if layer_name in target_layers:
                hook = layer.register_forward_hook(
                    lambda m, i, o, ln=layer_name: hook_fn(m, i, o, ln)
                )
                hooks.append(hook)

        self._teacher_model.eval()
        with torch.no_grad():
            # * preds include the graph features for the given batch_data
            # for idx in tqdm(range(len(batch_data)), desc="generating teacher embeddings"):
            # data = batch_data[idx]
            for k, v in batch_data.items():
                if not k == "utterance_texts":
                    batch_data[k] = v.to(self.device)
            embeddings_idx = self._teacher_model(batch_data)
            # print(embeddings_idx.shape)
            teacher_embeddings.append(intermediate_output["gcn"])

        # * remove the hooks
        for hook in hooks:
            hook.remove()

        return np.squeeze(teacher_embeddings)

    # TODO: create a function that generate data for the contrastive learning
    def generate_contrastive_data(self, data, teacher_embeddings):
        """
        The idea is to generate a list of lists where for each data point,
        the corresponding list contains the embedding of that data point (positive sample)
        and the other (N-1) embeddings of the other data points (negative samples)
        Here, N = number of emotion classes
        """
        permuted_data = []
        # * step 1: collect the negatives for the given label idx
        labels = data["label_tensor"].to("cpu").numpy()
        unique_labels = np.unique(labels)
        # * get indices of data points with same labels
        labels_list = {}
        for label in unique_labels:
            labels_list[label] = np.where(labels == label)[0]
            # print(f"Label {label} has {len(labels_list[label])} data points")
        # * loop over the labels and generate negatives
        for idx, label in enumerate(labels):
            tmp_list = []
            tmp_list.append((teacher_embeddings[idx], label))
            # * uniformly (random) sample an indice from the labels_list
            for negative_label, nl_list in labels_list.items():
                if not negative_label == label:
                    # * sample a negative label
                    negative_label_idx = np.random.choice(nl_list)
                    # * append the negative label to the permuted data
                    tmp_list.append(
                        (teacher_embeddings[negative_label_idx], negative_label)
                    )
            # * append the permuted data to the permuted_data list
            permuted_data.append(tmp_list)

        return permuted_data

    def _model(self):
        self._speech_cogmen = custom.model.COGMEN(self.args)

    # create COGMEN_A model only
    def forward(self, x):
        # teacher_gcn_embeddings = self.extract_gcn_features(x)
        # # * generate the contrastive data
        # pos_neg_samples = self.generate_contrastive_data(x, teacher_gcn_embeddings)
        # # * pass the data to the underlying model
        # self._speech_cogmen.set_teacher_emb(pos_neg_samples)
        return self._speech_cogmen(x)

    # * loss function of the model
    def get_loss(self, x, teacher_x=None):
        if teacher_x is not None:
            teacher_gcn_embeddings = self.extract_gcn_features(teacher_x)
            # * generate the contrastive data
            pos_neg_samples = self.generate_contrastive_data(teacher_x, teacher_gcn_embeddings)
            # * pass the data to the underlying model
            self._speech_cogmen.set_teacher_emb(pos_neg_samples)
        else:
            pos_neg_samples = None
        return self._speech_cogmen.get_loss(x, teacher_embeddings=pos_neg_samples)


class Config:
    def __init__(self, *args):
        self.__dict__.update(*args)

def generate_teacher_embeddings(cogmen_A: SpeechCOGMEN, args):
    """
    TODO: create a function to test the extraction of the features from the
    teacher model
    """
    # load test dataset
    dataset = osp.join(
        args.data_dir_path, args.dataset, "data_" + args.dataset + ".pkl"
    )
    data = load_pkl(dataset)
    dataset_obj = cogmen.Dataset(data["test"], cogmen_A._teacher_args)

    D = dataset_obj[0]
    print(D["input_tensor"].shape)
    print(D["label_tensor"].shape)
    print(D["label_tensor"].unique(return_counts=True))
    # * get the proportions of the labels in the dataset
    # * call the generate_teacher_embeddings function to generate emb
    emb = cogmen_A.extract_gcn_features(dataset_obj[0])
    emb_arr = np.array(emb)
    print(emb_arr.shape)
    pos_neg_samples = cogmen_A.generate_contrastive_data(D, emb_arr)
    print(len(pos_neg_samples))


def main(args):
    speech_cogmen = SpeechCOGMEN(args=args)
    # print(list(speech_cogmen._teacher_model.children())[:-1])
    generate_teacher_embeddings(speech_cogmen, args)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="speech_cogmen.py")
    # parser.add_argument(
    # "--dataset",
    # type=str,
    # required=True,
    # default="iemocap_4",
    # choices=["iemocap", "iemocap_4", "mosei"],
    # help="Dataset name.",
    # )

    # parser.add_argument(
    #     "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    # )

    # # Modalities
    # """ Modalities effects:
    #     -> dimentions of input vectors in dataset.py
    #     -> number of heads in transformer_conv in seqcontext.py"""
    # parser.add_argument(
    #     "--modalities",
    #     type=str,
    #     default="atv",
    #     # required=True,
    #     choices=["a", "at", "atv"],
    #     help="Modalities",
    # )

    # # Training parameters
    # parser.add_argument(
    #     "--from_begin", action="store_true", help="Training from begin.", default=False
    # )
    # parser.add_argument("--device", type=str, default="cuda", help="Computing device.")
    # parser.add_argument(
    #     "--epochs", default=1, type=int, help="Number of training epochs."
    # )

    # # Experiment details
    # parser.add_argument("--res_dir", type=str, default="./work_dirs", help="Directory to store results")
    # parser.add_argument("--ex_name", type=str, default="speech_cogmen", help="Experiment name")
    # parser.add_argument("--exp_mode", type=str, default="train", help="Experiment mode: train/eval")
    # # others
    # parser.add_argument("--seed", type=int, default=43, help="Random seed.")


    # args = parser.parse_args()
    args = custom.parser.speech_cogmen_parser()
    print(type(args))
    args = args.parse_known_args()[0]
    opt = args.__dict__

    # create load_config object
    print(">"*70)
    config = update_config(opt, load_config("custom/configs/speech_cogmen.py"))
    config = Config(config)

    print("From begin?" , config.from_begin)
    main(config)
