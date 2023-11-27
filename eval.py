import pickle
import os
import argparse
import torch
import os

#from .generate_masks import get_inference_mask
#from .cogmen.generate_masks import get_inference_mask

import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn import metrics
from tqdm import tqdm

import cogmen

log = cogmen.utils.get_logger()


def load_pkl(file):
    #print('Opening checkpoint file:', file)
    with open(file, "rb") as f:
        return pickle.load(f)

def get_inference_mask(cfg):
    audio_input_length = cfg['ail']
    video_input_length = cfg['vil']
    #text_input_length = cfg['til']

    #text_mask = np.ones(text_input_length)
    audio_mask = np.ones(audio_input_length)
    if cfg['v_mask']:
        video_mask = np.zeros(video_input_length)
    else:
        video_mask = np.ones(video_input_length)

    #return np.concatenate([audio_mask, text_mask, video_mask])
    return np.concatenate([audio_mask, video_mask])

def main(args):
    data = load_pkl(f"Cogmen_SLT/data/{args.dataset}/data_{args.dataset}.pkl")
    
    print('Opening checkpoint file:',str(args.dataset)+"_best_dev_f1_model_"+str(args.modalities)+".pt")
    model_dict = torch.load(
        "Cogmen_SLT/model_checkpoints/"
        + str(args.dataset)
        + "_best_dev_f1_model_"
        + str(args.modalities)
        + ".pt",
    )
    stored_args = model_dict["args"]
    model = model_dict["state_dict"]
    testset = cogmen.Dataset(data["test"], stored_args)

    test = True
    with torch.no_grad():
        golds = []
        preds = []
        for idx in tqdm(range(len(testset)), desc="test" if test else "dev"):
            data = testset[idx]
            golds.append(data["label_tensor"])

            mask_cfg = {
                    'vil':0,#512,
                    'ail':100,
                    'v_mask':True,
            }

            input_mask = get_inference_mask(mask_cfg)

            for k, v in data.items():
                if not k == "utterance_texts":
                    if k == 'input_tensor':
                        masked = (v*input_mask).type(v.dtype)
                        data[k] = masked.to(stored_args.device)
                    else:
                        data[k] = v.to(stored_args.device)
            y_hat = model(data)
            preds.append(y_hat.detach().to("cpu"))

        if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
            golds = torch.cat(golds, dim=0).numpy()
            preds = torch.cat(preds, dim=0).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)
        else:
            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            cm = confusion_matrix(golds, preds)

        if test:
            print(metrics.classification_report(golds, preds, digits=4))

            if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
                happy = metrics.f1_score(golds[:, 0], preds[:, 0], average="weighted")
                sad = metrics.f1_score(golds[:, 1], preds[:, 1], average="weighted")
                anger = metrics.f1_score(golds[:, 2], preds[:, 2], average="weighted")
                surprise = metrics.f1_score(
                    golds[:, 3], preds[:, 3], average="weighted"
                )
                disgust = metrics.f1_score(golds[:, 4], preds[:, 4], average="weighted")
                fear = metrics.f1_score(golds[:, 5], preds[:, 5], average="weighted")

                f1 = {
                    "happy": happy,
                    "sad": sad,
                    "anger": anger,
                    "surprise": surprise,
                    "disgust": disgust,
                    "fear": fear,
                }

            print(f"F1 Score: {f1}")
            print(cm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="eval.py")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="iemocap_4",
        choices=["iemocap", "iemocap_4", "mosei"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./Cogmen_SLT/data"
    )

    parser.add_argument("--device", type=str, default="cpu", help="Computing device.")

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in seqcontext.py"""
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        # required=True,
        choices=["a", "av", "at", "atv"],
        help="Modalities",
    )

    # emotion
    parser.add_argument(
        "--emotion", type=str, default=None, help="emotion class for mosei"
    )

    args = parser.parse_args()
    main(args)
