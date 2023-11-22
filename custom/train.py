from comet_ml import Experiment, Optimizer

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

# Add the COGMEN directory to the path
sys.path.append(str(Path(__file__).parent.parent.absolute()))


import argparse
import torch
import os
import cogmen
import custom

log = cogmen.utils.get_logger()
torch.autograd.set_detect_anomaly(True)


def func(experiment, trainset, devset, testset, model, opt, sched, args):
    # args.hidden_size = experiment.get_parameter("HIDDEN_DIM")
    args.seqcontext_nlayer = experiment.get_parameter("SEQCONTEXT")
    args.gnn_nheads = experiment.get_parameter("GNN_HEAD")
    args.learning_rate = experiment.get_parameter("LR")
    args.wp = experiment.get_parameter("WP")
    args.wf = experiment.get_parameter("WF")
    args.use_highway = experiment.get_parameter("HIGHWAY")
    args.class_weight = experiment.get_parameter("CLASS_WEIGHT")
    args.drop_rate = experiment.get_parameter("DROPOUT")
    args.experiment = experiment

    coach = cogmen.Coach(trainset, devset, testset, model, opt, sched, args)
    if not args.from_begin:
        ckpt = torch.load(args.model_ckpt)
        coach.load_ckpt(ckpt)
        print("Training from checkpoint...")

    # Train
    log.info("Start training...")
    (
        best_dev_f1,
        best_epoch,
        best_state,
        train_losses,
        dev_f1s,
        test_f1s,
    ) = coach.train()
    return best_dev_f1


def main(args):
    cogmen.utils.set_seed(args.seed)

    if args.emotion:
        args.data = os.path.join(
            args.data_dir_path,
            args.dataset,
            "data_" + args.dataset + "_" + args.emotion + ".pkl",
        )
    else:
        if args.transformers:
            args.data = os.path.join(
                args.data_dir_path,
                args.dataset,
                "transformers",
                "data_" + args.dataset + ".pkl",
            )
            print(os.path.join(args.data_dir_path, args.dataset, "transformers"))
        else:
            args.data = os.path.join(
                args.data_dir_path, args.dataset, "data_" + args.dataset + ".pkl"
            )

    # load data
    log.debug("Loading data from '%s'." % args.data)

    data = cogmen.utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = custom.Dataset(data["train"], args)
    devset = custom.Dataset(data["dev"], args)
    testset = custom.Dataset(data["test"], args)
    print("Train set length")

    log.debug("Building model...")
    if args.log_in_comet and not args.tuning:
        model_file = "./model_checkpoints/" + str(args.experiment.get_key()) + ".pt"
    else:
        model_zoo = os.listdir("./model_checkpoints")
        # * check if Cogmen_A.pt already exists
        if "Cogmen_A.pt" in model_zoo:
            # * create a file with name Cogmen_A_x.pt where x is the next available number
            i = 1
            while True:
                if "Cogmen_A_" + str(i) + ".pt" in model_zoo:
                    i += 1
                else:
                    break
            name = "Cogmen_A_" + str(i) + ".pt"
        else:
            name = "Cogmen_A.pt"
        model_file = "./model_checkpoints/" + name
    # model = custom.model.COGMEN(args).to(args.device)
    model = custom.speech_cogmen.SpeechCOGMEN(args).to(args.device)
    opt = cogmen.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)
    sched = opt.get_scheduler(args.scheduler)

    coach = cogmen.Coach(trainset, devset, testset, model, opt, sched, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)
        print("Training from checkpoint...")

    # Train
    log.info("Start training...")
    ret = coach.train()

    # Test after training before saving
    dash_line = ">"*70
    log.info(dash_line + "\n")
    log.info("Testing after training...")
    coach.evaluate(test=True)

    # Save.
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "state_dict": ret[2],
        "dev_f1s": ret[4],
        "test_f1s": ret[5],
        "args": args.__dict__,
    }
    torch.save(checkpoint, model_file)


class Config:
    def __init__(self, *args):
        self.__dict__.update(*args)

if __name__ == "__main__":

    # args = custom.parser.speech_cogmen_parser()

    args = custom.parser.speech_cogmen_parser()
    args = args.parse_known_args()[0]
    opt = args.__dict__

    # create load_config object
    print(">"*70)
    config = custom.utils.update_config(opt, custom.utils.load_config("custom/configs/speech_cogmen.py"))
    config = Config(config)

    config.dataset_embedding_dims = {
        "iemocap": {
            "a": 100,
            "t": 768,
            "v": 512,
            "at": 100 + 768,
            "tv": 768 + 512,
            "av": 612,
            "atv": 100 + 768 + 512,
        },
        "iemocap_4": {
            "a": 100,
            "t": 768,
            "v": 512,
            "at": 100 + 768,
            "tv": 768 + 512,
            "av": 612,
            "atv": 100 + 768 + 512,
        },
        "mosei": {
            "a": 80,
            "t": 768,
            "v": 35,
            "at": 80 + 768,
            "tv": 768 + 35,
            "av": 80 + 35,
            "atv": 80 + 768 + 35,
        },
    }

    print("From begin?" , config.from_begin)
    main(config)
