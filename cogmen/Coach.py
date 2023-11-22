import copy
import time

import os
import numpy as np
from numpy.core import overrides
import torch
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

from custom.utils import check_dir

import cogmen

log = cogmen.utils.get_logger()

from tensorboardX import SummaryWriter


class Coach:
    def __init__(self, trainset, devset, testset, model, opt, sched, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.scheduler = sched
        self.args = args
        self.dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
        }

        if args.dataset and args.emotion == "multilabel":
            self.dataset_label_dict["mosei"] = {
                "happiness": 0,
                "sadness": 1,
                "anger": 2,
                "surprise": 3,
                "disgust": 4,
                "fear": 5,
            }

        if args.emotion == "7class":
            self.label_to_idx = {
                "Strong Negative": 0,
                "Weak Negative": 1,
                "Negative": 2,
                "Neutral": 3,
                "Positive": 4,
                "Weak Positive": 5,
                "Strong Positive": 6,
            }
        else:
            self.label_to_idx = self.dataset_label_dict[args.dataset]

        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None
        
        # * create a epoch tracker for evaluation
        self.epoch_tracker_dev = 0
        self.epoch_tracker_test = 0

        # * Create a tensorboard summary writer
        if "log_in_tensorboard" in self.args.__dict__ and self.args.log_in_tensorboard:
            # if there is nothing in side tb_log_dir then create exp run with name
            # ex_name_i where i is the next available number
            check_dir(self.args.tb_log_dir)
            list_tb_runs = os.listdir(self.args.tb_log_dir)
            tb_exp_run = None
            if len(list_tb_runs) == 0:
                tb_exp_run = self.args.ex_name+"_1"
            else:
                i = 1
                while True:
                    if self.args.ex_name + str(i) in list_tb_runs:
                        i += 1
                    else:
                        break
                tb_exp_run = self.args.ex_name + str(i)
            tb_exp_run = os.path.join(self.args.tb_log_dir, tb_exp_run)
            self.writer = SummaryWriter(log_dir=tb_exp_run, flush_secs=30)
        else:
            self.writer = None


    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["state_dict"]
        self.model.load_state_dict(self.best_state)
        print("Loaded best model.....")

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = (
            self.best_dev_f1,
            self.best_epoch,
            self.best_state,
        )

        dev_f1s = []
        test_f1s = []
        train_losses = []
        best_test_f1 = None

        # Train
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            dev_f1, dev_loss = self.evaluate()
            self.scheduler.step(dev_loss)
            test_f1, _ = self.evaluate(test=True)
            if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                test_f1 = np.array(list(test_f1.values())).mean()
            log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                if self.args.dataset == "mosei":
                    torch.save(
                        {"args": self.args, "state_dict": self.model},
                        "model_checkpoints/mosei_best_dev_f1_model_"
                        + self.args.modalities
                        + "_"
                        + self.args.emotion
                        + ".pt",
                    )
                else:
                    torch.save(
                        {"args": self.args, "state_dict": self.model},
                        "model_checkpoints/"
                        + self.args.dataset
                        + "_best_dev_f1_model_"
                        + self.args.modalities
                        + ".pt",
                    )

                log.info("Save the best model.")
            log.info("[Test set] [f1 {:.4f}]".format(test_f1))

            dev_f1s.append(dev_f1)
            test_f1s.append(test_f1)
            train_losses.append(train_loss)
            if self.args.log_in_comet or self.args.tuning:
                self.args.experiment.log_metric("F1 Score (Dev)", dev_f1, epoch=epoch)
                self.args.experiment.log_metric("F1 Score (test)", test_f1, epoch=epoch)
                self.args.experiment.log_metric("train_loss", train_loss, epoch=epoch)
                self.args.experiment.log_metric("val_loss", dev_loss, epoch=epoch)

        if self.args.tuning:
            self.args.experiment.log_metric("best_dev_f1", best_dev_f1, epoch=epoch)
            self.args.experiment.log_metric("best_test_f1", best_test_f1, epoch=epoch)

            return best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s

        # The best

        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_f1, _ = self.evaluate()
        log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        test_f1, _ = self.evaluate(test=True)
        log.info("[Test set] f1 {}".format(test_f1))

        return best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        epoch_l_contrast=0
        epoch_l_pred=0
        self.model.train()

        self.trainset.shuffle()
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            # * generate batch for teacher data
            l_contrast, l_pred = None, None
            if "student_modality" in self.args.__dict__:
                self.trainset._update_modalities(self.args.modalities)
                teacher_data = self.trainset[idx]
                for k, v in teacher_data.items():
                    if not k == "utterance_texts":
                        teacher_data[k] = v.to(self.args.device)

                # * generate batch for student model
                self.trainset._update_modalities(self.args.student_modality)
                data = self.trainset[idx]
                for k, v in data.items():
                    if not k == "utterance_texts":
                        data[k] = v.to(self.args.device) 

                nll, l_contrast, l_pred = self.model.get_loss(data, teacher_data)
            else:
               # * generate batch for student model (this means we are training teacher)
                data = self.trainset[idx]
                for k, v in data.items():
                    if not k == "utterance_texts":
                        data[k] = v.to(self.args.device) 

                nll = self.model.get_loss(data) 
            epoch_loss += nll.item()
            if l_contrast is not None:
                epoch_l_contrast += l_contrast.item()
            if l_pred is not None:
                epoch_l_pred += l_pred.item()

            nll.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info(
            "[Epoch %d] [Loss: %f] [Time: %f]"
            % (epoch, epoch_loss, end_time - start_time)
        )
        # * log losses to tensorboard
        if self.writer is not None:
            self.writer.add_scalar("Loss/train_overall", epoch_loss, epoch)
            self.writer.add_scalar("Loss/train_contrast", epoch_l_contrast, epoch)
            self.writer.add_scalar("Loss/train_pred", epoch_l_pred, epoch)
        return epoch_loss

    def evaluate(self, test=False):
        dev_loss = 0
        dataset = self.testset if test else self.devset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    if not k == "utterance_texts":
                        data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))
                returned_losses = self.model.get_loss(data)
                # print("Returned losses: ", returned_losses)
                if isinstance(returned_losses, tuple):
                    nll, _, _ = returned_losses
                else:
                    nll = returned_losses
                # nll, _, _ = self.model.get_loss(data)
                dev_loss += nll.item()

            if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                golds = torch.cat(golds, dim=0).numpy()
                preds = torch.cat(preds, dim=0).numpy()
                f1 = metrics.f1_score(golds, preds, average="weighted")
                acc = metrics.accuracy_score(golds, preds)
                if self.args.tuning:
                    self.args.experiment.log_metric("dev_acc", acc)
            else:
                golds = torch.cat(golds, dim=-1).numpy()
                preds = torch.cat(preds, dim=-1).numpy()
                f1 = metrics.f1_score(golds, preds, average="weighted")

            if test:
                print(
                    metrics.classification_report(
                        golds, preds, target_names=self.label_to_idx.keys(), digits=4
                    )
                )

                if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                    happy = metrics.f1_score(
                        golds[:, 0], preds[:, 0], average="weighted"
                    )
                    sad = metrics.f1_score(golds[:, 1], preds[:, 1], average="weighted")
                    anger = metrics.f1_score(
                        golds[:, 2], preds[:, 2], average="weighted"
                    )
                    surprise = metrics.f1_score(
                        golds[:, 3], preds[:, 3], average="weighted"
                    )
                    disgust = metrics.f1_score(
                        golds[:, 4], preds[:, 4], average="weighted"
                    )
                    fear = metrics.f1_score(
                        golds[:, 5], preds[:, 5], average="weighted"
                    )

                    f1 = {
                        "happy": happy,
                        "sad": sad,
                        "anger": anger,
                        "surprise": surprise,
                        "disgust": disgust,
                        "fear": fear,
                    }

                if self.args.log_in_comet or self.args.tuning:
                    self.args.experiment.log_confusion_matrix(
                        golds,
                        preds,
                        labels=list(self.label_to_idx.keys()),
                        overwrite=True,
                    )

                    if (
                        self.args.dataset == "mosei"
                        and self.args.emotion == "multilabel"
                    ):
                        self.args.experiment.log_metric(
                            "accuracy score", metrics.accuracy_score(golds, preds)
                        )
                        self.args.experiment.log_metric("happiness_f1", happy)
                        self.args.experiment.log_metric("sadness_f1", sad)
                        self.args.experiment.log_metric("anger_f1", anger)
                        self.args.experiment.log_metric("surprise_f1", surprise)
                        self.args.experiment.log_metric("disgust_f1", disgust)
                        self.args.experiment.log_metric("fear_f1", fear)
                
            # * log to tensorboard
            if self.writer is not None:
                # convert gold labels to string using the mappiong
                # make new dict with key as index and value as emotion
                new_dict = {}
                for key, value in self.label_to_idx.items():
                    new_dict[value] = key
                
                gold_emotion = []
                for gold in golds:
                    gold_emotion.append(new_dict[gold])
                golds = np.array(gold_emotion)

                preds_emotion = []
                for pred in preds:
                    preds_emotion.append(new_dict[pred])
                preds = np.array(preds_emotion)

                cm = metrics.confusion_matrix(gold_emotion, preds_emotion)
                disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(values_format='d', cmap='Blues', ax=plt.gca())

                # * log to tensorboard
                is_test = 'test' if test else 'dev'
                self.writer.add_figure('Confusion Matrix'+is_test, plt.gcf(), 
                                    self.epoch_tracker_test+1 if test else self.epoch_tracker_dev+1)

                # * log accuracy and f1 scores
                if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                    pass
                else:
                    # golds = torch.cat(golds, dim=-1).numpy()
                    # preds = torch.cat(preds, dim=-1).numpy()

                    # * compute f1 score for each class and weight by the class proportion
                    f1_scores = metrics.f1_score(golds, preds, average=None)
                    acc = metrics.accuracy_score(golds, preds)

                    f1_dict = {}
                    for class_label, f1_c in enumerate(f1_scores):
                        f1_dict[str(class_label)] = f1_c

                    self.writer.add_scalars(
                        "F1_score/"+str(is_test)+"_classwise", f1_dict, 
                        self.epoch_tracker_test+1 if test else self.epoch_tracker_dev+1
                    )
                    # * overall f1 score
                    overall_f1_score = metrics.f1_score(golds, preds, average="weighted")
                    self.writer.add_scalar("F1_score/"+str(is_test)+"overall", overall_f1_score, 
                                    self.epoch_tracker_test+1 if test else self.epoch_tracker_dev+1)

 
                    # * log accuracy
                    self.writer.add_scalar("Accuracy/"+is_test, acc,
                                    self.epoch_tracker_test+1 if test else self.epoch_tracker_dev+1)

            # * update epoch tracker
            if test:
                self.epoch_tracker_test += 1
            else:
                self.epoch_tracker_dev += 1
        return f1, dev_loss
