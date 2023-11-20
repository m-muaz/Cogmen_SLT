import torch
import torch.nn as nn
import numpy as np
from custom.SeqContext import SeqContext
from cogmen.model.GNN import GNN
from custom.Classifier import Classifier
from cogmen.model.functions import batch_graphify
import cogmen

log = cogmen.utils.get_logger()


class COGMEN(nn.Module):
    def __init__(self, args):
        super(COGMEN, self).__init__()
        u_dim = 100
        if args.rnn == "transformer":
            g_dim = args.hidden_size
        else:
            g_dim = 200
        h1_dim = args.hidden_size
        h2_dim = args.hidden_size
        hc_dim = args.hidden_size
        dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
        }

        dataset_speaker_dict = {
            "iemocap": 2,
            "iemocap_4": 2,
            "mosei": 1,
        }

        if args.dataset and args.emotion == "multilabel":
            dataset_label_dict["mosei"] = {
                "happiness": 0,
                "sadness": 1,
                "anger": 2,
                "surprise": 3,
                "disgust": 4,
                "fear": 5,
            }

        tag_size = len(dataset_label_dict[args.dataset])
        args.n_speakers = dataset_speaker_dict[args.dataset]
        self.concat_gin_gout = args.concat_gin_gout

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.rnn = SeqContext(u_dim, g_dim, args)
        self.gcn = GNN(g_dim, h1_dim, h2_dim, args)
        if args.concat_gin_gout:
            self.clf = Classifier(
                g_dim + h2_dim * args.gnn_nheads, hc_dim, tag_size, args
            )
        else:
            self.clf = Classifier(h2_dim * args.gnn_nheads, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + "0"] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + "1"] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

        # * get teacher network gcn embeddings
        self.teacher = None
        # * initalize pairwise distance
        self.pdist = nn.PairwiseDistance(p=2)
        self.margin = 1.0
        self.triplet_loss = TripletLossCustom(distance_function=self.pdist, margin=self.margin, device=self.device)
        self.alpha_l_contrast = 1
        self.alpha_l_pred = 1

    def get_rep(self, data):
        # [batch_size, mx_len, D_g]
        node_features = self.rnn(data["text_len_tensor"], data["input_tensor"])
        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            node_features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.edge_type_to_idx,
            self.device,
        )

        graph_out = self.gcn(features, edge_index, edge_type)
        return graph_out, features

    def forward(self, data):
        graph_out, features = self.get_rep(data)

        if self.concat_gin_gout:
            out = self.clf(
                torch.cat([features, graph_out], dim=-1), data["text_len_tensor"]
            )
        else:
            out = self.clf(graph_out, data["text_len_tensor"])

        return out

    def get_loss(self, data, teacher_embeddings=None):
        graph_out, features = self.get_rep(data)
        
        # * apart from the classifier loss, we also need a loss term for the contrastive loss
        # * use the pos neg samples passed to the class
        # ! NOTE: Contrastive loss used here is the n-way triplet loss with euclidean distance

        if teacher_embeddings is not None:
            L_contrast = self.triplet_loss(graph_out, teacher_embeddings)
        else:
            L_contrast = None

        if self.concat_gin_gout:
            L_pred = self.clf.get_loss(
                torch.cat([features, graph_out], dim=-1),
                data["label_tensor"],
                data["text_len_tensor"],
            )
        else:
            L_pred = self.clf.get_loss(
                graph_out, data["label_tensor"], data["text_len_tensor"]
            )
            
        if teacher_embeddings is not None: # * i.e. during training
            loss = self.alpha_l_contrast * L_contrast + self.alpha_l_pred * L_pred
        else:
            loss = L_pred
        return loss, L_contrast, L_pred
    
    def set_teacher_emb(self, new_data):
        self.teacher = new_data


class TripletLossCustom(nn.Module):
    def __init__(self, distance_function=None, margin=1.0, device="cpu") -> None:
        super(TripletLossCustom, self).__init__()
        self.distance_function = distance_function
        self.margin = margin
        self.device=device
        self.relu = nn.ReLU()
    
    def forward(self, anchor, pos_neg_samples):
        L_contrast = 0.0
        graph_out = anchor
        for idx, student_sample in enumerate(graph_out):
            tmp_loss = 0.0
            pos_neg_sample = pos_neg_samples[idx]
            pos_sample = pos_neg_sample[0]
            tmp_loss = len(pos_neg_sample[1:]) * self.distance_function(student_sample, torch.tensor(pos_sample[0]).to(self.device))
            for neg_sample in pos_neg_sample[1:]:
                tmp_loss += (-1.0 * self.distance_function(student_sample, torch.tensor(neg_sample[0]).to(self.device)))
            
            # print(tmp_loss)
            L_contrast += self.relu(tmp_loss + self.margin)
        
        return torch.mean(L_contrast)