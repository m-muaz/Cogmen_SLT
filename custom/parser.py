import argparse

def speech_cogmen_parser():
    # * create a parser for the speech cogmen model
    parser = argparse.ArgumentParser(description="Training for speech cogmen")
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="iemocap_4",
        choices=["iemocap", "iemocap_4", "mosei"],
        help="Dataset name.",
    )
    ### adding other pre-trained text models
    parser.add_argument("--transformers", action="store_true", default=False)

    """ Dataset specific info (effects)
            -> tag_size in COGMEN.py
            -> n_speaker in COGMEN.py
            -> class_weights in classifier.py
            -> label_to_idx in Coach.py """

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )

    # Training parameters
    parser.add_argument(
        "--from_begin", action="store_true", help="Training from begin.", default=False
    )
    parser.add_argument("--model_ckpt", type=str, help="Training from a checkpoint.")

    parser.add_argument("--device", type=str, default="cuda", help="Computing device.")
    parser.add_argument(
        "--epochs", default=1, type=int, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "rmsprop", "adam", "adamw"],
        help="Name of optimizer.",
    )
    parser.add_argument(
        "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="Weight decay."
    )
    parser.add_argument(
        "--max_grad_value",
        default=-1,
        type=float,
        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""",
    )
    parser.add_argument("--drop_rate", type=float, default=0.5, help="Dropout rate.")

    # Model parameters
    parser.add_argument(
        "--wp",
        type=int,
        default=5,
        help="Past context window size. Set wp to -1 to use all the past context.",
    )
    parser.add_argument(
        "--wf",
        type=int,
        default=5,
        help="Future context window size. Set wp to -1 to use all the future context.",
    )
    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers.")
    parser.add_argument(
        "--hidden_size", type=int, default=100, help="Hidden size of two layer GCN."
    )
    parser.add_argument(
        "--rnn",
        type=str,
        default="transformer",
        choices=["lstm", "gru", "transformer"],
        help="Type of RNN cell.",
    )
    parser.add_argument(
        "--class_weight",
        action="store_true",
        default=False,
        help="Use class weights in nll loss.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        choices=["relational", "relative", "multi"],
        help="Type of positional encoding",
    )
    parser.add_argument(
        "--trans_encoding",
        action="store_true",
        default=False,
        help="Use dynamic embedding or not",
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
        choices=["a", "t", "v", "at", "tv", "av", "atv"],
        help="Modalities",
    )

    parser.add_argument(
    "--student_modality",
    type=str,
    default="a",
    # required=True,
    choices=["a", "t", "v", "at", "tv", "av", "atv"],
    help="Input Modalities for Student Model",
    )

    # emotion
    parser.add_argument(
        "--emotion", type=str, default=None, help="emotion class for mosei"
    )

    # Teacher architecutre ckpt location
    parser.add_argument("--teacher_model", type=str, default="model", help="Training from a checkpoint.")


    # Model Architecture changes
    parser.add_argument("--concat_gin_gout", action="store_true", default=False)
    parser.add_argument("--seqcontext_nlayer", type=int, default=2)
    parser.add_argument("--gnn_nheads", type=int, default=1)
    parser.add_argument("--num_bases", type=int, default=7)
    parser.add_argument("--use_highway", action="store_true", default=False)

    # others
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    # Logging
    ## Comet
    parser.add_argument(
        "--log_in_comet",
        action="store_true",
        default=False,
        help="Logs the experiment data to comet.ml",
    )
    parser.add_argument(
        "--comet_api_key",
        type=str,
        help="comet api key, required for logging experiments on comet.ml",
    )
    parser.add_argument(
        "--comet_workspace",
        type=str,
        help="comet comet_workspace, required for logging experiments on comet.ml",
    )

    ## tensorboard logging
    parser.add_argument(
            "--log_in_tensorboard",
            action="store_true",
            default=False,
            help="Logs the experiment data to tensorboard (Locally)",
    ) 
    parser.add_argument(
        "--tb_log_dir",
        type=str,
        default="./tensorboard_logdir",
        help="Directory to store tensorboard logs",
    )


    parser.add_argument("--use_pe_in_seqcontext", action="store_true", default=False)
    parser.add_argument("--tuning", action="store_true", default=False)
    parser.add_argument("--tag", type=str, default="hyperparameters_opt")

    # Experiment details
    parser.add_argument("--res_dir", type=str, default="./work_dirs", help="Directory to store results")
    parser.add_argument("--ex_name", type=str, default="speech_cogmen", help="Experiment name")
    parser.add_argument("--exp_mode", type=str, default="train", help="Experiment mode: train/eval")

    # args = parser.parse_args()

    # args.dataset_embedding_dims = {
    #     "iemocap": {
    #         "a": 100,
    #         "t": 768,
    #         "v": 512,
    #         "at": 100 + 768,
    #         "tv": 768 + 512,
    #         "av": 612,
    #         "atv": 100 + 768 + 512,
    #     },
    #     "iemocap_4": {
    #         "a": 100,
    #         "t": 768,
    #         "v": 512,
    #         "at": 100 + 768,
    #         "tv": 768 + 512,
    #         "av": 612,
    #         "atv": 100 + 768 + 512,
    #     },
    #     "mosei": {
    #         "a": 80,
    #         "t": 768,
    #         "v": 35,
    #         "at": 80 + 768,
    #         "tv": 768 + 35,
    #         "av": 80 + 35,
    #         "atv": 80 + 768 + 35,
    #     },
    # }

    return parser


    # log.debug(args)