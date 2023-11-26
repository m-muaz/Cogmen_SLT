# model 
seqcontext_nlayer=4
gnn_nheads=7
# dataset
dataset = "iemocap_4"
# teacher modality
modalities = "av"
# training
drop_rate=0.1
learning_rate=1e-4
epochs = 55
from_begin=True
# teacher model name
teacher_model = "Cogmen_av_2"
# tensorboard
log_in_tensorboard=True
tb_log_dir = "tensorboard_logdir"
ex_name="Cogmen_a_iemocap_4_"
