from transformers import Wav2Vec2Processor, HubertModel, SequenceFeatureExtractor
from datasets import load_dataset
import soundfile as sf

import numpy as np 
import pandas as pd 
import datasets
from datasets import Audio, Dataset
import torch, torchaudio
import torch.nn as nn
import pickle


audio_meta = pd.read_csv('iemocap_full_dataset.csv')
audio_meta = audio_meta[audio_meta['n_annotators'] != 0]
audio_meta = audio_meta[audio_meta['emotion'] != 'dis']
AUDIO_PATH_PREFIX = '/scratch/06519/jahnavi/SLT/IEMOCAP_full_release/'

wav_folders = audio_meta['path'].str.split('/').str[-2].tolist()
#print("\n\nwav_folders:",wav_folders)

audio_meta['path'] = AUDIO_PATH_PREFIX + audio_meta['path'] 
audio_meta['wav_folder']=wav_folders




#filtered to only 4 emotions
iemocap_4 = {"hap": 0, "sad": 1, "neu": 2, "ang": 3}
audio_meta_4_orig = audio_meta[(audio_meta['emotion'] == 'ang') | 
                          (audio_meta['emotion'] == 'neu') | 
                          (audio_meta['emotion'] == 'sad') | 
                          (audio_meta['emotion'] == 'hap')] 

#This is used to exclude few rows, make sure to comment it later
audio_meta_4_filterd = audio_meta_4_orig#[~audio_meta_4_orig['wav_folder'].isin(vids_1)]

grouped_df = audio_meta_4_filterd.groupby('wav_folder')


# Load pre-trained model and processor
processor = SequenceFeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


reduced_size = 100
hidden_reduced_size = 1024
linear_projection = nn.Linear(hidden_reduced_size, reduced_size)

video_ids = []

file_path = './data/iemocap_4/hubert_iemocap_4_features_audio.pkl'
file_path_label = './data/iemocap_4/hubert_iemocap_4_features_label.pkl'
file_path_vids = './data/iemocap_4/hubert_iemocap_4_features_vids.pkl'




video_labels = {}
# Access each group as a separate DataFrame
for category, audio_meta_4 in grouped_df:

    video_ids.append(category)
    print(f"DataFrame for Category {category}")#:\n{audio_meta_4}\n)
    folder = category
    wav_paths_4 = audio_meta_4['path'].tolist()
    #labels_4 = audio_meta_4['emotion'].tolist()
    labels_4 = audio_meta_4['emotion'].map(iemocap_4).tolist()
    wav_folders_4 = audio_meta_4["wav_folder"].tolist()
    audio_dataset_4 = Dataset.from_dict({"audio": wav_paths_4, 'label': labels_4, 'wav_folder': wav_folders_4}).cast_column("audio", Audio())


    print("length of rest of the video ids:",len(video_ids)) 
    
    #video_audio = {}
    
    #print("audio dataset_4:", audio_meta_4)
    

    video_audio = {}
    video_labels[folder]=[]
    video_audio[folder]=[]
    print("folder:",folder)

    for sample in audio_dataset_4:

        label = sample['label']
        #folder = category
        input_audio = sample['audio']['array']
        #input_values = processor(input_audio, return_tensors="pt",sampling_rate=16000).input_values  # Batch size 1

        input_values = processor(input_audio, return_tensors="pt",sampling_rate=16000,feature_size=100).input_values  # Batch size 1
   
        #reduced_features = input_values[0].detach().numpy()
        #print("length of input features:",len(reduced_features))
        #print("input values:",input_values)
        #print("reduced features:",reduced_features)

        output = model(input_values)
        hidden_states = output.last_hidden_state
        reduced_features = hidden_states.mean(dim=1)

        reduced_features = linear_projection(reduced_features)
        
        video_labels[folder].append(label) 
        video_audio[folder].append(reduced_features)
        #break
    
    #with open(file_path, 'rb') as f2:
    #    additional_dict = pickle.load(f2)
    
    #additional_dict.update(video_audio)

    # Open the file in binary write mode and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(video_audio, file)

    
    # Open the file in binary write mode and save the data
    with open(file_path_label, 'wb') as file:
        pickle.dump(video_labels, file)

    with open(file_path_vids, 'wb') as file:
        pickle.dump(video_ids, file)
    print(f'Data has been saved to {file_path}')





#with open(file_path, 'rb') as f2:
#         audio_data = pickle.load(f2)

data_to_save = [video_ids,video_labels,video_audio]
file_path = './data/iemocap_4/hubert_iemocap_4_features.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(data_to_save, file)




