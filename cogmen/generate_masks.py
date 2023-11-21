import numpy as np
import torch

def generate_training_mask(cfg):
    p_mask_audio_full = cfg['pmaf']
    p_mask_video_full = cfg['pmvf']
    p_mask_audio_rand = cfg['pmar']
    p_mask_video_rand = cfg['pmvr']

    audio_mask_length = cfg['aml']
    video_mask_length = cfg['vml']

    audio_input_length = cfg['ail']
    video_input_length = cfg['vil']
    #text_input_length = cfg['til']

    #text_mask = np.ones(text_input_length)

    p = np.random.random()

    if p<p_mask_audio_full:
        # fully mask audio
        audio_mask = np.zeros(audio_input_length)
        video_mask = np.ones(video_input_length)
    elif p<(p_mask_video_full + p_mask_audio_full):
        # fully mask video
        audio_mask = np.ones(audio_input_length)
        video_mask = np.zeros(video_input_length)
    else:
        # random masking

        audio_mask = np.ones(audio_input_length)
        video_mask = np.ones(video_input_length)

        audio_mask_start_ids = np.where(np.random.random(size=audio_input_length,) < p_mask_audio_rand)
        video_mask_start_ids = np.where(np.random.random(size=video_input_length) < p_mask_video_rand)

        audio_mask_id_offsets = np.arange(audio_mask_length)
        video_mask_id_offsets = np.arange(video_mask_length)

        audio_mask_ids = np.unique(np.add.outer(audio_mask_start_ids, audio_mask_id_offsets).flatten())
        audio_mask_ids = audio_mask_ids[audio_mask_ids<audio_input_length]

        video_mask_ids = np.unique(np.add.outer(video_mask_start_ids, video_mask_id_offsets).flatten())
        video_mask_ids = video_mask_ids[video_mask_ids<video_input_length]

        audio_mask[audio_mask_ids] = 0
        video_mask[video_mask_ids] = 0

    #return np.concatenate([audio_mask, text_mask, video_mask])
    return np.concatenate([audio_mask, video_mask])

def get_inference_mask(cfg):
    audio_input_length = cfg['ail']
    video_input_length = cfg['vil']
    #text_input_length = cfg['til']

    #text_mask = np.ones(text_input_length)
    audio_mask = np.ones(audio_input_length)
    video_mask = np.zeros(video_input_length)

    #return np.concatenate([audio_mask, text_mask, video_mask])
    return np.concatenate([audio_mask, video_mask])

"""
if __name__ == '__main__':
    cfg = {
        'pmaf': 0.0,    # probability of fully masking audio
        'pmvf': 0.0,    # probability of fully masking video
        'pmar': 0.5,    # probability of masking an audio sample randomly
        'pmvr': 0.5,    # probability of masking a video sample randomly
        'aml': 1,       # audio mask length
        'vml': 3,       # video mask length
        'ail': 5,       # size of audio input 
        'vil': 10,      # size of video input
    }

    mask = generate_training_mask(cfg)
    print()
    print(mask)
"""