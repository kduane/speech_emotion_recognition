import librosa as lb
import os, glob, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# file name parsing dictionaries

emotions = {
    '01' : 'neutral',
    '02' : 'calm',
    '03' : 'happy',
    '04' : 'sad',
    '05' : 'angry',
    '06' : 'fearful',
    '07' : 'disgust',
    '08' : 'surprised'
}
vocal_channels = {
    '01' : 'speech',
    '02' : 'song'
}
emotional_intensities = {
    '01' : 'normal',
    '02' : 'strong'
}
statements = {
    '01' : 'Kids are talking by the door',
    '02' : 'Dogs are sitting by the door'
}

def extract_feature(file, mfcc, chroma, mel):
    audio, sample_rate = lb.load(file)
    stft=np.abs(lb.stft(audio))
    result = []
    
    if mfcc:
        mfccs=np.mean(lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        result.extend(mfccs)
#         print(f"Len after mfcc: {len(result)}")
    if chroma:
        chroma_r=np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result.extend(chroma_r)
#         print(f"Len after chroma: {len(result)}")
    if mel:
        mel=np.mean(lb.feature.melspectrogram(audio, sr=sample_rate).T,axis=0)
        result.extend(mel)
#         print(f"Len after mel: {len(result)}")
    return result

def load_targets(target_emotions, target_actors, target_channels = ['song', 'speech']):
    #load files that contain target emotion
    sounds = []
    
    for file in glob.glob('./data/samples/*.wav'):
        file_name = os.path.basename(file)
        
        #filter out non-target channels- set to all by default
        channel = vocal_channels[file_name.split('-')[1]]
        if channel not in target_channels:
            continue
            
        #filter out non-target emotions
        emotion = emotions[file_name.split('-')[2]]
        if emotion not in target_emotions:
            continue
            
        #filter out non-target actors
        actor = file_name.split('-')[6][:-4]
        if actor not in target_actors:
            continue
        
        sound_dict = {
            'file_name'  : file_name[:-4],
            'emotion'    : emotions[file_name.split("-")[2]],
            'feature'    : extract_feature(file, mfcc = True, chroma = True, mel = True)
        }
        
        sounds.append(sound_dict)
        
    return pd.DataFrame.from_dict(sounds)