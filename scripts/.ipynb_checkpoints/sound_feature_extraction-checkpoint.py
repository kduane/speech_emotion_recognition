import librosa as lb
import os, glob, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

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

def extract_features(audio, sample_rate, mfcc = False, chroma = False, mel = False):
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

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

def extract_mfccs(audio, sample_rate):
    result = lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    if result.shape[1] != 275:
        result = pad_along_axis(result, 275, axis = 1)
    result = scale(result, axis = 1)
    return result

def extract_melspec(audio, sample_rate):
    result = lb.feature.melspectrogram(y = audio, sr = sample_rate)
    if result.shape[1] != 275:
        result = pad_along_axis(result, 275, axis = 1)
    result = scale(result, axis = 1)
    return result

def extract_chroma(audio, sample_rate):
    result = lb.feature.chroma_stft(audio, sample_rate)
    if result.shape[1] != 275:
        result = pad_along_axis(result, 275, axis = 1)

    result = scale(result, axis = 1)    
    return result

def extract_feature_array(audio, sample_rate, mfcc = False, chroma = False, mel = False):
    result = []
    
    if mfcc:
        mfccs = extract_mfccs(audio, sample_rate)
    
    if chroma:
        chroma = extract_chroma(audio, sample_rate)
    
    if mel:
        mel = extract_melspec(audio, sample_rate)

    result = np.concatenate((mfccs, chroma, mel), axis = 0)
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
        
        wave, sample_rate = lb.load(file)
        duration = lb.get_duration(wave, sample_rate)
            
        sound_dict = {
            'file_name'       : file_name[:-4],
            'emotion'         : emotions[file_name.split("-")[2]],
            'statement'       : statements[file_name.split('-')[4]],
            'channel'         : vocal_channels[file_name.split('-')[1]],
            'mfccs'           : extract_mfccs(wave, sample_rate),
            'melspec'         : extract_melspec(wave, sample_rate),
            'chroma'          : extract_chroma(wave, sample_rate),
            'wave'            : wave,
            'duration'        : duration,
            'sr'              : sample_rate,
            'flat_feature'    : extract_features(wave, sample_rate, mfcc = True, chroma = True, mel = True),
            'feature_array'   : extract_feature_array(wave, sample_rate, mfcc = True, chroma = True, mel = True)
        }
        
        sounds.append(sound_dict)
        
    return pd.DataFrame.from_dict(sounds)

