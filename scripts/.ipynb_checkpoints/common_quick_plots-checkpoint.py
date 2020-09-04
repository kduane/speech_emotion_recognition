import librosa as lb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix

def plot_spectrograms_and_waveform(y):
    sr = 22050 #standard librosa sample rate.
    fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize=(12, 12))
    D = lb.amplitude_to_db(np.abs(lb.stft(y)), ref=np.max)
    img = lb.display.specshow(D, y_axis='linear', x_axis='time',
                                sr=sr, ax=ax[0])
    ax[0].set(title='Linear-frequency power spectrogram')
    ax[0].label_outer()
    hop_length = 1024
    D = lb.amplitude_to_db(np.abs(lb.stft(y, hop_length=hop_length)),
                                ref=np.max)
    lb.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
                            x_axis='time', ax=ax[1])
    ax[1].set(title='Log-frequency power spectrogram')
    ax[1].label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    lb.display.waveplot(y, sr)
    ax[2].set(title='Waveplot');
    
    
def plot_waves(data, emotions, channels = ['speech'], figsize = (10, 10)):    
    subset = data[(data['emotion'].isin(emotions)) & (data['channel'].isin(channels))]
    sr = 22050

    if len(emotions) == 1:
        fig, ax = plt.subplots(nrows = int(subset.shape[0]), 
                               ncols = 1,
                               sharex = True,
                               figsize=figsize)
        row_i = 0
        for index, row in subset.iterrows():
            if row['emotion'] == emotions[0]:
                ax[row_i]
                lb.display.waveplot(row['wave'], sr, ax=ax[row_i])
                ax[row_i].set_title(f"{row['emotion']} {row['channel']}")
                row_i += 1
    else:
        fig, axs = plt.subplots(nrows = int(subset.shape[0] / len(emotions)),
                    ncols = len(emotions),
                    sharey = 'row',
                    figsize = figsize)
        for col_i, emotion in enumerate(emotions):
            row_i = 0
            for index, row in subset.iterrows():
                if row['emotion'] == emotion:
                    axs[row_i, col_i]
                    lb.display.waveplot(row['wave'], sr, ax=axs[row_i, col_i])
                    axs[row_i, col_i].set_title(f"{row['emotion']} {row['channel']}")
                    row_i += 1
                    
    plt.tight_layout()
    plt.show();
    


def plot_log_spectrograms(data, emotions, figsize):
    song_i = 0
    speech_i = 0
    subset = data[data['emotion'].isin(emotions) ]
    length = int(subset.shape[0] / 2)
    sr = 22050
    hop_length = 1024
    if len(emotions) == 1:
        fig, ax = plt.subplots(nrows = int(subset.shape[0]), 
                               ncols = 1,
                               sharex = True,
                               figsize=figsize)
        row_i = 0
        for index, row in subset.iterrows():
            if row['emotion'] == emotions[0]:
                ax[row_i]
                y = row['wave']
                D = lb.amplitude_to_db(np.abs(lb.stft(y, hop_length=hop_length)),
                                ref=np.max)
                lb.display.specshow(D, y_axis = 'log', sr=sr, hop_length=hop_length,
                            x_axis='time', ax=ax[row_i])
                ax[row_i].set_title(f"{row['emotion']} {row['channel']}")
                row_i += 1
    else:
        fig, axs = plt.subplots(nrows = int(subset.shape[0] / len(emotions)),
                    ncols = len(emotions),
                    sharey = 'row',
                    figsize = figsize)
        for col_i, emotion in enumerate(emotions):
            row_i = 0
            for index, row in subset.iterrows():
                if row['emotion'] == emotion:
                    y = row['wave']
                    D = lb.amplitude_to_db(np.abs(lb.stft(y, hop_length=hop_length)),
                                        ref=np.max)
                    lb.display.specshow(D, y_axis = 'log',
                                        sr=sr, hop_length=hop_length,
                                        x_axis='time', ax=axs[row_i, col_i])
                    axs[row_i, col_i].set_title(f"{row['emotion']}")
                    row_i += 1 
    
    plt.tight_layout()
    plt.show();
    

def plot_loss(res, title, figsize = (12,8)):
    train_loss = res.history['loss']
    test_loss = res.history['val_loss']
  
    plt.figure(figsize = figsize)
    plt.plot(train_loss, label='Training loss', color='navy')
    plt.plot(test_loss, label='Testing loss', color='skyblue')
    
    plt.title(title)
    plt.legend();

    
def plot_accuracy(res, title, figsize = (12,8)):
    train_accuracy = res.history['accuracy']
    test_accuracy = res.history['val_accuracy']

    plt.figure(figsize = figsize)
    plt.plot(train_accuracy, label = 'Training Accuracy', color = 'lightgreen')
    plt.plot(test_accuracy, label = 'Testing Accuracy', color = 'darkgreen')
    
    plt.title(title)
    plt.legend();

    
def plot_confusion_matrix(model, X_test, y_test, title):
    y_pred_pct = model.predict(X_test)
    label_dict = dict(enumerate(y_test.columns))
    y_pred = np.vectorize(label_dict.get)(np.argmax(y_pred_pct, axis = 1))
    y_test_flat = pd.Series(y_test.columns[np.where(y_test!=0)[1]])
    cm = confusion_matrix(y_test_flat, y_pred)
    
    sns.heatmap(cm, annot = True, cmap = 'Greens', xticklabels = y_test.columns, yticklabels= y_test.columns)
    plt.title(title);