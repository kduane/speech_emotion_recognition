import soundfile
import os, glob

from pydub import AudioSegment

for file in glob.glob('./data/Audio_Speech_Actors_01-24/*/*.wav'):
    #read in the file
    sound = AudioSegment.from_wav(file)
    #get the base name
    file_name = os.path.basename(file)
    #check if stero or mono
    sound = sound.set_channels(1) #set to mono
    #export to bulk file location as .wav
    sound.export(f"./data/samples/{file_name}", format="wav")
