import os
import librosa
import soundfile
# dataFilepath = "./Mel2test/"
tarFilepath = "F:/DATA/LJSpeech-1/LJSpeech-1.0-16k/"
filenames = os.listdir(tarFilepath)
# plFilepath = 'D:/VCwork-Py/waveglow-modified/pl_wave/'
# subfiles = os.listdir(plFilepath)
# with open("pl_files.txt", 'w+') as f:
#     for i in subfiles:
#         print(i)
#         tempfile = os.path.join(plFilepath, i)
#         # filenames = os.listdir(tempfile)
#         # for item in tempfile:
#         #     if item[-3:] != 'wav':
#         #         continue
#         #     else:
#         #         print(item)
#         #         tempname = os.path.join(tempfile, item)

#         f.write(tempfile+'\n')
# for item in filenames:
#     print(item)
#     temp = os.path.join(dataFilepath, item)
#     audio, sr = librosa.load(temp, sr=22050)
#     print("origin samplerate is: {}".format(sr))
#     audio_16 = librosa.resample(audio, sr, 16000)
#     print("target sample rate is: {}".format(16000))
#     soundfile.write(os.path.join(tarFilepath, item), audio_16, 16000)
#     print("successfully resample audio {}".format(item))

# print(len(filenames))
with open("train_files.txt", 'w+') as f:
    for item in filenames[:-100]:
        f.write('../waveglow-modified/LJSpeech-1.0-16k/'+item+'\n')
with open("test_files.txt", 'w+') as f:
    for item in filenames[-50:]:
        f.write('../waveglow-modified/LJSpeech-1.0-16k/'+item+'\n')
# with open("mel_files.txt", 'w+') as f:
#     for item in filenames:
#         f.write('Mel2test/'+item+'\n')