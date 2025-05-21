import pickle as pkl
with open("dataset/meld_features_emoberta.pkl", 'rb') as f:
    iemocap = pkl.load(f)

videoIDs, videoSpeakers, videoLabels, videoText,\
            videoAudio, videoVisual,videoSentence, trainVid,\
            testVid, _ , textFea = iemocap
print(videoSpeakers)









