import sys
import pandas as pd
import numpy as np
import string
import matplotlib.pylab as plot
from preprocessor import preprocessDataset, plotStats, tokenizeClasses
from collections import Counter
import pickle



def computeLikeFreq():
    data = preprocessDataset()
    userLikes = data.UserLike
    plotStats(data, 'UserLike', 'user_likes_freq')
    userLikesDict = Counter(userLikes)
    return userLikesDict

def classesJointFreq():
    data = preprocessDataset()
    classes = data.Classes
    classesArr = []
    for i, entry in enumerate(classes):
        for j, token in enumerate(entry):
            classesArr.append(token)
    distinctClasses = list(set(classesArr))
    columns = ['Classes', 'UserLike']
    data = data[columns]
    classesDict = {}
    for j, token in enumerate(distinctClasses):
        tokenArray = np.array([0,0,0,0,0])
        for i, entry in data.iterrows():
            if token in entry['Classes'] and entry['UserLike'] == 1:
                tokenArray[0]+=1
            if token in entry['Classes'] and entry['UserLike'] == 2:
                tokenArray[1]+=1
            if token in entry['Classes'] and entry['UserLike'] == 3:
                tokenArray[2]+=1
            if token in entry['Classes'] and entry['UserLike'] == 4:
                tokenArray[3]+=1
            if token in entry['Classes'] and entry['UserLike'] == 5:
                tokenArray[4]+=1
        classesDict[token] = tokenArray
        tokenArray = np.array([0,0,0,0,0])
        print ('{0}- {1}'.format(j, token))
    with open('../Data/joint_freq.pickle', 'wb') as handle:
        pickle.dump(classesDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


classesJointFreq()