import sys
import pandas as pd
import numpy as np
import string
import matplotlib.pylab as plot
from collections import Counter
import pickle
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from extractor import addExtractedFeatures
from sklearn import metrics


def LinearRegressionModel():
    data = addExtractedFeatures()
    trainData = data [0:int(len(data)/2)]
    testData = data[int(len(data)/2):]
    columns = ['Relevant', 'MouseMs', 'Readability', 'Novelty', 'PageMs', 'Authority', 'ConditionalFreq', 'MeanConditionalFreq']
    xTrainData = trainData[columns]
    yTrainData = trainData['UserLike']
    xTestData = testData[columns]
    zTestData = testData['UserLikeBinary']
    lm = LinearRegression()
    lm.fit(xTrainData, yTrainData)
    zTestData = zTestData.values
    predictions = lm.predict(xTestData)
    predictions = [int(i) for i in predictions]
    predictionsArray = []
    for i, entry in enumerate(predictions):
        if (predictions[i] >= 4):
            predictionsArray.append(1)
        else:
            predictionsArray.append(0)
    predictions = [int(i) for i in predictions]
    predictionsArray = []
    for i, entry in enumerate(predictions):
        if (predictions[i] >= 4):
            predictionsArray.append(1)
        else:
            predictionsArray.append(0)
    accuracy = metrics.accuracy_score(zTestData, predictionsArray)
    print('------------------------------------------------------------')
    print('Linear Regression model accuracy = {0}%'.format(accuracy*100))
    print('------------------------------------------------------------')


def LogisticRegressionModel():
    data = addExtractedFeatures()
    trainData = data [0:int(len(data)/2)]
    testData = data[int(len(data)/2):]
    columns = ['Relevant', 'MouseMs', 'Readability', 'Novelty', 'PageMs', 'Authority', 'ConditionalFreq', 'MeanConditionalFreq']
    xTrainData = trainData[columns]
    yTrainData = trainData['UserLike']
    xTestData = testData[columns]
    zTestData = testData['UserLikeBinary']
    lm = LogisticRegression()
    lm.fit(xTrainData, yTrainData)
    predictions = lm.predict(xTestData)
    zTestData = zTestData.values
    predictions = [int(i) for i in predictions]
    predictionsArray = []
    for i, entry in enumerate(predictions):
        if (predictions[i] >= 4):
            predictionsArray.append(1)
        else:
            predictionsArray.append(0)
    accuracy = metrics.accuracy_score(zTestData, predictionsArray)
    
    print('------------------------------------------------------------')
    print('Logistic Regression model accuracy = {0}%'.format(accuracy*100))
    print('------------------------------------------------------------')

def NaiveBayesModel():
    data = addExtractedFeatures()
    trainData = data [0:int(len(data)/2)]
    testData = data[int(len(data)/2):]
    columns = ['Relevant', 'MouseMs', 'Readability', 'Novelty', 'PageMs', 'Authority', 'ConditionalFreq', 'MeanConditionalFreq']
    xTrainData = trainData[columns]
    yTrainData = trainData['UserLike']
    xTestData = testData[columns]
    zTestData = testData['UserLikeBinary']
    lm = GaussianNB()
    lm.fit(xTrainData, yTrainData)
    zTestData = zTestData.values
    predictions = lm.predict(xTestData)
    predictions = [int(i) for i in predictions]
    predictionsArray = []
    for i, entry in enumerate(predictions):
        if (predictions[i] >= 4):
            predictionsArray.append(1)
        else:
            predictionsArray.append(0)
    predictions = [int(i) for i in predictions]
    predictionsArray = []
    for i, entry in enumerate(predictions):
        if (predictions[i] >= 4):
            predictionsArray.append(1)
        else:
            predictionsArray.append(0)
    accuracy = metrics.accuracy_score(zTestData, predictionsArray)
    print('------------------------------------------------------------')
    print('Naive Bayes model accuracy = {0}%'.format(accuracy*100))
    print('------------------------------------------------------------')

# NaiveBayesModel()