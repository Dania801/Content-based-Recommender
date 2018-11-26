import sys
import pandas as pd
import numpy as np
import string
import matplotlib.pylab as plot
from collections import Counter



def importDataset(file):
    """
    import data set into a panda's dataframe
    @params file: filename to import
    @rtype {dataFrame}
    """
    columnNames = ['ClickScroll', 'DownArrowMs', 'VScrollMs', 'Relevant',
                   'PageUp', 'MouseMs', 'PageDown', 'ClickWindow',
                   'LogId', 'ServerTimeVisit', 'PageUpMs', 'UpArrowMs',
                   'ClickUpArrow', 'Classes', 'PageDownMs', 'RssId',
                   'Readability', 'HScrollMs', 'Novelty', 'UserLike',
                   'DocId', 'PageMs', 'TimeVisit', 'UserId',
                   'ClickDownArrow', 'Authority']
    data = pd.read_excel('{0}'.format(file), dtype=None, names=columnNames)
    return data


def extractRequiredDate(data):
    """
    Extract columns from dataframe with non-sparse values
    @params data: the imported dataset 
    @rtpe {dataFrame}
    """
    columnNames = ['Relevant', 'MouseMs', 'LogId',
                   'Classes', 'RssId', 'Readability', 'Novelty',
                   'PageMs', 'UserId', 'Authority', 'UserLike']
    return data[columnNames]


def removeNullValues(data):
    """
    Remove records with no classes 
    Replace blanks with zeros for features 'MouseMs' and 'PageMs'
    Replace blanks and -1 with min value for features 'Readability', 'Novelty', 'Authority' and 'Relevant'
    Remove negative user likes
    @params data: dataframe of the dataset
    @rtype {dataFrame}
    """
    dataWithClasses = data.dropna(subset=['Classes'])
    dataWithCleanClasses = dataWithClasses[dataWithClasses.Classes != '|']
    dataWithCleanMouseMs = dataWithCleanClasses.MouseMs.replace(np.nan, 0)
    dataWithCleanPageMs = dataWithCleanClasses.PageMs.replace(np.nan, 0)
    dataWithCleanClasses.MouseMs = dataWithCleanMouseMs
    dataWithCleanClasses.PageMs = dataWithCleanPageMs

    cleanData = data.dropna(subset=['Novelty'])
    cleanNovelty = cleanData[cleanData.Novelty.astype(int) > 0]
    minNovelty = cleanNovelty.Novelty.values.astype(float).min()

    cleanData = data.dropna(subset=['Relevant'])
    cleanRelevant = cleanData[cleanData.Relevant.values.astype(int) > 0]
    minRelevant = cleanRelevant.Relevant.values.astype(float).min()

    cleanData = data.dropna(subset=['Readability'])
    cleanReadability = cleanData[cleanData.Readability.values.astype(int) != -1]
    minReadability = cleanReadability.Readability.values.astype(float).min()

    cleanData = data.dropna(subset=['Authority'])
    cleanAuthority = cleanData[cleanData.Authority.values.astype(int) != -1]
    minAuthority = cleanAuthority.Authority.values.astype(float).min()

    dataWithCleanNovelty = dataWithCleanClasses.Novelty.replace(-1, minNovelty)
    dataWithCleanNovelty = dataWithCleanNovelty.replace(0, minNovelty)
    dataWithCleanNovelty = dataWithCleanNovelty.replace(np.nan, minNovelty)
    dataWithCleanRelevant = dataWithCleanClasses.Relevant.replace(-1, minRelevant)
    dataWithCleanRelevant = dataWithCleanRelevant.replace(0, minRelevant)
    dataWithCleanRelevant = dataWithCleanRelevant.replace(np.nan, minRelevant)
    dataWithCleanReadability = dataWithCleanClasses.Readability.replace(-1, minReadability)
    dataWithCleanReadability = dataWithCleanReadability.replace(np.nan, minReadability)
    dataWithCleanAuthority = dataWithCleanClasses.Authority.replace(-1, minAuthority)
    dataWithCleanAuthority = dataWithCleanAuthority.replace(np.nan, minAuthority)

    dataWithCleanClasses.Novelty = dataWithCleanNovelty
    dataWithCleanClasses.Relevant = dataWithCleanRelevant
    dataWithCleanClasses.Readability = dataWithCleanReadability
    dataWithCleanClasses.Authority = dataWithCleanAuthority

    dataWithCleanClasses = dataWithCleanClasses[dataWithCleanClasses.UserLike > 0]
    return dataWithCleanClasses


def tokenizeClasses(data):
    """
    Convert classes string into an array of tokens for each record in dataframe
    @params data: dataframe of the dataset
    @rtype {dataFrame, array}
    """
    classes = data.Classes
    punctuationList = [p for p in string.punctuation]
    punctuationList.remove('|')
    punctuationList.remove('/')
    punctuationList.remove('\\')
    tokensArr = []
    for i, entry in enumerate(classes):
        cleanEntry = "".join([letter for letter in entry if letter not in punctuationList])
        entryTokens = str(cleanEntry).lower().lstrip('|').split('|')
        tokensArr.append(entryTokens)
    classesArr = [ item for innerlist in tokensArr for item in innerlist ]
    data['Classes'] = tokensArr
    return data, classesArr


def getLowFrequencyClasses(classes):
    """
    Find classes with frequency = 1, and find the top highest frequency classes
    @params classes: array of all classes in dataframe
    @rtype {array}
    """
    classesCount = Counter(classes)
    names = list(classesCount.keys())
    values = list(classesCount.values())
    lowFrequencyClasses = [key for key, value in classesCount.items() if value == 1]
    highFrequencyClassesKeys = [key for key, value in classesCount.items() if value > 125]
    highFrequencyClassesValues = [value for key, value in classesCount.items() if value > 125]
    plot.title('Feature: Classes')
    plot.bar(highFrequencyClassesKeys,highFrequencyClassesValues)
    plot.xticks(rotation=30)
    plot.savefig('../Stats/hightest_freq_classes.png')
    # plot.show()
    return lowFrequencyClasses


def removeLowFreqClasses(data, classes):
    """
    Remove records with class frequency = 1
    @params data: dataframe of the dataset
    @params classes: array of distinct classes
    @rtype {dataFrame}
    """
    classesDF = data.Classes
    for i, entry in enumerate(classesDF):
        for j, token in enumerate(entry):
            if token in classes or token == '':
                entry.remove(token)
    cleanData = data[data.astype(str).Classes != '[]']
    return cleanData


def removeClassesAmbiguity(data):
    """
    Remove ambiguity from classes by replacing the classes referencing the same entity with a single class
    @params data: dataframe of the dataset
    @rtype {dataFrame}
    """
    classes = data.Classes
    classesArray = []
    tokensArray = []
    for i, entry in enumerate(classes):
        for j, token in enumerate(entry):
            if token == 'united states' or token == 'us':
                tokensArray.append('usa')
            elif token == 'world response to iraq war':
                tokensArray.append('iraq war')
            elif token == 'us news' or token == 'very short and simple news':
                tokensArray.append('news')
            elif token == 'fresh outlook on war':
                tokensArray.append('war')
            else:
                tokensArray.append(token)
        tokensArray = []
        classesArray.append(tokensArray)
    data['Classes'] = classesArray
    return data


def analyzeFeature(data, featureName):
    """
    Displays and write into a file the min/max/mean/variance values of a specific feature
    @params data: dataframe of the dataset
    @params featureName: the feature as a string
    @rtype {string}
    """
    if (data[featureName].dtype == 'int64' or data[featureName].dtype == 'float64'):
        minVal = data[featureName].values.astype(float).min()
        maxVal = data[featureName].values.astype(float).max()
        mean = data[featureName].values.astype(float).mean()
        variance = data[featureName].values.astype(float).var()
        featureInfo = '''
        Feature: {0}
        --------------------------
        Min value = {1}
        Max value = {2}
        Mean value = {3}
        Variance value = {4}
        --------------------------
        '''.format(featureName, minVal, maxVal, mean, variance)
        return featureInfo


def writeIntoDesk(data, fileName):
    """
    Write/create into a file
    @params data: dataframe of the dataset
    @params fileName
    """
    file = open('../Stats/{0}.txt'.format(fileName), 'w+')
    if data: 
        file.write(data)
    file.close()


def appendToDesk(data, fileName):
    """
    Append to file 
    @params data: dataframe of the dataset
    @params fileName
    """
    file = open('../Stats/{0}.txt'.format(fileName), 'a+')
    if data :
        file.write(data)
    file.close()


def datasetStats(data):
    """
    Displays and write into a file stats about features
    @params data: dataframe of the dataset
    """
    dataInfo = '''
    ----------------------
    Info about dataset 
    ----------------------
    {0}
    ---------------------------
    A sample of dataset records
    ---------------------------
    {1}
    ------------------
    Columns of dataset
    ------------------
    {2}
    --------------------
    The shape of dataset
    --------------------
    {3}
    --------------------
    The index of dataset
    --------------------
    {4}
    '''.format(data.info(), data.head(5), data.columns, data.shape, data.index)
    print (dataInfo)
    writeIntoDesk(dataInfo, 'dataInfo')
    writeIntoDesk('', 'featuresInfo')
    [appendToDesk (analyzeFeature(data, row), 'featuresInfo') for row in data]
    [print (analyzeFeature(data, row)) for row in data]
    relevant = data.Relevant
    plotStats(data, 'Relevant', 'relevant_freq', True)
    plotStats(data, 'Readability', 'readability_freq', True)
    plotStats(data, 'Novelty', 'novelty_freq', True)
    plotStats(data, 'Authority', 'authority_freq', True)


def plotStats(data, featureName, fileName, showFigure):
    """
    Save stats about feature as png image 
    @params data: dataframe of the dataset
    @params featureName: feature as a string
    @params fileName
    """
    feature = data[featureName]
    featureArr = []
    for i, entry in enumerate(feature):
        featureArr.append(entry)
    featureDict = Counter(featureArr)
    featureNames = list(featureDict.keys())
    featureValues = list(featureDict.values())
    plot.title('Feature: {0}'.format(featureName))
    plot.bar(range(len(featureDict)),featureValues,tick_label=featureNames)
    if showFigure :
        plot.savefig('../Stats/{0}.png'.format(fileName))
        plot.show()


def preprocessorScript():
    """
    Computes the stats and perform the preprocessing steps
    """
    file = '../Data/yow_userstudy_raw.xls'
    data = importDataset(file)
    extractedData = extractRequiredDate(data)
    nullData = removeNullValues(extractedData)
    datasetStats(nullData)
    data, classes = tokenizeClasses(nullData)
    classes = getLowFrequencyClasses(classes)
    data = removeLowFreqClasses(data, classes)
    data = removeClassesAmbiguity(data)
    data = removeLowFreqClasses(data, classes)
    data.to_csv('../Data/clean_data.csv')


def preprocessDataset():
    """
    Performs the preprocessing steps and returns clean data
    @rtype {dataFrame}
    """
    file = '../Data/yow_userstudy_raw.xls'
    data = importDataset(file)
    extractedData = extractRequiredDate(data)
    nullData = removeNullValues(extractedData)
    data, classes = tokenizeClasses(nullData)
    classes = getLowFrequencyClasses(classes)
    data = removeLowFreqClasses(data, classes)
    data = removeClassesAmbiguity(data)
    data = removeLowFreqClasses(data, classes)
    return data



# preprocessorScript()