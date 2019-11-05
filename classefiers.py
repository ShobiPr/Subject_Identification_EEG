from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from features import get_features
import math
import numpy as np
import warnings

warnings.filterwarnings("ignore")


"""
CLASSIFIERS = [lambda l_dt, l_tt: random_forest(l_dt, l_tt),
               lambda l_dt, l_tt: decision_tree(l_dt, l_tt),
               lambda l_dt, l_tt: knn(l_dt, l_tt),
               lambda l_dt, l_tt: SVM(l_dt, l_tt),
               lambda l_dt, l_tt: naive_bayes(l_dt, l_tt)]
"""

def decision_tree(dataTraining, targetTraining):
    '''
    Creating decision tree classifier
    '''
    clf = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(dataTraining, targetTraining)
    scores = cross_val_score(clf, dataTraining, targetTraining, cv=C_F_V)
    accuracy = scores.mean()
    return {"classifier": "DecisionTree", "accuracy": str(accuracy), "clf": clf}


def naive_bayes(dataTraining, targetTraining):
    '''
    Creating naive Bayes classifier
    '''
    clf = GaussianNB().fit(dataTraining, targetTraining)
    scores = cross_val_score(clf, dataTraining, targetTraining, cv=C_F_V)
    accuracy = scores.mean()
    return {"classifier": "NaiveBayes", "accuracy": str(accuracy), "clf": clf}


def get_samples(_index, s_s_chs, sr, _size=1.3):
    instances = []
    for _ind in _index:
        instances.append(s_s_chs[_ind:int(math.ceil(_ind + (_size * sr)))][:])
    return np.array(instances)


def get_subdataset(_S=1, Sess=1):
    _file = 'train/Data_S%02d_Sess%02d.csv' % (_S, Sess)
    _f = open(_file).readlines()
    channels = []
    _header = []
    for i, _rows in enumerate(_f):
        if i > 0:
            channels.append(eval(_rows))
        else:
            _header = _rows
            _header = _header.split(',')
    return np.array(channels), np.array(_header[1:-1])


def get_dataset():
    sr = 200
    ch_fs_instances = []
    ch_tags_instances = []
    for subject in range(1, 4):  # 27
        for session in range(1, 2):  # 6
            s_s_chs, _header = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(1, 30):  # len(instances) 60 instances
                instance = np.array(instances[f_instance, :, 1:-1]).transpose()
                ch_fs_instances.append(get_features(instance))
                ch_tags_instances.append('subject_{0}'.format(subject))
    return {"data": ch_fs_instances, "target": ch_tags_instances}  # 2 (data, target), data:9, target: 9


def random_forest(dataTraining, targetTraining):
    '''
    Creating random forest classifier
    '''
    clfArray = []
    meanScore = []
    DEPTHS = [2, 3, 4, 5, 6]
    for depth in DEPTHS:
        clf = RandomForestClassifier(max_depth=depth, random_state=RANDOM_STATE, criterion='gini').fit(dataTraining,
                                                                                                       targetTraining)
        scores = cross_val_score(clf, dataTraining, targetTraining, cv=C_F_V)
        meanScore.append(scores.mean())
        clfArray.append(clf)
    maxScore = max(meanScore)
    position = meanScore.index(maxScore)
    bestDepth = DEPTHS[position]
    bestClf = clfArray[position]
    return {"classifier": "RandomForest depth {0}".format(str(bestDepth)), "accuracy": str(maxScore), "clf": bestClf}


def knn(dataTraining, targetTraining):
    '''
    Creating KNN classifier
    '''
    NEIGHBORS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    clfArray = []
    meanScore = []
    for neighbor in NEIGHBORS:
        clf = KNeighborsClassifier(n_neighbors=neighbor).fit(dataTraining, targetTraining)
        scores = cross_val_score(clf, dataTraining, targetTraining, cv=C_F_V)
        meanScore.append(scores.mean())
        clfArray.append(clf)
    maxScore = max(meanScore)
    position = meanScore.index(maxScore)
    bestNneighbor = NEIGHBORS[position]
    bestClf = clfArray[position]
    # Try for bagging
    clfBag = BaggingClassifier(KNeighborsClassifier(n_neighbors=bestNneighbor).fit(dataTraining,
                                                                                   targetTraining))
    scoresBag = cross_val_score(clfBag, dataTraining, targetTraining, cv=C_F_V)
    scoreBag = scoresBag.mean()
    if scoreBag >= maxScore:
        maxScore = scoreBag
        bestClf = clfBag
    return {"classifier": (str(bestNneighbor) + "-NN (KNN)"), "accuracy": str(maxScore), "clf": bestClf}


def SVM(dataTraining, targetTraining, C_F_V):
    '''
    Creating SVM classifier
    '''
    clfArray = []
    meanScore = []
    kernels = ['linear', 'rbf', 'sigmoid']  # poly sometimes is not working
    for kernel in kernels:
        clf = svm.SVC(kernel=kernel, probability=True).fit(dataTraining, targetTraining)
        scores = cross_val_score(clf, dataTraining, targetTraining, cv=C_F_V)
        meanScore.append(scores.mean())
        clfArray.append(clf)
    maxScore = max(meanScore)
    position = meanScore.index(maxScore)
    bestKernel = kernels[position]
    bestClf = clfArray[position]
    return {"classifier": "{0} SVM".format(bestKernel), "accuracy": str(maxScore), "clf": bestClf}


def selector(dataTraining, targetTraining):
    '''
    Creating the classifier
    '''
    classArray = []
    accuracyArray = []
    clfArray = []
    for i, classifier in enumerate(CLASSIFIERS):
        results = classifier(dataTraining, targetTraining)
        classArray.append(results["classifier"])
        accuracyArray.append(results["accuracy"])
        clfArray.append(results["clf"])
        print("Selector step {0}: {1}, {2}".format(i, results["classifier"], results["accuracy"]))
    print("--------------------------")
    maxAccuracy = max(accuracyArray)
    pos = accuracyArray.index(maxAccuracy)
    bClassifier = classArray[pos]
    bestClf = clfArray[pos]
    return {"model": bestClf, "classifier": bClassifier, "accuracy": maxAccuracy}

"""
dataset = get_dataset()
dataTraining = dataset['data']
targetTraining = dataset['target']

C_F_V = 3
RANDOM_STATE = 0
CLASSIFIERS = [lambda l_dt, l_tt: random_forest(l_dt, l_tt),
               lambda l_dt, l_tt: knn(l_dt, l_tt),
               lambda l_dt, l_tt: SVM(l_dt, l_tt)]

clf = selector(dataTraining, targetTraining)
print("Best classifier {0} with accuracy {1}".format(clf['classifier'], clf['accuracy']))
"""