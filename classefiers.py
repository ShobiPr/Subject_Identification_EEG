from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from emperical_mode_decompositoin import get_dataset_EMD
from hilbert_huang_model import get_dataset_HHT
import warnings

warnings.filterwarnings("ignore")


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
    # NEIGHBORS = [1, 2, 3]
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

    #scoresBag = cross_val_score(clfBag, dataTraining, targetTraining, cv=C_F_V)
    #scoreBag = scoresBag.mean()
    """
    if scoreBag >= maxScore:
        maxScore = scoreBag
        bestClf = clfBag
        #clfBag = BaggingClassifier(KNeighborsClassifier(n_neighbors=bestNneighbor).fit(dataTraining,
                                                                                      targetTraining))
    """
    return {"classifier": (str(bestNneighbor) + "-NN (KNN)"), "accuracy": str(maxScore), "clf": bestClf}


def SVM(dataTraining, targetTraining):
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


dataset = get_dataset_EMD()
dataTraining = dataset['data']
targetTraining = dataset['target']

C_F_V = 3
#C_F_V = 3
RANDOM_STATE = 0
CLASSIFIERS = [lambda l_dt, l_tt: random_forest(l_dt, l_tt),
               lambda l_dt, l_tt: decision_tree(l_dt, l_tt),
               lambda l_dt, l_tt: knn(l_dt, l_tt),
               lambda l_dt, l_tt: SVM(l_dt, l_tt),
               lambda l_dt, l_tt: naive_bayes(l_dt, l_tt)]

clf = selector(dataTraining, targetTraining)
print("Best classifier {0} with accuracy {1}".format(clf['classifier'], clf['accuracy']))

