import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from joblib import Parallel, delayed, parallel
import multiprocessing

from modules.loadData import *
from modules.displayUtils import printFeatures
from modules.scalers import *


def main():
    train_features, train_languages = loadTrainingData("train")
    test_features, test_languages = loadTestData("test")
    
    # we played around with scaling but it only made things worse
    train_featuresScaled = scaleStandard(train_features)

    classification_models = [
        KNeighborsClassifier(),#(3),
        DecisionTreeClassifier(),#max_depth=5),
        RandomForestClassifier(),#max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]

    def processInput(model):
        with warnings.catch_warnings():
            # ignore all caught warnings
            warnings.filterwarnings("ignore")
            print(f"Testing model: {model}")
            model.fit(train_features, train_languages)
            score = model.score(test_features, test_languages)
            model_name = type(model).__name__
            print(f"tested model: {model_name}")
            #print(model_name,(f'{100*score:.2f}%'))        
            return (model_name,(f'{100*score:.2f}%'))
    
    scores = []
    endscores = []
    for i in range(3):
        print(f"\n\n\trun: {i}\n\n")
        num_cores = multiprocessing.cpu_count()
        scores = Parallel(n_jobs=num_cores)(delayed(processInput)(model) for model in classification_models)
        print(scores)
        endscores.append(scores)
    print(endscores)

if __name__ == "__main__":
    main()