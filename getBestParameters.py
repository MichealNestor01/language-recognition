#
#
#   WARNING ON AN 8 CORE Ryzen 7 5800 X This took 1.5 hours to run
#   Used to find the best parameters for the mlp model
#
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from modules.loadData import *
from modules.scalers import *

def main():
    train_features, train_languages = loadTrainingData()
    test_features, test_languages = loadTestData()

    train_featuresScaled = scaleMinMax(train_features)

    X_train, X_validate, y_train, y_validate = train_test_split(
        train_features, 
        train_languages,
        test_size=0.2,
        random_state=69
    )

    X_train_scaled, X_validate_scaled, _, _ = train_test_split(
        train_featuresScaled,
        train_languages,
        test_size=0.2,
        random_state=69
    )

    # default MLP
    model = MLPClassifier(
        max_iter=2000,
        batch_size=255,
        random_state=69
    )

    # Choose the grid of hyperparameters we want to use for Grid Search to build our candidate models
    parameter_space = {
        # A single hidden layer of size between 8 (output classes) and 180 (input features) neurons is most probable
        # It's a bad idea at guessing the number of hidden layers to have
        # ...but we'll give 2 and 3 hidden layers a shot to reaffirm our suspicions that 1 is best
        'hidden_layer_sizes': [(8,), (180,), (300,),(100,50,),(10,10,10)], 
        'activation': ['tanh','relu', 'logistic'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'epsilon': [1e-08, 0.1 ],
        'learning_rate': ['adaptive', 'constant']
    }
    
    # Create a grid search object which will store the scores and hyperparameters of all candidate models 
    grid = GridSearchCV(
        model, 
        parameter_space,
        cv=3,
        n_jobs=-1 #-1 so it uses all cpu cores
    )
    # Fit the models specified by the parameter grid 
    grid.fit(X_train, y_train)

    # get the best hyperparameters from grid search object with its best_params_ attribute
    print('Best parameters found:\n', grid.best_params_)

if __name__ == "__main__":
    main()