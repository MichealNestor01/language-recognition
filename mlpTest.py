import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
import seaborn as sn

from modules.loadData import *
from modules.displayUtils import printFeatures
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
        max_iter=1000,
        batch_size=255,
        random_state=69,
        activation='logistic',
        alpha=0.0001,
        epsilon=1e-08,
        hidden_layer_sizes=(300,),
        learning_rate='adaptive',
        solver='adam'
    )

    model.fit(X_train, y_train)
    
    print(f"Possible languages predicted by model = {model.classes_}")
    print(f"Unscaled Augmented MLP model\'s accuracy on training set is {100*model.score(X_train, y_train)}")
    print(f"Unscaled Augmented MLP model\'s accuracy on validation set is {100*model.score(X_validate, y_validate)}")
    print(f"Unscaled Augmented MLP model\'s accuracy on test set is {100*model.score(test_features, test_languages)}")

    # ====================================================== #        
    #                                                        #
    #           NOW DRAW CONFUSION MATRIX                    #
    #                                                        #
    # ====================================================== #

    #get predicitions on test set
    test_language_predictions = model.predict(test_features)
    test_language_groundtruth = test_languages
    #print(test_language_predictions)

    #build confusion matrix
    conf_matrix = confusion_matrix(test_language_groundtruth, test_language_predictions)
    conf_matrix_norm = confusion_matrix(test_language_groundtruth, test_language_predictions, normalize='true')

    #set labels for matric axes from languages
    language_list = ['english', 'german', 'spanish']
    language_name = [lang for lang in language_list]

    #make a confusion matrix with labels using a dataframe
    confmatrix_df = pd.DataFrame(conf_matrix, index=language_name, columns=language_name)
    confmatrix_df_norm = pd.DataFrame(conf_matrix_norm, index=language_name, columns=language_name)

    #plot confusion matrices
    plt.figure(figsize=(16,6))
    sn.set(font_scale=1.8)
    plt.subplot(1, 2, 1)
    plt.title('Confusion Matrix')
    sn.heatmap(confmatrix_df, annot=True, annot_kws={"size": 18})
    plt.subplot(1, 2, 2)
    plt.title('Normalised Confusion Matrix')
    sn.heatmap(confmatrix_df_norm, annot=True, annot_kws={"size": 13})

    plt.show()
    
if __name__ == "__main__":
    main()






















