import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import sklearn.metrics as met
from termcolor import colored


def main():

    df = pd.read_csv("../data/PCA/p53_33%_pca_(575 fields, 10166 records).csv", low_memory=False)
    df = df.replace('$null$', np.nan).dropna()

    features = df.columns[:574].tolist()
    x = df[features]
    y = df['inactive']

    scaler = preprocessing.StandardScaler().fit(x)
    x = pd.DataFrame(scaler.transform(x))
    x.columns = features
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y)

    """
    hidden_layer_sizes activation solver batch_size learning_rate learning_rate_init power_t max_iter tol shuffle 
    verbose early_stopping validation_fraction
    """

    params = [{'solver': ['sgd'],
               'learning_rate': ['constant', 'invscaling', 'adaptive'],
               'learning_rate_init': [0.01, 0.005, 0.002, 0.001],
               'activation': ['identity', 'logistic', 'tanh', 'relu'],
               'hidden_layer_sizes': [(10, 3), (10, 10)],
               'max_iter': [500]
               }]

    clf = GridSearchCV(MLPClassifier(), params, cv=5)
    clf.fit(x_train, y_train)

    print("Najbolji parametri:")
    print(clf.best_params_)
    print()
    print("Ocena uspeha po klasifikatorima:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) za %s" % (mean, std * 2, params))
    print()

    print("Izvestaj za test skup:")
    y_true, y_pred = y_test, clf.predict(x_test)
    cnf_matrix = met.confusion_matrix(y_test, y_pred)
    print("Matrica konfuzije", cnf_matrix, sep="\n")
    print("\n")

    accuracy = met.accuracy_score(y_test, y_pred)
    print("Preciznost", accuracy)
    print("\n")

    class_report = met.classification_report(y_test, y_pred, target_names=clf.classes_)
    print("Izvestaj klasifikacije", class_report, sep="\n")

    print('Broj iteracija: ', clf.best_estimator_.n_iter_)
    print('Broj slojeva: ', clf.best_estimator_.n_layers_)
    print('Koeficijenti:', clf.best_estimator_.coefs_, sep='\n')
    print('Bias:', clf.best_estimator_.intercepts_, sep='\n')


if __name__=='__main__':
    main()