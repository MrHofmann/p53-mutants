import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import time


def main():

    t0 = time.time()

    df = pd.read_csv("../data/PCA/p53_33%_pca_(575 fields, 10166 records).csv", low_memory=False)
    #df = df.replace('$null$', np.nan).dropna()

    features = df.columns[:574].tolist()
    x = df[features]
    y = df['inactive']

    x = pd.DataFrame(MinMaxScaler().fit_transform(x))
    x.columns = features
    #print(x.head())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    parameters = [{'C': [pow(2,x) for x in range(-6, 7, 2)],
                   'kernel': ['linear']
                   },

                  {'C': [pow(2, x) for x in range(-6, 7, 2)],
                   'kernel': ['rbf'],
                   'gamma': np.arange(0.1, 1.1, 0.3),
                   },

                  {'C': [pow(2, x) for x in range(-6, 7, 2)],
                   'kernel': ['poly'],
                   'degree': [2, 4],
                   'gamma': np.arange(0.1, 1.1, 0.3),
                   'coef0': np.arange(0, 1.5, 0.5)
                   }
                  ]

    """
    
                   {'C': [pow(2,x) for x in range(-6,10,2)],
                   'kernel' : ['sigmoid'],
                   'gamma': np.arange(0.1, 1.1, 0.1),
                   'coef0': np.arange(0, 2, 0.5)
                   }
    
    
    
    
    
    SVM
    C : default=1.0
    parametar za regularizaciju
    
    kernel : default=вЂ™rbfвЂ™
             вЂlinearвЂ™  ( <x, x'>),
    
             вЂpolyвЂ™ : ( gamma*<x, x'> + coef0)^degree
                        vezani parametri:
                         degree (stepen): default=3,
                         gamma (koeficijent) : default= 1/n_features
                         coef0 (nezavisni term) default=0.0
    
             вЂrbfвЂ™,  exp(-gamma*|x-x'|^2)
                         vezani parametri:
                         gamma (koeficijent) : default= 1/n_features
                                               gamma>0
    
             вЂsigmoidвЂ™, (tanh(gamma*<x, x'> + coef0)
                         vezani parametri:
                         gamma (koeficijent) : default= 1/n_features
                         coef0 (nezavisni term) default=0.0
    
    
    atributi:
    support_  -indeksi podrzavajucih vektora
    support_vectors_ : podrzavajuci vektori
    n_support_ : broj podrzavajucih vektora za svaku klasu
    dual_coef_ : niz oblika [n_class-1, n_SV]
    koeficijenti podrzavajucih vektora.
    Ukoliko postoji vise klasa, postoje koeficijenti za sve 1-vs-1 klasifikatore.
    coef_ : tezine dodeljene aributima ( samo za linearni kernel)
    intercept_ : konstane u funckiji odlucivanja
        """

    clf = GridSearchCV(SVC(), parameters, cv=5, scoring='f1_macro')
    clf.fit(x_train, y_train)

    print()
    print("Ocena uspeha po klasifikatorima:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) za %s" % (mean, std * 2, params))
    print()

    print("Najbolji parametri:")
    print(clf.best_params_)
    print(clf.best_estimator_.classes_)
    print('Broj podrzavajucih vektora', clf.best_estimator_.n_support_)

    print("Izvestaj za trening skup:")
    y_true, y_pred = y_train, clf.predict(x_train)
    print(classification_report(y_true, y_pred))
    print()

    print("Izvestaj za test skup:")
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()

    t1 = time.time()
    print(t1 - t0)


if __name__=='__main__':
    main()