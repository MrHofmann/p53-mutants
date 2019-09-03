import numpy as np
import pandas as pd
from sklearn.tree import  DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as met

import subprocess
import time


def visualize_tree(tree, feature_names, class_names):

    with open("dt.dot", "w") as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names,
                        class_names=class_names,
                        filled=True,
                        rounded=True)

    f.close()

    subprocess.call("dot -Tpng dt.dot -o tree.png", shell=True)


def main():
    t0 = time.time()

    df = pd.read_csv("../data/PCA/p53_33%_pca_(575 fields, 10166 records).csv", low_memory=False)
    df = df.replace('$null$', np.nan).dropna()

    features = df.columns[:574].tolist()
    x = df[features]
    y = df['inactive']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y)

    params = [{'criterion': ['gini'],
               'max_depth': [None, 5, 10, 20],
               'min_samples_split': [2, 4, 8],
               'max_leaf_nodes': [None]
               }]
    clf = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
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

    print("Matrica konfuzije:")
    y_true, y_pred = y_test, clf.predict(x_test)
    print(met.confusion_matrix(y_true, y_pred))
    print()

    print("Izvestaj za trening skup:")
    y_true, y_pred = y_train, clf.predict(x_train)
    print(met.classification_report(y_true, y_pred))
    print()

    print("Izvestaj za test skup:")
    y_true, y_pred = y_test, clf.predict(x_test)
    print(met.classification_report(y_true, y_pred))
    print()

    visualize_tree(clf.best_estimator_, features, df['inactive'].unique())

    t1 = time.time()
    print(t1 - t0)


if __name__=='__main__':
    main()



#print('Klase', dt.classes_)
#print('feature_importances_', '\n', pd.Series(dt.feature_importances_, index=features))
#print('tree', dt.tree_)
#print('Predvidjena verovatnoca', dt.predict_proba(x_test), sep='\n')
#print('\n\n')
#x_proba =  pd.DataFrame(dt.predict_proba(x_test), index= x_test.index, columns=['prob_' + x for x in dt.classes_])
#x_test_proba = pd.concat([x_test, x_proba], axis=1)
#print(x_test_proba.head(10))


#primena modela na trening podacima
#y_pred = dt.predict(x_train)
#calculate_metrics('Trening ',y_train, y_pred )

##primena modela na test podacima
#y_pred = dt.predict(x_test)
#cnf_matrix = met.confusion_matrix(y_test, y_pred)
#print(cnf_matrix)
#calculate_metrics('Test',y_test, y_pred )