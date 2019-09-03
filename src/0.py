import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import numpy as np


df = pd.read_csv("../data/CSV/p53_33%_(5,409 fields, 10,231 records).csv", low_memory=False)
df = df.replace('$null$', np.nan).dropna()

num_active = len(df[df['inactive'] == 'active'])
num_inactive = len(df[df['inactive'] == 'inactive'])

print(num_active)
print(num_inactive)

features = df.columns[:5408].tolist()
x = df[features]
y = df['inactive']
num_features = x.shape[1]

#standardizacija podataka
scaler = preprocessing.StandardScaler().fit(x)
x = pd.DataFrame(scaler.transform(x))
x.columns = features

#primena pca
pca = PCA()
pca.fit(x)
x_pca = pd.DataFrame(pca.transform(x))
pca_columns = ['pca%d'%i for i in range(1, min(x.shape[0], pca.n_components_)+1)]
x_pca.columns = pca_columns

xy_df = x_pca
xy_df['inactive'] = y.tolist()

relevant_eigenvalues = [e for e in pca.explained_variance_ if e >= 1]
relevant_eigenvalues_sum = sum(pca.explained_variance_ratio_[:len(relevant_eigenvalues)])
print(relevant_eigenvalues_sum)
print(len(relevant_eigenvalues))

df_pca = pd.DataFrame(xy_df.iloc[:, :len(relevant_eigenvalues)], index=xy_df.index.values, columns=xy_df.columns.values[:len(relevant_eigenvalues)])
df_pca['inactive'] = y.tolist()
with open('../data/PCA/p53_33%_pca_({} fields, {} records).csv'.format(len(relevant_eigenvalues)+1, xy_df.shape[0]), 'w') as f:
    df_pca.to_csv(f, index=False)
f.close()

#cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
#pca_int_columns = [int(i[3:]) for i in pca_columns]
#plt.figure()
#plt.bar(pca_int_columns,  pca.explained_variance_ratio_,  label='Procenat varijanse')
#plt.plot(pca_int_columns,  cumulative_variance, color='darkorange', label='Kumulativna varijansa')
#plt.plot(pca_int_columns, [relevant_eigenvalues_sum for i in pca_int_columns], color='red', label='Objasnjena varijansa')

#plt.xlabel('Glavne komponente')
#plt.ylabel('Procent objasnjene varijanse')
#plt.legend()
#plt.show()


#print('components_ ')
#for i, component in zip(range(1, pca.n_components_+1), pca.components_):
#    pca_desc="pca%d"%i + "="
#    for j, value in zip(range(0, num_features), component):
#        pca_desc+="%.2f*%s"%(value, features[j])
#    print(pca_desc)

