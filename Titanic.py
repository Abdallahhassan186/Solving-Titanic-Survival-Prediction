# Solving-Titanic-Survival-Prediction
import os
import pandas as pp
df1 = pp.read_csv('train.csv',delimiter=',', header=0,usecols=(1,2,4,5))
df2 = pp.read_csv('test.csv',delimiter=',', header=0, usecols=(0,1,3,4))

import numpy as np
df1 = pp.DataFrame(df1)
df2 = pp.DataFrame(df2)
np1 = np.array(df1)
np2 = np.array(df2)

test_id_ONLY = np2[:,0]
test = np2[:,(1,2,3)]
passid = np1[:,(0)]
stats = np1[:,(1,2,3)]

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(stats,passid)
clf = clf.predict(test)

from matplotlib import pyplot as plt
plt.plot(test_id_ONLY, clf)
plt.show()


ans = np.array([test_id_ONLY,clf])
ans = np.transpose(ans)
ans = pp.DataFrame(ans)
ans.to_csv('answers.csv')
