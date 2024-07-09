#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries


# In[2]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("C:/Users/MADHUSUDAN/Downloads/final_edX.csv")
df.head()


# In[4]:


df.shape


# In[5]:


#adding column


# In[6]:


df['windex'] = np.where(df.WAB > 7, 'True','False')


# In[7]:


#data visualization


# In[8]:


df1 = df.loc[df['POSTSEASON'].str.contains('F4|S16|E8', na = False)]
df1.head()


# In[9]:


df1['POSTSEASON'].value_counts()


# In[10]:


#installing seaborn


# In[11]:


get_ipython().system('conda install -c anaconda seaborn -y')


# In[12]:


conda update scikit-learn


# In[13]:


import seaborn as sns
bins = np.linspace(df1.BARTHAG.min(), df1.BARTHAG.max(), 10)
g = sns.FacetGrid(df1, col = "windex", hue = "POSTSEASON", palette = "Set1", col_wrap = 6)
g.map(plt.hist, 'BARTHAG', bins = bins, ec = "k" )
g.axes[-1].legend()
plt.show()


# In[14]:


bins = np.linspace(df1.ADJOE.min(), df1.ADJOE.max(), 10)
g = sns.FacetGrid(df1, col = "windex", hue = "POSTSEASON", palette = "Set1", col_wrap = 2)
g.map(plt.hist, 'ADJOE', bins = bins, ec = "k")
g.axes[-1].legend()
plt.show()


# In[15]:


#data preprocessing


# In[16]:


bins = np.linspace(df1.ADJDE.min(), df1.ADJDE.max(), 10)
g = sns.FacetGrid(df1, col = "windex", hue = "POSTSEASON", palette = "Set1", col_wrap = 2)
g.map(plt.hist, 'ADJDE', bins = bins, ec = "k")
g.axes[-1].legend()
plt.show()


# In[17]:


#categorical to numerical values


# In[18]:


df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize = True)


# In[19]:


df1['windex'].replace(to_replace = ['False', 'True'],value = [0,1],inplace = True)
df1.head()


# In[20]:


#feature selection


# In[21]:


df.columns


# In[22]:


df1.columns


# In[23]:


X = df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
X[0:5]


# In[24]:


y = df1['POSTSEASON'].values
y[0:5]


# In[25]:


#normalizing data


# In[26]:


X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[27]:


#training & validation


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 4)
print('Train Set :', X_train.shape, y_train.shape)
print('Validation Set :', X_val.shape, y_val.shape)


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


#classification


# In[32]:


#k-nearest neighbor(KNN)
#Question-1 : Build a KNN model using a value of k equals 5 & find the accuracy on the validation data


# In[33]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# In[34]:


knn_model = KNeighborsClassifier(n_neighbors = 5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_val)


# In[35]:


knn_model_train_acc = accuracy_score(y_train, knn_model.predict(X_train))
knn_model_val_acc = accuracy_score(y_val, y_pred)
print("Training accuracy :",knn_model_train_acc)
print("Validation accuracy :",knn_model_val_acc)


# In[36]:


#Question-2 : Determine & print the accuracy for the first 15 values of k on the validation set


# In[37]:


k = 15
train_acc = np.zeros((k-1))
val_acc = np.zeros((k-1))


# In[38]:


for n in range(1, k):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_val)
    train_acc[n-1] = accuracy_score(y_val, pred)
    val_acc[n-1] = np.std(pred == y_val)/np.sqrt(pred.shape[0])


# In[39]:


print("Training accuracy :",train_acc)
print("Validation accuracy :",val_acc)


# In[40]:


#decision tree
#Question-3 : Determine the minimun value for the parameter max_depth that improves results


# In[41]:


from sklearn.tree import DecisionTreeClassifier


# In[42]:


max_depth = 4
accuracy = []
for n in range(1, max_depth):
    DecisionTree = DecisionTreeClassifier(criterion = "entropy", max_depth = n)
    DecisionTree.fit(X_train, y_train)
    score = DecisionTree.score(X_val, y_val)
    accuracy.append(score)
print(accuracy)
print("Minimum value for max-depth :",accuracy.index(min(accuracy)))


# In[43]:


#support vector machine
#Question-4 : train the svm model & determine the accuracy on the validation data for each kernel.
#Find the kernel that provides the best score on the validation data & train a svm using it


# In[44]:


#linear
from sklearn import svm
model = svm.SVC(kernel = 'linear')
model.fit(X_train, y_train)
yhat = model.predict(X_val)


# In[45]:


from sklearn.metrics import f1_score
linear = f1_score(y_val, yhat, average = 'weighted')
print(linear)
from sklearn.metrics import jaccard_score
print(jaccard_score(y_val, yhat, average = 'weighted'))


# In[46]:


#poly
from sklearn import svm
model = svm.SVC(kernel = 'poly')
model.fit(X_train, y_train)
yhat = model.predict(X_val)


# In[47]:


from sklearn.metrics import f1_score
poly = f1_score(y_val, yhat, average = 'weighted')
print(poly)
from sklearn.metrics import jaccard_score
print(jaccard_score(y_val, yhat, average = 'weighted'))


# In[48]:


#rbf
from sklearn import svm
model = svm.SVC(kernel = 'rbf')
model.fit(X_train, y_train)
yhat = model.predict(X_val)


# In[49]:


from sklearn.metrics import f1_score
rbf = f1_score(y_val, yhat, average = 'weighted')
print(rbf)
from sklearn.metrics import jaccard_score
print(jaccard_score(y_val, yhat, average = 'weighted'))


# In[50]:


#sigmoid
from sklearn import svm
model = svm.SVC(kernel = 'sigmoid')
model.fit(X_train, y_train)
yhat = model.predict(X_val)


# In[51]:


from sklearn.metrics import f1_score
sigmoid = f1_score(y_val, yhat, average = 'weighted')
print(sigmoid)
from sklearn.metrics import jaccard_score
print(jaccard_score(y_val, yhat, average = 'weighted'))


# In[52]:


best = []
best.append(linear)
best.append(poly)
best.append(rbf)
best.append(sigmoid)
print(best)


# In[53]:


print("The best score :", max(best))


# In[54]:


#logistic regression
#Question-5 : train a logistic regression model & determine the accuracy of the validation data


# In[55]:


from sklearn.linear_model import LogisticRegression
linear_model = LogisticRegression(C = 0.01, solver = 'liblinear')
linear_model.fit(X_train, y_train)
linear_model


# In[56]:


from sklearn.metrics import jaccard_score
yh = linear_model.predict(X_val)
print(jaccard_score(y_val, yh,average = 'macro'))


# In[57]:


#model_evaluation_using_test_set


# In[58]:


from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import jaccard_score


# In[59]:


def jaccard_index(predictions, true):
    if(len(predictions) == len(true)):
        intersect = 0;
        for x,y in zip(predictions, true):
            if(x == y):
                intersect += 1
        return intersect / len(predictions) + len(true) - intersect
    else:
        return -1


# In[60]:


#Question-6 : calculate f1_score & jaccard_score for each model from above.
#Use the hyper paramter that performed best on the validation data.


# In[61]:


test_df = pd.read_csv("C:/Users/MADHUSUDAN/Downloads/test_edX.csv")
test_df.head()


# In[62]:


test_df['windex'] = np.where(test_df.WAB > 7, 'True', 'False')
test_df1 = test_df[test_df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
test_Feature = test_df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
test_Feature['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
test_X=test_Feature
test_X= preprocessing.StandardScaler().fit(test_X).transform(test_X)
test_X[0:5]


# In[63]:


test_y = test_df1['POSTSEASON'].values
test_y[0:5]


# In[64]:


test_knn = knn.predict(test_X)
knn_prob = knn.predict_proba(test_X)
print("F1-score :", f1_score(test_y, test_knn, average = 'micro'))
print("Jaccard score : ",jaccard_score(test_y, test_knn, average = 'micro'))
print("Log loss : %.2f" %log_loss(test_y, knn_prob))


# In[65]:


test_dt = DecisionTree.predict(test_X)
dt_prob = DecisionTree.predict_proba(test_X)
print("F1-score :", f1_score(test_y, test_dt, average = 'micro'))
print("Jaccard score : ",jaccard_score(test_y, test_dt, average = 'micro'))
print("Log loss : %.2f" %log_loss(test_y, dt_prob))


# In[66]:


test_svm = model.predict(test_X)
print("F1-score :", f1_score(test_y, test_svm, average = 'micro'))
print("Jaccard score : ",jaccard_score(test_y, test_svm, average = 'micro'))


# In[67]:


test_lr = linear_model.predict(test_X)
lr_prob = linear_model.predict_proba(test_X)
print("F1-score :", f1_score(test_y, test_lr, average = 'micro'))
print("Jaccard score : ",jaccard_score(test_y, test_lr, average = 'micro'))
print("Log loss : %.2f" %log_loss(test_y, lr_prob))


# In[68]:


#report


# 
# |     Algorithm        |  F1-score  |  Jaccard  |  LogLoss  |
# |----------------------|------------|-----------|-----------|
# |       KNN            |    0.57    |    0.40   |    0.87   |
# |    Decision Tree     |    0.70    |    0.53   |    5.05   |
# |       SVM            |    0.60    |    0.42   |     NA    |
# | Logistics Regression |    0.68    |    0.52   |    1.04   |
# 

# In[ ]:




