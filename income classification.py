#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linearmodel import SGDCClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import csv


# In[3]:


income_data=pd.read_csv("OneDrive\Documents\income_evaluation.csv")
income_data.head(10)


# In[4]:


income_data.info()


# In[5]:


income_data.isnull().any() #check no null values


# In[6]:


income_data.shape


# In[7]:


for x in income_data.columns:
    x_new = x.strip()
    income_data=income_data.rename(columns={x:x_new})

income_data.columns


# In[8]:


data = income_data.drop(["fnlwgt","capital-gain","capital-loss","native-country"],axis=1)
data.head(10)


# In[9]:


for column in data[["workclass","education","marital-status","occupation","race","sex"]]:
    data[column] = data[column].str.strip()
data.head(10)


# In[10]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score


# In[11]:


#workclass labels
lb_workclass = preprocessing.LabelEncoder()
lb_workclass.fit(["Private","Self-emp-not-inc","Local-gov","?",
                  "State-gov","Self-emp-inc",
                 "Federal-gov","Without-pay","Never-worked"])
data.iloc[:,1] = lb_workclass.transform(data.iloc[:,1])

#education labels
lb_educ = preprocessing.LabelEncoder()
lb_educ.fit(["HS-grad","Some-college","Bachelors","Masters",
             "Assoc-voc","11th","Assoc-acdm","10th","7th-8th","Prof-school",
             "9th","12th","Doctorate","5th-6th","1st-4th","Preschool"])
data.iloc[:,2] = lb_educ.transform(data.iloc[:,2])

#marriage labels
lb_marry = preprocessing.LabelEncoder()
lb_marry.fit(["Married-civ-spouse","Never-married","Divorced","Separated",
              "Widowed","Married-spouse-absent","Married-AF-spouse"])
data.iloc[:,4] = lb_marry.transform(data.iloc[:,4])

#occupation labels
lb_occ = preprocessing.LabelEncoder()
lb_occ.fit(['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
       'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
       'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
       'Tech-support', '?', 'Protective-serv', 'Armed-Forces',
       'Priv-house-serv'])
data.iloc[:,5] = lb_occ.transform(data.iloc[:,5])
#relationship labels
lb_rel = preprocessing.LabelEncoder()
lb_rel.fit([' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried',
       ' Other-relative'])
data.iloc[:,6] = lb_rel.transform(data.iloc[:,6])

#race labels
lb_race = preprocessing.LabelEncoder()
lb_race.fit(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
       'Other'])
data.iloc[:,7] = lb_race.transform(data.iloc[:,7])

#gender labels
lb_sex = preprocessing.LabelEncoder()
lb_sex.fit(['Male', 'Female'])
data.iloc[:,8] = lb_sex.transform(data.iloc[:,8])


# In[12]:


X=data.iloc[:,:-1]
y=data[["income"]]
X,y


# In[13]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)


# In[14]:


print("Size of train set:", X_train.shape, y_train.shape)
print("Size of test set: ", X_test.shape, y_test.shape)


# In[ ]:





# In[15]:


#decision tree


# In[16]:


from sklearn.tree import DecisionTreeClassifier

incomeGuess = DecisionTreeClassifier(criterion="entropy",max_depth=3)
incomeGuess


# In[17]:


incomeGuess.fit(X_train,y_train)


# In[18]:


predict_income = incomeGuess.predict(X_test)


# In[19]:


from sklearn import metrics
print("Accuracy of decision tree model regarding to income prediction: ", metrics.accuracy_score(y_test,predict_income))


# In[20]:


print('Train accuracy:', incomeGuess.fit(X_train, y_train).score(X_train, y_train))


# In[21]:


print('train vs test:', 1-incomeGuess.fit(X_train, y_train).score(X_train, y_train), 1-metrics.accuracy_score(y_test,predict_income))


# In[22]:


print(classification_report(y_test,predict_income))
print(confusion_matrix(y_test,predict_income))


# In[23]:


accuracy_list=[]
for i in range(1,10):
    incomeGuess = DecisionTreeClassifier(criterion="entropy",max_depth=i)
    incomeGuess.fit(X_train,y_train)
    predict_income = incomeGuess.predict(X_test)
    print("Accuracy with depth {}: ".format(i), metrics.accuracy_score(y_test,predict_income))
    accuracy_list.append(metrics.accuracy_score(y_test,predict_income))
print(accuracy_list)


# In[65]:


plt.plot(range(1,10),accuracy_list,color='midnightblue', marker='o')
ax=plt.axes()
ax.set_facecolor('lightsteelblue')
plt.xlabel('max depth of decision tree')
plt.ylabel('accuracy of classification')
plt.title('max depth vs accuracy')


# In[30]:


#KNN


# In[26]:


sns.pairplot(income_data,hue='income')


# In[32]:


from sklearn.neighbors import KNeighborsClassifier
list2=[]
for i in range(1,40):
    #Train the model
    neigh=KNeighborsClassifier(n_neighbors=i).fit(X_train,np.ravel(y_train))
    y_pred=neigh.predict(X_test)
    acc=metrics.accuracy_score(y_test,y_pred)
    list2.append(acc)
    print(i,acc)


# In[64]:


plt.plot(range(1,40),list2,color='midnightblue', marker='o')
ax=plt.axes()
ax.set_facecolor('lightsteelblue')
plt.xlabel('K-value')
plt.ylabel('accuracy of classification')
plt.title('K-value vs accuracy')


# In[34]:


#k=17 enough
neigh=KNeighborsClassifier(n_neighbors=17).fit(X_train,np.ravel(y_train))
y_pred=neigh.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# In[35]:


print('train vs test:', 1-neigh.fit(X_train, y_train).score(X_train, y_train), 1-metrics.accuracy_score(y_test,y_pred))


# In[36]:


#logistic regression


# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
log_r = LogisticRegression(C=0.01,solver="liblinear").fit(X_train,np.ravel(y_train))
y_hat = log_r.predict(X_test)
y_hat


# In[38]:


import math
con_mat = confusion_matrix(y_test,y_hat)
total_accuracy = (con_mat[0, 0] + con_mat[1, 1]) / float(np.sum(con_mat))
class1_accuracy = (con_mat[0, 0] / float(np.sum(con_mat[0, :])))
class2_accuracy = (con_mat[1, 1] / float(np.sum(con_mat[1, :])))
print(con_mat)
print('Total accuracy of income model: %.2f' % total_accuracy)
print('Accuracy "Income more than 50K": %.2f' % class1_accuracy)
print('Accuracy "Income less than 50K": %.2f' % class2_accuracy)
print('Geometric mean accuracy: %.5f' % math.sqrt((class1_accuracy * class2_accuracy)))


# In[39]:


print(classification_report(y_test,y_hat))
print(confusion_matrix(y_test,y_hat))


# In[40]:


print('train vs test:', 1-log_r.fit(X_train, y_train).score(X_train, y_train), 1-metrics.accuracy_score(y_test,y_hat))


# In[41]:


#random forest classifier


# In[42]:


rfc=RandomForestClassifier(n_estimators=200, max_depth=8) #8 is enough
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)


# In[43]:


print(classification_report(y_test,pred_rfc))
print(confusion_matrix(y_test,pred_rfc))


# In[44]:


print('train vs test:', 1-rfc.fit(X_train, y_train).score(X_train, y_train), 1-metrics.accuracy_score(y_test,pred_rfc))


# In[45]:


list3=[]
for i in range(1,20):
    rfc=RandomForestClassifier(n_estimators=200,max_depth=i)
    rfc.fit(X_train,y_train)
    pred_rfc=rfc.predict(X_test)
    print("Accuracy with depth {}: ".format(i), metrics.accuracy_score(y_test,pred_rfc))
    list3.append(metrics.accuracy_score(y_test,pred_rfc))


# In[63]:


plt.plot(range(1,20),list3,color='midnightblue', marker='o')
ax=plt.axes()
ax.set_facecolor('lightsteelblue')
plt.xlabel('max depth of random forest')
plt.ylabel('accuracy of classification')
plt.title('max depth vs accuracy')


# In[44]:


#SVM classifier


# In[46]:


clf=svm.SVC()
clf.fit(X_train,y_train)
pred_clf=clf.predict(X_test)


# In[47]:


print(classification_report(y_test,pred_clf))
print(confusion_matrix(y_test,pred_clf))


# In[48]:


print('train vs test:', 1-clf.fit(X_train, y_train).score(X_train, y_train), 1-metrics.accuracy_score(y_test,pred_clf))


# In[ ]:




