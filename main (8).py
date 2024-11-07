import numpy as np
import pandas as pd
import seaborn as sns
#Scipy may make it possible to see to graphs at once?
import scipy
import matplotlib.pyplot as plt

df=pd.read_csv("wine_fraud.csv")
print(df.head())
#What are the unique variables in the target column we are trying to predict(quality)?

#.unique take all the unique value of the column
df['quality'].unique()

#sns.countplot(x='quality',data=df)
#sns.countplot(x='type',hue='quality',data=df)

#TASK: What percentage of red wines are fraud? What percentage of white wines are fraud?

reds=df[df['type']=='red']
whites=df[df["type"]=='white']
print("Percentage of fraud in Red Wines")
print(100*(len(reds[reds['quality']=='Fraud'])/len(reds)))

print("Percentage of fraud in White Wines:")
print(100*(len(reds[reds['quality']=='Fraud'])/len(whites)))

df['Fraud']=df['quality'].map({'Legit':0,'Fraud':1})
print(df.corr(numeric_only=True)['Fraud'])

df.corr(numeric_only=True)['Fraud'][:-1].sort_values().plot(kind='bar')

sns.clustermap(df.corr(numeric_only=True),cmap='viridis')

df['type']=pd.get_dummies(df['type'],drop_first=True)
df=df.drop('Fraud',axis=1)
X=df.drop('quality',axis=1)
y=df['quality']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=101)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)

from sklearn.svm import SVC
svc=SVC(class_weight='balanced')

"""
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.001,0.01,0.1,0.5,1],'gamma':['scale','auto']}
grid=GridSearchCV(svc,param_grid)
print(grid.fit(scaled_X_train,y_train))

print(grid.best_params_)


from sklearn.metrics import confusion_matrix,classification_report
grid_pred=grid.predict(scaled_X_test)
print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))
"""

#It is not printing the confusion matirx? why?

plt.show()
