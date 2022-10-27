# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 11:49:21 2022

@author: elif
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score
veri = pd.read_csv("C:/Users/AttilA/Desktop/MachineLearning/Odev_2/SydneyHousePrices.csv")

veri2=veri.drop(['Date','postalCode','Id','suburb','propType'],axis=1)

veri2=veri2.dropna()
veri2=veri2.head(1000)

x=veri2.iloc[:,1:4]
y=veri2.iloc[:,0:1]
print(y)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

sc =StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

#logistic
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
y_pred=logr.predict(X_test)

print(y_pred)
print(y_test)

cm=confusion_matrix(y_test, y_pred)
print(cm)

print("Sınıflandırma raporu \n",metrics.classification_report(y_test, y_pred))
print("doğruluk değeri \n", metrics.accuracy_score(y_test, y_pred))

basarı = cross_val_score(estimator=logr, X=X_train,y=y_train,cv=4)

print(basarı.mean())
print(basarı.std())


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train,y_train)
y_pred_knn=knn.predict(X_test)

print(y_pred_knn)
print(y_test)

cm_knn=confusion_matrix(y_test, y_pred_knn)
print(cm_knn)
print("Sınıflandırma raporu \n",metrics.classification_report(y_test, y_pred_knn))
print("doğruluk değeri \n", metrics.accuracy_score(y_test, y_pred_knn))

bas_knn=cross_val_score(estimator=knn, X=X_train,y=y_train,cv=4)
print(bas_knn.mean())
print(bas_knn.std())

from sklearn.svm import SVC


for i in (1,2,3,4,5,6):
    svc2 = SVC(kernel='poly',degree=i)
    svc2.fit(X_train,y_train)
    y_pred_poly=svc2.predict(X_test)
    print(y_test)
    print("{} . dereceden poly  \n {} \n prediction sonucu".format(i,y_pred_poly))
    cm_svc2=confusion_matrix(y_test, y_pred_poly)
    print("{} . dereceden poly  \n {} \n confusion matrisi".format(i,cm_svc2))
    print("{} dereceden \n {} Sınıflandırma raporu \n".format(i,metrics.classification_report(y_test, y_pred_poly)))
    print("{} dereceden \n {} doğruluk değeri \n".format(i,metrics.accuracy_score(y_test, y_pred_poly)))
    bas_svc2=cross_val_score(estimator=svc2, X=X_train,y=y_train,cv=4)
    print(bas_svc2.mean())
    print(bas_svc2.std())


svc3= SVC(kernel='linear')
svc3.fit(X_train,y_train)
y_pred_lr=svc3.predict(X_test)
print(y_test)
print(y_pred_lr)

cm_svc=confusion_matrix(y_test, y_pred_lr)
print(cm_svc)

print("Sınıflandırma raporu \n",metrics.classification_report(y_test, y_pred_lr))
print("doğruluk değeri \n", metrics.accuracy_score(y_test, y_pred_lr))
bas_svc3=cross_val_score(estimator=svc3, X=X_train,y=y_train,cv=4)
print(bas_svc3.mean())
print(bas_svc3.std())


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred_nb=gnb.predict(X_test)


print(y_test)
print(y_pred_nb)

cm_nbg=confusion_matrix(y_test, y_pred_nb)
print(cm_nbg)


print("Sınıflandırma raporu \n",metrics.classification_report(y_test, y_pred_nb))
print("doğruluk değeri \n", metrics.accuracy_score(y_test, y_pred_nb))

bas_gnb=cross_val_score(estimator=gnb, X=X_train,y=y_train,cv=4)
print(bas_gnb.mean())
print(bas_gnb.std())


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc2=DecisionTreeClassifier(criterion='gini')

dtc.fit(X_train,y_train)
y_pred_dtc_en=dtc.predict(X_test)

cm_dtc_en=confusion_matrix(y_test, y_pred_dtc_en)

print(y_test)
print(y_pred_dtc_en)
print(cm_dtc_en)


print("Sınıflandırma raporu \n",metrics.classification_report(y_test, y_pred_dtc_en))
print("doğruluk değeri \n", metrics.accuracy_score(y_test, y_pred_dtc_en))

bas_dtc=cross_val_score(estimator=dtc, X=X_train,y=y_train,cv=4)
print(bas_dtc.mean())
print(bas_dtc.std())


dtc2.fit(X_train,y_train)
y_pred_dtc_gini=dtc2.predict(X_test)

cm_dtc_gini=confusion_matrix(y_test, y_pred_dtc_gini)

print(y_test)
print(y_pred_dtc_gini)
print(cm_dtc_gini)

print("Sınıflandırma raporu \n",metrics.classification_report(y_test, y_pred_dtc_gini))
print("doğruluk değeri \n", metrics.accuracy_score(y_test, y_pred_dtc_gini))

bas_dtc2=cross_val_score(estimator=dtc2, X=X_train,y=y_train,cv=4)
print(bas_dtc2.mean())
print(bas_dtc2.std())

from sklearn.ensemble import RandomForestClassifier
for i in range(1,11):
    rf = RandomForestClassifier(n_estimators=i, criterion='entropy')
    rf.fit(X_train,y_train)
    y_pred_rf = rf.predict(X_test)
    print(y_test)
    print(y_pred_rf)
    
    print("{} karar ağacı sayısı \n {} Sınıflandırma raporu \n".format(i,metrics.classification_report(y_test, y_pred_rf)))
    print("{} karar ağacı sayısı \n {} doğruluk değeri \n".format(i,metrics.accuracy_score(y_test, y_pred_rf)))
    bas_rbf=cross_val_score(estimator=rf, X=X_train,y=y_train,cv=4)
    print(bas_rbf.mean())
    print(bas_rbf.std())

# rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
# rfc.fit(X_train,y_train)
# y_pred_rfc=rfc.predict(X_test)
# cm_rf=confusion_matrix(y_test,y_pred_rf)
# print(cm_rf)

fpr ,tpr,thold =metrics.roc_curve(y_test,y_pred_rf,pos_label=0 )

print(fpr)
print(tpr)

svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)
y_pred_svc=svc.predict(X_test)
print(y_test)
print(y_pred_svc)

cm_svc=confusion_matrix(y_test, y_pred_svc)
print(cm_svc)

print("Sınıflandırma raporu \n",metrics.classification_report(y_test, y_pred_svc))
print("doğruluk değeri \n", metrics.accuracy_score(y_test, y_pred_svc))

bas_svc=cross_val_score(estimator=svc, X=X_train,y=y_train,cv=4)
print(bas_svc.mean())
print(bas_svc.std())

from sklearn.model_selection import GridSearchCV

p= [{'C':[1,2,3,4,5],'kernel':['linear']},
    {'C':[1,2,3,4,5],'kernel':['rbf'],'gamma':[1,0.5,0.1,0.01,0.001]},
    {'C':[1,2,3,4,5],'kernel':['poly'],'degree':[1,2,3,4,5,6,7],'gamma':[1,0.5,0.1,0.01,0.001]}]

gs= GridSearchCV(estimator=svc, param_grid=p,scoring='accuracy',cv=4)
grid_search=gs.fit(X_train,y_train)

eniyiparamat=grid_search.best_params_
eniyiscore=grid_search.best_score_

print(eniyiparamat)
print(eniyiscore)






















