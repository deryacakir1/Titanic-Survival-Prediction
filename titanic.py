# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:39:40 2025

@author: Derya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import LabelEncoder

# Titanic veri setini yükleme
train_data = pd.read_csv('C:/Users/Derya/Desktop/Yüksek Lisans/Makine Öğrenmesi/Final/titanic/train.csv')
test_data = pd.read_csv('C:/Users/Derya/Desktop/Yüksek Lisans/Makine Öğrenmesi/Final/titanic/test.csv')

# İlk 5 satırı görüntüleme
print(train_data.head())

# Veri seti hakkında genel bilgi
print(train_data.info())
print(train_data.describe())

# Eksik verileri tespit etme
print(train_data.isnull().sum())

# Age sütunundaki eksik değerleri doldurma (ortalama ile)
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

# Embarked sütunundaki eksik değerleri doldurma (mod ile)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Cabin sütununu veri setinden kaldırma
train_data.drop('Cabin', axis=1, inplace=True)

# Eksik değerlerin kontrolü
print(train_data.isnull().sum())

# Title sütunu oluşturma
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# FamilySize sütunu oluşturma
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

# Cinsiyet sütununu sayısal değerlere dönüştürme
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# Embarked sütununu one-hot encoding ile dönüştürme
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)

# Title sütununu sayısal verilere dönüştürme (Label Encoding)
label_encoder = LabelEncoder()
train_data['Title'] = label_encoder.fit_transform(train_data['Title'])

# Hedef değişken ve özelliklerin belirlenmesi
X = train_data.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)  # Hedef dışındaki sütunlar
y = train_data['Survived']  # Hedef değişken

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim ve test verilerinin boyutlarını kontrol etme
print("Eğitim veri seti boyutu (X_train):", X_train.shape)
print("Test veri seti boyutu (X_test):", X_test.shape)
print("Eğitim hedef boyutu (y_train):", y_train.shape)
print("Test hedef boyutu (y_test):", y_test.shape)

# Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Model 2: Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Model 3: Support Vector Machine
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Performans değerlendirme
def evaluate_model(y_test, y_pred, model_name):
    print(f"{model_name} Modeli:")
    print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
    print("F1 Skoru:", f1_score(y_test, y_pred))
    print("ROC AUC Skoru:", roc_auc_score(y_test, y_pred))
    print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
    print("-" * 50)

# Modelleri değerlendirme
evaluate_model(y_test, y_pred_log_reg, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_svm, "Support Vector Machine")

# ROC Eğrisi görselleştirme
def plot_roc_curve(y_test, y_pred, model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_pred):.2f})')
    
# ROC eğrisini çizme
plt.figure(figsize=(10, 6))
plot_roc_curve(y_test, y_pred_log_reg, "Logistic Regression")
plot_roc_curve(y_test, y_pred_rf, "Random Forest")
plot_roc_curve(y_test, y_pred_svm, "Support Vector Machine")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Eğrisi')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
