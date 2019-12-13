
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc,accuracy_score,f1_score,precision_score,recall_score
import itertools
import joblib
import lightgbm as lgbm
from sklearn.preprocessing import LabelBinarizer




def turn_cat(df):
    cat_col = []
    for n,c in df.items():
        if pd.api.types.is_string_dtype(c):
            cat_col.append(c.name)
            df[n] = c.astype('category').cat.as_ordered()
    return df, cat_col


def plot_confusion_matrix(cm, classes,normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

def print_class(m, y_pred):
    res = [format(accuracy_score(y_true=y_test, y_pred=y_pred),'.4f'), 
           format(precision_score(y_true=y_test, y_pred=y_pred, average='weighted' ), '.4f'),
           format(recall_score(y_true=y_test, y_pred=y_pred, average='weighted'), '.4f'), 
           format(f1_score(y_true=y_test, y_pred=y_pred, average='weighted'),'.4f')]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,8))
    plt.grid(b=False)
    plot_confusion_matrix(cm, classes=labels, normalize=True, title='Normalized confusion matrix', cmap = plt.cm.Greens)
    plt.show()
    
    print(f'accuracy score for our model is {res[0]}')
    print(f'precision score for our model is {res[1]}')
    print(f'recall score for our model is {res[2]}')
    print(f'f1 score for our model is {res[3]}')
    report = classification_report(y_test, y_pred)
    print('\n-------------------------------------\n')
    print('Classification report is as follows: ')
    print(report)
    
    
    
def predict_lgbm(lgb_model,x_test,y_test):
    lgb_prediction = lgb_model.predict(x_test)
    lgb_prediction = lgb_prediction.argmax(axis = 1)
    plt.figure(figsize=(8,8))
    plt.grid(b=False)
    cm = metrics.confusion_matrix(y_test_, lgb_prediction)
    plot_confusion_matrix(cm, classes=labels, normalize=True, title='Normalized confusion matrix')
    lgb_F1 = f1_score(lgb_prediction, y_test, average = 'weighted')
    print("The Light GBM F1 is", lgb_F1)
    
def preprocess_cat_data_lgbm(train_data,cols):
    #print(cols)
    for col in train_data.columns:
        if(col in cols):
            std=LabelEncoder() 
            temp=list(train_data[col].values)
            res=std.fit_transform(temp)
            train_data[col]=res
    return train_data

def lgb_f1_score(y_pred,data):
    y_true = data.get_label().astype('int')
    #F1 score not improving in multiclass lgbm custom metric:https://github.com/Microsoft/LightGBM/issues/1483
    y_pred =y_pred.reshape((3,-1)).argmax(axis=0) 
    y_pred=np.round(y_pred)
    return 'f1', f1_score(y_true, y_pred,average='weighted'), True

def predict_lgbm(lgb_model,x_test,y_test):
    lgb_prediction = lgb_model.predict(x_test)
    lgb_prediction = lgb_prediction.argmax(axis = 1)
    lgb_F1 = f1_score(lgb_prediction, y_test, average = 'weighted')
    print("The Light GBM F1 is", lgb_F1)