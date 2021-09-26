import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def sample_model(model1, model2, df1, df2):
    
    """for looking at # training examples"""
    
    train, test = train_test_split(df1, test_size=0.2, random_state=21)
    train2, test2 = train_test_split(df2, test_size=0.2, random_state=21)
    
    X_test1, y_test1 = test.drop('SHOT_RESULT', axis=1), test['SHOT_RESULT']
    X_test2, y_test2 = test2.drop('Class', axis=1), test2['Class']
    
    # training sample sizes
    
    sizes = np.arange(0.05, 1.05, 0.05)
    
    train_accuracy1 = []
    test_accuracy1 = []
    train_accuracy2 = []
    test_accuracy2 = []
    train_recall2 = []
    test_recall2 = []
    train_times1 = []
    train_times2 = []
    
    for i in sizes:
        
        sample1 = df1.sample(frac=i)
        sample2 = df2.sample(frac=i)
        
        X_train1, y_train1 = sample1.drop('SHOT_RESULT', axis=1), sample1['SHOT_RESULT']
        X_train2, y_train2 = sample2.drop('Class', axis=1), sample2['Class']
        
        # model fit
        train_start1 = time.time()
        model1.fit(X_train1, y_train1)
        train_time1 = time.time() - train_start1
        train_times1.append(train_time1)
        
        train_start2 = time.time()
        model2.fit(X_train2, y_train2)
        train_time2 = time.time() - train_start2
        train_times2.append(train_time2)
        
        # predictions
        train_preds1 = model1.predict(X_train1)
        test_preds1 = model1.predict(X_test1)
        
        train_preds2 = model2.predict(X_train2)
        test_preds2 = model2.predict(X_test2)
        
        # scoring
        train_accuracy1.append(accuracy_score(y_train1, train_preds1))
        test_accuracy1.append(accuracy_score(y_test1, test_preds1))
        
        train_accuracy2.append(accuracy_score(y_train2, train_preds2))
        test_accuracy2.append(accuracy_score(y_test2, test_preds2))
        train_recall2.append(recall_score(y_train2, train_preds2))
        test_recall2.append(recall_score(y_test2, test_preds2))
        
    results = pd.DataFrame()
    results['Count'] = sizes
    results['Train Accuracy 1'] = train_accuracy1
    results['Test Accuracy 1'] = test_accuracy1
    results['Train Time 1'] = train_times1
    
    results['Train Accuracy 2'] = train_accuracy2
    results['Test Accuracy 2'] = test_accuracy2
    results['Train Recall 2'] = train_recall2
    results['Test Recall 2'] = test_recall2
    results['Train Time 2'] = train_times2
    
    return results


def gridsearches(model1, model2, X1, y1, X2, y2):
    
    """runs gridsearch on the two datasets"""
    
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=21)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=21)
    
    model1.fit(X_train, y_train)
    model2.fit(X_test, y_test)
    
    print('Best Params 1: ', model1.best_params_)
    print('Best Params 2: ', model2.best_params_)
    
    pd.DataFrame(model1.cv_results_).to_csv(f'{str(model1)[:5]}_grid1.csv')
    pd.DataFrame(model2.cv_results_).to_csv(f'{str(model2)[:5]}_grid1.csv')
    
    return model1, model2


def plot_metrics(results, model_name):
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(16, 12))
    ax0.plot(results['Count'], results['Train Accuracy 1'], label = 'Training Accuracy')
    ax0.plot(results['Count'], results['Test Accuracy 1'], label = 'Testing Accuracy')
    ax0.set_title('Learning Curve: NBA Shots')
    ax0.set_xlabel('Number of Training Examples (% out of 98,001)')
    ax0.set_ylabel('Accuracy Score')
    
    ax0.legend()

    ax1.plot(results['Count'], results['Train Accuracy 2'], label = 'Training Accuracy')
    ax1.plot(results['Count'], results['Test Accuracy 2'], label = 'Testing Accuracy')

    ax1.plot(results['Count'], results['Train Recall 2'], label = 'Training Recall')
    ax1.plot(results['Count'], results['Test Recall 2'], label = 'Testing Recall')

    ax1.set_title('Learning Curve: Fraud')
    ax1.set_xlabel('Number of Training Examples (% out of 227,845)')
    ax1.set_ylabel('Accuracy/Recall Score')
    ax1.legend()
    
    ax2.plot(results['Count'], results['Train Time 1'], label = 'Training Time', color = 'lime')
    ax2.set_title('Model Fit Time: NBA Shots')
    ax2.set_xlabel('Number of Training Examples (% out of 98,001)')
    ax2.set_ylabel('Time (Seconds)')
    ax2.legend()

    ax3.plot(results['Count'], results['Train Time 2'], label = 'Training Time', color = 'aqua')
    ax3.set_title('Model Fit Time: Fraud')
    ax3.set_xlabel('Number of Training Examples (% out of 227,845)')
    ax3.set_ylabel('Time (Seconds)')
    ax3.legend()
    
    
    fig.suptitle(f'Learning Curves and Fit Times: {model_name}', fontsize='x-large', y=0.93)

    plt.show()

def sample_nnmodel(model1, model2, df1, df2):
    
    """for looking at # training examples"""
    
    train, test = train_test_split(df1, test_size=0.2, random_state=21)
    train2, test2 = train_test_split(df2, test_size=0.2, random_state=21)
    
    X_test1, y_test1 = test.drop('SHOT_RESULT', axis=1), test['SHOT_RESULT']
    X_test2, y_test2 = test2.drop('Class', axis=1), test2['Class']
    
    # training sample sizes
    
    sizes = np.arange(0.05, 1.05, 0.05)
    
    train_accuracy1 = []
    test_accuracy1 = []
    train_accuracy2 = []
    test_accuracy2 = []
    train_recall2 = []
    test_recall2 = []
    train_times1 = []
    train_times2 = []
    
    for i in sizes:
        
        sample1 = df1.sample(frac=i)
        sample2 = df2.sample(frac=i)
        
        X_train1, y_train1 = sample1.drop('SHOT_RESULT', axis=1), sample1['SHOT_RESULT']
        X_train2, y_train2 = sample2.drop('Class', axis=1), sample2['Class']
        
        # model fit
        train_start1 = time.time()
        model1.fit(X_train1, y_train1)
        train_time1 = time.time() - train_start1
        train_times1.append(train_time1)
        
        train_start2 = time.time()
        model2.fit(X_train2, y_train2)
        train_time2 = time.time() - train_start2
        train_times2.append(train_time2)
        
        # predictions
        train_preds1 = np.round(model1.predict(X_train1)).flatten()
        test_preds1 = np.round(model1.predict(X_test1)).flatten()
        
        train_preds2 = np.round(model2.predict(X_train2)).flatten()
        test_preds2 = np.round(model2.predict(X_test2)).flatten()
        
        # scoring
        train_accuracy1.append(accuracy_score(y_train1, train_preds1))
        test_accuracy1.append(accuracy_score(y_test1, test_preds1))
        
        train_accuracy2.append(accuracy_score(y_train2, train_preds2))
        test_accuracy2.append(accuracy_score(y_test2, test_preds2))
        train_recall2.append(recall_score(y_train2, train_preds2))
        test_recall2.append(recall_score(y_test2, test_preds2))
        
    results = pd.DataFrame()
    results['Count'] = sizes
    results['Train Accuracy 1'] = train_accuracy1
    results['Test Accuracy 1'] = test_accuracy1
    results['Train Time 1'] = train_times1
    
    results['Train Accuracy 2'] = train_accuracy2
    results['Test Accuracy 2'] = test_accuracy2
    results['Train Recall 2'] = train_recall2
    results['Test Recall 2'] = test_recall2
    results['Train Time 2'] = train_times2
    
    return results

