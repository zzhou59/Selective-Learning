### COPYRIGHT Ziyun Zhou ###

import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

def SL(df,feature,target,param1=None,param2=None,param3=None):
    '''

    :param df: Dataframe
    :param feature: indicators
    :param target: target
    :param param1: parameters of Gradient Boost
    :param param2: parameters of Random Forest
    :param param3: parameters of SVM
    :return: print the comparison of original and new prediction MSEs
    '''
    test_percent = 0.3
    train, test = train_test_split(df, test_size=test_percent, random_state=15)
    x_train, x_test = train[feature], test[feature]
    y_train, y_test = train[target], test[target]
    # split training set into 70% train_train and 30% train_test
    train_train, train_test = train_test_split(train,test_size=test_percent,random_state=15)
    x_t_train, x_t_test,y_t_train,y_t_test = train_test_split(x_train,y_train,test_size=test_percent,random_state=15)
    pct = len(train_test)/10

    list1 = [GradientBoostingRegressor,RandomForestRegressor,SVR]
    list2 = [param1,param2,param3]
    list3 = ['GradientBoost','RandomForest','SVM']
    MSEs = {}

    def knnlist(k, t_test, t_train):
        indexlist = []
        for i in t_test.index:
            diff = t_train - t_test.loc[i]
            df_t = t_train.copy()
            df_t['dist'] = diff.abs().sum(axis=1)
            sort_k = df_t.sort_values(['dist'], ascending=True).head(k)
            idx = sort_k.index.tolist()
            indexlist += idx
        return indexlist

    for (regressor,param,name) in zip(list1,list2,list3):

        # original machine learning model
        if param is not None:
            model = regressor(**param)
        else:
            model = regressor()
        model.fit(x_train,y_train)
        y0 = model.predict(x_test)
        mse0 = ((y0 - y_test) ** 2).sum() / len(y_test)
        MSEs['{}0'.format(name)]= mse0
        print ('\nMSE of original {}: '.format(name) + str(mse0))

        # use train_train set to find outliers of train_test set
        model.fit(x_t_train, y_t_train)
        pred_t = model.predict(x_t_test)
        err = (pred_t - y_t_test).abs()
        train_test['err'] = err
        bot = train_test.sort_values(by=['err']).tail(pct)

        indexlist = knnlist(10, bot[feature], x_t_train)
        indexlist = indexlist + bot.index.tolist()
        # get cleaned training set
        train_clean = train.drop(indexlist)
        # get cleaned testing set
        indexlist2 = knnlist(5, bot[feature], x_test)
        test_clean = test.drop(indexlist2)

        x1_train, x1_test = train_clean[feature], test_clean[feature]
        y1_train, y1_test = train_clean[target], test_clean[target]

        # machine learning on new dataset
        model.fit(x1_train, y1_train)
        y1 = model.predict(x1_test)
        mse1 = ((y1 - y1_test) ** 2).sum() / len(y1_test)
        MSEs['{}1'.format(name)] = mse1
        print ('\nMSE of new {}: '.format(name) + str(mse1))

    return MSEs



