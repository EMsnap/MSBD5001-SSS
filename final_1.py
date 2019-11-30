import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, cross_val_score, train_test_split
from sklearn import tree

train_ori = pd.read_csv("train.csv", parse_dates=['purchase_date', 'release_date'])

train_ori['purchase_date'] = train_ori.purchase_date.fillna(method='backfill')
train_ori['total_positive_reviews'] = train_ori.total_positive_reviews.fillna(method='backfill')
train_ori['total_negative_reviews'] = train_ori.total_negative_reviews.fillna(method='backfill')
test_ori = pd.read_csv("test.csv", parse_dates=['purchase_date', 'release_date'])

test_ori['purchase_date'] = test_ori.purchase_date.fillna(method='backfill')
test_ori['total_positive_reviews'] = test_ori.total_positive_reviews.fillna(method='backfill')
test_ori['total_negative_reviews'] = test_ori.total_negative_reviews.fillna(method='backfill')

###########   cate_dum


train_cate_dum = train_ori["categories"].str.get_dummies(",")
test_cate_dum = test_ori["categories"].str.get_dummies(",")

cate_test = train_cate_dum.columns.difference(test_cate_dum.columns)
cate_train = test_cate_dum.columns.difference(train_cate_dum.columns)
print(train_ori["categories"].head())


test_cate_dum = pd.concat([test_cate_dum, pd.DataFrame(columns=list(cate_test))], axis=1).fillna(0)


col_cate_list = []
print (train_cate_dum.shape)
for col in train_cate_dum.columns:
    if train_cate_dum[col].value_counts()[1] < 180:
        train_cate_dum.drop(col, axis=1, inplace=True)
    else:
        col_cate_list.append(col)
print (col_cate_list)
print (train_cate_dum.shape)



###########   genres_dum

train_genres_dummy = train_ori["genres"].str.get_dummies(",")
test_genres_dummy = test_ori["genres"].str.get_dummies(",")
genre_test = train_genres_dummy.columns.difference(test_genres_dummy.columns)
genres_train = test_genres_dummy.columns.difference(train_genres_dummy.columns)
test_genres_dummy = pd.concat([test_genres_dummy, pd.DataFrame(columns=list(genre_test))], axis=1).fillna(0)
print(genre_test)
print(genres_train)

print (train_genres_dummy.shape)
col_genres_list = []
for col in train_genres_dummy.columns:
    if train_genres_dummy[col].value_counts()[1] < 180:
        train_genres_dummy.drop(col, axis=1, inplace=True)
    else :
        col_genres_list.append(col)
print (col_genres_list)
print (train_genres_dummy.shape)



###########   tags_dum

train_tags_dummy = train_ori["tags"].str.get_dummies(",")
test_tags_dummy = test_ori["tags"].str.get_dummies(",")
tags_train = train_tags_dummy.columns.difference(test_tags_dummy.columns)
tags_test = test_tags_dummy.columns.difference(train_tags_dummy.columns)
print(tags_train)
print(tags_test)



def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x)
    rmse= np.sqrt(-cross_val_score(model, train_x, train_y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# Date process



def date_process(df_copy, df, column):
    df_copy[column+'_year']=df[column].apply(lambda x: x.year)
    df_copy[column+'_month']=df[column].apply(lambda x: x.month)
    df_copy[column+'_day']=df[column].apply(lambda x: x.day)
    return df_copy

train_copy = train_ori.copy()
train_purchase = date_process(train_copy, train_ori, 'purchase_date')
train_purchase_copy = train_purchase.copy()
train_date = date_process(train_purchase_copy, train_purchase, 'release_date')
train_date['date_interval'] = (train_date['purchase_date'] - train_date['release_date'])\
.apply(lambda x: x.days)

test_copy = test_ori.copy()
test_purchase = date_process(test_copy, test_ori, 'purchase_date')

test_purchase_copy = test_purchase.copy()
test_date = date_process(test_purchase_copy, test_purchase, 'release_date')
test_date['date_interval'] = (test_date['purchase_date'] - test_date['release_date'])\
.apply(lambda x: x.days)



######### dummy concat


test_tags_dummy = pd.concat([test_tags_dummy, pd.DataFrame(columns=list(tags_train))], axis=1).fillna(0)
train_tags_dummy = pd.concat([train_tags_dummy, pd.DataFrame(columns=list(tags_test))], axis=1).fillna(0)
train = pd.concat([train_date, train_cate_dum, train_genres_dummy, train_tags_dummy], axis=1)
test = pd.concat([test_date, test_cate_dum, test_genres_dummy, test_tags_dummy], axis=1)

train_final = train.drop(['categories', 'genres', 'tags', 'purchase_date', 'release_date'], axis=1)
test_final = test.drop(['categories', 'genres', 'tags', 'purchase_date', 'release_date'], axis=1)

train_x = train_final.drop(['playtime_forever'], axis=1)
train_y = train_final[['playtime_forever']]



######### model



dt = tree.DecisionTreeRegressor()
dt.fit(train_x,train_y)
pred = dt.predict(test_final)
pred = pd.DataFrame(pred)
pred.columns=['playtime_forever']
res = pd.DataFrame({'id':test_ori['id'], 'playtime_forever':pred['playtime_forever']})

from sklearn.model_selection import cross_val_score
scores = np.sqrt(-cross_val_score(dt, train_x, train_y, cv=50,scoring='neg_mean_squared_error'))
score = np.mean(scores)

from sklearn.model_selection import KFold, cross_val_score, train_test_split
#Validation function
n_folds = 10


score = rmsle_cv(dt)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


print ('score:',score)
print (res)
res.to_csv('submit.csv',index=False,header=1)


'''