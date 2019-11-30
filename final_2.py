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

train_cate_dum = train_ori["categories"].str.get_dummies(",")
test_cate_dum = test_ori["categories"].str.get_dummies(",")
cate_test = train_cate_dum.columns.difference(test_cate_dum.columns)
cate_train = test_cate_dum.columns.difference(train_cate_dum.columns)
print(train_ori["categories"].head())
print(cate_test)
print(cate_train)

for col in cate_test:
    train_cate_dum.drop(col, axis=1, inplace=True)

print (train_cate_dum.shape)
print (test_cate_dum.shape)
# test_cate_dum = pd.concat([test_cate_dum, pd.DataFrame(columns=list(cate_test))], axis=1).fillna(0)
train_genres_dummy = train_ori["genres"].str.get_dummies(",")
test_genres_dummy = test_ori["genres"].str.get_dummies(",")
genre_test = train_genres_dummy.columns.difference(test_genres_dummy.columns)
genres_train = test_genres_dummy.columns.difference(train_genres_dummy.columns)

print(genre_test)
print(genres_train)

for col in genre_test:
    train_genres_dummy.drop(col, axis=1, inplace=True)

# test_genres_dummy = pd.concat([test_genres_dummy, pd.DataFrame(columns=list(genre_test))], axis=1).fillna(0)
print (train_genres_dummy.shape)
print (test_genres_dummy.shape)

train_tags_dummy = train_ori["tags"].str.get_dummies(",")
test_tags_dummy = test_ori["tags"].str.get_dummies(",")
tags_train = train_tags_dummy.columns.difference(test_tags_dummy.columns)
tags_test = test_tags_dummy.columns.difference(train_tags_dummy.columns)
print(tags_train)
print(tags_test)

for col in tags_test:
    test_tags_dummy.drop(col, axis=1, inplace=True)
for col in tags_train:
    train_tags_dummy.drop(col, axis=1, inplace=True)

print (test_tags_dummy.shape)
print (train_tags_dummy.shape)

for col in train_tags_dummy.columns:
    if train_tags_dummy[col].value_counts()[1] < 80:
        train_tags_dummy.drop(col, axis=1, inplace=True)
        test_tags_dummy.drop(col, axis=1, inplace=True)

print (test_tags_dummy.shape)
print (train_tags_dummy.shape)

#test_tags_dummy = pd.concat([test_tags_dummy, pd.DataFrame(columns=list(tags_train))], axis=1).fillna(0)
#train_tags_dummy = pd.concat([train_tags_dummy, pd.DataFrame(columns=list(tags_test))], axis=1).fillna(0)


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x)
    rmse= np.sqrt(-cross_val_score(model, train_x, train_y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

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


categories = train_cate_dum.columns
genres_copy = train_genres_dummy.columns
tags_copy = train_tags_dummy.columns

c = tags_copy
b = genres_copy
a = categories
cb = [val for val in c if val in b]
ca = [val for val in c if val in a]
ab = [val for val in a if val in b]
print(cb)
print (ca)
print (ab)

# df['value'] = df.apply(lambda row: my_test(row['c1'], row['c2']), axis=1)
print (len(test_tags_dummy.columns))

for col_name in cb:
    genres = train_genres_dummy[col_name]
    test_genres = test_genres_dummy[col_name]
    tags = train_tags_dummy[col_name]
    test_tags = test_tags_dummy[col_name]
    length = len(genres)
    for i in range(length):
        genres[i] = genres[i] or tags[i]
    for i in range(len(test_genres)):
        test_genres[i] = test_genres[i] or test_tags[i]
    train_tags_dummy.drop([col_name], axis = 1, inplace=True)
    test_tags_dummy.drop([col_name], axis = 1, inplace=True)
print (len(test_tags_dummy.columns))


for col_name in ca:
    genres = train_cate_dum[col_name]
    test_genres = test_cate_dum[col_name]
    tags = train_tags_dummy[col_name]
    new_genres = []
    length = len(genres)
    for i in range(length):
        genres[i] = genres[i] or tags[i]
    for i in range(len(test_genres)):
        test_genres[i] = test_genres[i] or tags[i]
    train_tags_dummy.drop([col_name], axis = 1, inplace=True)
    test_tags_dummy.drop([col_name], axis = 1, inplace=True)
print (len(test_tags_dummy.columns))


train = pd.concat([train_date, train_cate_dum, train_genres_dummy, train_tags_dummy], axis=1)
test = pd.concat([test_date, test_cate_dum, test_genres_dummy, test_tags_dummy], axis=1)

train_final = train.drop(['categories', 'genres', 'tags', 'purchase_date', 'release_date'], axis=1)
test_final = test.drop(['categories', 'genres', 'tags', 'purchase_date', 'release_date'], axis=1)
train_x = train_final.drop(['playtime_forever'], axis=1)
train_y = train_final[['playtime_forever']]

print (train_x.shape)

dt = tree.DecisionTreeRegressor()


dt.fit(train_x,train_y)
test_input_renamed = test_final[train_x.columns]

pred = dt.predict(test_input_renamed)
pred = pd.DataFrame(pred)
pred.columns=['playtime_forever']
res = pd.DataFrame({'id':test_ori['id'], 'playtime_forever':pred['playtime_forever']})

from sklearn.model_selection import cross_val_score
scores = np.sqrt(-cross_val_score(dt, train_x, train_y, cv=50,scoring='neg_mean_squared_error'))
score = np.mean(scores)
print (score)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
#Validation function
n_folds = 20

score = rmsle_cv(dt)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print ('score:',score)
print (res)
# res.to_csv('submit.csv',index=False,header=1)




from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

GBoost = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05,
                                   max_depth=5, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
GBoost.fit(train_x,train_y)
GBoost_y_pred = GBoost.predict(test_input_renamed)
# print (GBoost_y_pred)

pred = pd.DataFrame(GBoost_y_pred)
print (pred)
pred[pred < 0] = 0
pred.columns=['playtime_forever']
res = pd.DataFrame({'id':test_ori['id'], 'playtime_forever':pred['playtime_forever']})

#rms = sqrt(mean_squared_error(y_val, GBoost_y_pred))
#print("GBoost_y_pred:", rms)

#score = rmsle_cv(GBoost)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


############# Random forest
####随机森林回归####

print (train_x.shape)
print (train_y.shape)
from sklearn import ensemble
rf_model = ensemble.RandomForestRegressor(n_estimators=10)
rf_model.fit(train_x, train_y)
pred_rf = rf_model.predict(test_input_renamed)
pred_rf_df = pd.DataFrame(pred_rf)
pred_rf_df.columns=['playtime_forever']
res_rf = pd.DataFrame({'id':test_ori["id"], 'playtime_forever':pred_rf_df['playtime_forever']})

from sklearn.model_selection import cross_val_score
rf_score = np.sqrt(-cross_val_score(rf_model, train_x, train_y, cv=50, scoring='neg_mean_squared_error'))
mean_score = np.mean(rf_score)
print(rf_score)
print('mean_scores_rf_2_big:',mean_score)

print (res_rf)
#score = rmsle_cv(rf_model)
res_rf.to_csv('submit.csv',index=False,header=1)
#print (score)

