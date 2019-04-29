import pandas as pd
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from spacy.lang.en.stop_words import STOP_WORDS
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


df_spambots2 = pd.read_csv("./sample_dataset/cresci-2017.csv/datasets_full.csv/social_spambots_2.csv/users.csv")
df_spambots3 = pd.read_csv("./sample_dataset/cresci-2017.csv/datasets_full.csv/social_spambots_3.csv/users.csv")
df_traditional_spambots1 = pd.read_csv("./sample_dataset/cresci-2017.csv/datasets_full.csv/traditional_spambots_1.csv/users.csv")
df_fake_follower = pd.read_csv("./sample_dataset/cresci-2017.csv/datasets_full.csv/fake_followers.csv/users.csv")
df_russian = pd.read_csv("./sample_dataset/russian-troll-tweets/users.csv")
df_genuine = pd.read_csv("./sample_dataset/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/users.csv")

sample_number = 100 # 500 for whole dataset
sub_sample_number = 10 # 100 for whole dataset


def bots_cleaning(df_genuine):
    df_genuine_new = df_genuine.copy()
    df_genuine_new = df_genuine_new[['id','statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    df_genuine_new['target']=1

    df_genuine_new.fillna(0, inplace=True)
    df_genuine_new=df_genuine_new.replace(np.nan, 0)
    return df_genuine_new

def genuine_cleaning(df_genuine):
    df_genuine_new = df_genuine.copy()
    df_genuine_new = df_genuine_new[['id','statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    df_genuine_new['target']=0

    df_genuine_new.fillna(0, inplace=True)
    df_genuine_new=df_genuine_new.replace(np.nan, 0)
    return df_genuine_new


text_spambots2_bow = bots_cleaning(df_spambots2)
print(text_spambots2_bow.shape)
text_spambots2_bow = text_spambots2_bow.sample(n=sample_number,random_state=0)
print(text_spambots2_bow.shape)

text_spambots3_bow = bots_cleaning(df_spambots3)
print(text_spambots3_bow.shape)

text_traditional1_bow = bots_cleaning(df_traditional_spambots1)
print(text_traditional1_bow.shape)
text_traditional1_bow = text_traditional1_bow.sample(n=sample_number,random_state=0)
print(text_traditional1_bow.shape)

text_genuine_bow = genuine_cleaning(df_genuine)
print(text_genuine_bow.shape)

text_russian_bow = bots_cleaning(df_russian)
print(text_russian_bow.shape)

text_fake_follower_bow = bots_cleaning(df_fake_follower)
print(text_fake_follower_bow.shape)
text_fake_follower_bow = text_fake_follower_bow.sample(n=sample_number,random_state=0)
print(text_fake_follower_bow.shape)

text_clf_lr = RandomForestClassifier(n_estimators=100, max_depth=20, criterion='entropy',random_state=0)

plot_df = pd.DataFrame()
train = []
test = []
micro = []
macro = []
add=[]

text_genuine_bow_train,text_genuine_bow_test = train_test_split(text_genuine_bow, test_size=0.5, random_state=0)

########################################################
def adding_func(text_spambots2_bow,text_spambots3_bow,training_set='social2',adding_set='social3'):
    train_frames = [text_genuine_bow_train,text_spambots2_bow]
    df_train = pd.concat(train_frames)
    X_train = df_train[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow]
    df_test = pd.concat(test_frames)
    X_test = df_test[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('0')
    train.append(training_set)
    test.append(adding_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))

    ########################################################

    text_spambots3_bow_5 = text_spambots3_bow.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_5.shape)
    text_spambots3_bow_95 = text_spambots3_bow[~text_spambots3_bow.id.isin(text_spambots3_bow_5.id)]
    print(text_spambots3_bow_95.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5]
    df_train = pd.concat(train_frames)
    X_train = df_train[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_95]
    df_test = pd.concat(test_frames)
    X_test = df_test[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('100')
    train.append(training_set)
    test.append(adding_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))

    ########################################################

    text_spambots3_bow_10 = text_spambots3_bow_95.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_10.shape)
    text_spambots3_bow_90 = text_spambots3_bow_95[~text_spambots3_bow_95.id.isin(text_spambots3_bow_10.id)]
    print(text_spambots3_bow_90.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5,text_spambots3_bow_10]
    df_train = pd.concat(train_frames)
    X_train = df_train[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_90]
    df_test = pd.concat(test_frames)
    X_test = df_test[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('200')
    train.append(training_set)
    test.append(adding_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))

    ########################################################

    text_spambots3_bow_15 = text_spambots3_bow_90.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_15.shape)
    text_spambots3_bow_85 = text_spambots3_bow_90[~text_spambots3_bow_90.id.isin(text_spambots3_bow_15.id)]
    print(text_spambots3_bow_85.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5,text_spambots3_bow_10,text_spambots3_bow_15]
    df_train = pd.concat(train_frames)
    X_train = df_train[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_85]
    df_test = pd.concat(test_frames)
    X_test = df_test[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('300')
    train.append(training_set)
    test.append(adding_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))

    ########################################################

    text_spambots3_bow_20 = text_spambots3_bow_85.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_20.shape)
    text_spambots3_bow_80 = text_spambots3_bow_85[~text_spambots3_bow_85.id.isin(text_spambots3_bow_20.id)]
    print(text_spambots3_bow_80.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5,text_spambots3_bow_10,text_spambots3_bow_15,text_spambots3_bow_20]
    df_train = pd.concat(train_frames)
    X_train = df_train[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_80]
    df_test = pd.concat(test_frames)
    X_test = df_test[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('400')
    train.append(training_set)
    test.append(adding_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))

datasets_name_list = ['social2','social3','traditional1','russian','fake']
datasets_list = [text_spambots2_bow,text_spambots3_bow,text_traditional1_bow,text_russian_bow,text_fake_follower_bow]
num_list = [0,1,2,3,4]
for i in range(len(datasets_list)):
    dataset_train = datasets_list[i]
    dataset_name_train = datasets_name_list[i]
    for ele in datasets_name_list:
        ele_index = datasets_name_list.index(ele)
        if ele_index != i:
            dataset_target = datasets_list[ele_index]
            dataset_name_target = datasets_name_list[ele_index]
            adding_func(dataset_train,dataset_target,adding_set=dataset_name_target,training_set=dataset_name_train)


def adding_func_within(text_spambots2_bow,training_set='social2'):
    genuine_train,genuine_test = train_test_split(text_genuine_bow_train, test_size=0.3, random_state=0)
    social2_train,social2_test = train_test_split(text_spambots2_bow, test_size=0.3, random_state=0)

    train_frames = [genuine_train,social2_train]
    df_train = pd.concat(train_frames)
    X_train = df_train[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_train = df_train['target']

    test_frames = [genuine_test,social2_test]
    df_test = pd.concat(test_frames)
    X_test = df_test[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('0')
    train.append(training_set)
    test.append(training_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))



    text_spambots3_bow_5 = df_test.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_5.shape)
    text_spambots3_bow_95 = df_test[~df_test.id.isin(text_spambots3_bow_5.id)]
    print(text_spambots3_bow_95.shape)

    train_frames = [df_train,text_spambots3_bow_5]
    df_train = pd.concat(train_frames)
    X_train = df_train[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_95]
    df_test = pd.concat(test_frames)
    X_test = df_test[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('100')
    train.append(training_set)
    test.append(training_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))

    ########################################################

    text_spambots3_bow_10 = text_spambots3_bow_95.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_10.shape)
    text_spambots3_bow_90 = text_spambots3_bow_95[~text_spambots3_bow_95.id.isin(text_spambots3_bow_10.id)]
    print(text_spambots3_bow_90.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5,text_spambots3_bow_10]
    df_train = pd.concat(train_frames)
    X_train = df_train[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_90]
    df_test = pd.concat(test_frames)
    X_test = df_test[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('200')
    train.append(training_set)
    test.append(training_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))


    ########################################################

    text_spambots3_bow_15 = text_spambots3_bow_90.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_15.shape)
    text_spambots3_bow_85 = text_spambots3_bow_90[~text_spambots3_bow_90.id.isin(text_spambots3_bow_15.id)]
    print(text_spambots3_bow_85.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5,text_spambots3_bow_10,text_spambots3_bow_15]
    df_train = pd.concat(train_frames)
    X_train = df_train[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_85]
    df_test = pd.concat(test_frames)
    X_test = df_test[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
            'verified']]
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('300')
    train.append(training_set)
    test.append(training_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))


    ########################################################

    text_spambots3_bow_20 = text_spambots3_bow_85.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_20.shape)
    text_spambots3_bow_80 = text_spambots3_bow_85[~text_spambots3_bow_85.id.isin(text_spambots3_bow_20.id)]
    print(text_spambots3_bow_80.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5,text_spambots3_bow_10,text_spambots3_bow_15,text_spambots3_bow_20]
    df_train = pd.concat(train_frames)
    X_train = df_train[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_80]
    df_test = pd.concat(test_frames)
    X_test = df_test[['statuses_count', 'followers_count',
           'friends_count', 'favourites_count', 'listed_count',
           'verified']]
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('400')
    train.append(training_set)
    test.append(training_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))

datasets_name_list = ['social2','social3','traditional1','russian','fake']
datasets_list = [text_spambots2_bow,text_spambots3_bow,text_traditional1_bow,text_russian_bow,text_fake_follower_bow]

for i in range(len(datasets_list)):
    dataset = datasets_list[i]
    dataset_name = datasets_name_list[i]
    adding_func_within(dataset, training_set=dataset_name)

plot_df['add'] = add
plot_df['train'] = train
plot_df['test'] = test
plot_df['micro'] = micro
plot_df['macro'] = macro
plot_df.to_csv("profile_feature_rf_result.csv",index=False)



