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

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier


def get_text_df(data_frame):
    data_frame_text = data_frame.copy()
    data_frame_text = data_frame_text[['user_id','text']]
    return data_frame_text

def clean_text(data_frame_text):
    bots_df_clean_regularexpression = data_frame_text.copy()

    regex_pat = re.compile(r'\@\s?[a-zA-z\']+')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['text'].str.replace(regex_pat, '')
    regex_pat2 = re.compile(r'&amp;')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['cleaned_text'].str.replace(regex_pat2, '')
    regex_pat4 = re.compile(r'http://\S+')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['cleaned_text'].str.replace(regex_pat4, '')
    regex_pat5 = re.compile(r'RT')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['cleaned_text'].str.replace(regex_pat5, '')
    regex_pat6 = re.compile(r'\.')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['cleaned_text'].str.replace(regex_pat6, '')
    regex_pat7 = re.compile(r'\#')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['cleaned_text'].str.replace(regex_pat7, '')
    regex_pat8 = re.compile(r'\,')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['cleaned_text'].str.replace(regex_pat8, '')
    regex_pat9 = re.compile(r'\(')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['cleaned_text'].str.replace(regex_pat9, '')
    regex_pat10 = re.compile(r'\)')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['cleaned_text'].str.replace(regex_pat10, '')
    regex_pat11 = re.compile(r'\...')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['cleaned_text'].str.replace(regex_pat11, '')
    regex_pat12 = re.compile(r'[^\'\s\w]')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['cleaned_text'].str.replace(regex_pat12, '')
    regex_pat13 = re.compile(r'\_+')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['cleaned_text'].str.replace(regex_pat12, '')
    regex_pat14 = re.compile(r'https://\S+')
    bots_df_clean_regularexpression['cleaned_text'] = bots_df_clean_regularexpression['cleaned_text'].str.replace(regex_pat14, '')
    
    return bots_df_clean_regularexpression

def clean_stop_word(bots_df_clean_regularexpression):
    clean_stop_word_df = bots_df_clean_regularexpression.copy()
    clean_stop_word_df['splited'] = clean_stop_word_df.cleaned_text.apply(lambda x:str(x).split())
    clean_stop_word_df['lowercase'] = clean_stop_word_df.cleaned_text.apply(lambda x:str(x).lower())
    clean_stop_word_df['lowercase_splited'] = clean_stop_word_df.lowercase.apply(lambda x:str(x).split())
    clean_stop_word_df['clean_stopwords'] = clean_stop_word_df['lowercase_splited'].apply(lambda x: [i for i in x if i not in STOP_WORDS])
    return clean_stop_word_df

def count(clean_stop_word_df):
    df_with_count = clean_stop_word_df.copy()
    df_with_count['num_words_raw'] = df_with_count.splited.apply(lambda x:len(x))
    df_with_count['num_words_after_clean'] = df_with_count.clean_stopwords.apply(lambda x:len(x))
    return df_with_count

def back_to_string(df_with_count):
    cleaned_string_df = df_with_count.copy()
    cleaned_string_df['cleaned_text_string'] = cleaned_string_df['clean_stopwords'].apply(lambda x:' '.join(x))
    return cleaned_string_df

def group_to_string(cleaned_string_df):
    grouped_df = cleaned_string_df.copy()
    grouped_df = grouped_df[['user_id','cleaned_text_string']]
    grouped_df = grouped_df.groupby('user_id').agg(lambda x: x.sum())
    return grouped_df

def group_to_list(cleaned_string_df):
    grouped_df = cleaned_string_df.copy()
    grouped_df = grouped_df[['user_id','cleaned_text_string']]
    grouped_df = grouped_df.groupby('user_id').agg(lambda x: x.tolist())
    grouped_df['len'] = grouped_df['cleaned_text_string'].apply(lambda x:len(x))
    # len_list = grouped_df.len.tolist()
    length = grouped_df.len.median
    print(length)
    return length

def most_common_word(text_spambots2_grouped):
    text_spambots2_text_list = text_spambots2_grouped.cleaned_text_string.tolist()
    text_spambots2_text_all = ' '.join(text_spambots2_text_list)
    text_spambots2_text_all_splited = text_spambots2_text_all.split()
    text_spambots2_text_counter = Counter(text_spambots2_text_all_splited)
    output = text_spambots2_text_counter.most_common(20)
    return output,text_spambots2_text_counter

def get_word2vec_mean(grouped_df):
    word2vec_df = grouped_df.copy()
    word2vec_df['prepare'] = word2vec_df['cleaned_text_string'].apply(lambda x:x.split())
    word2vec_df['word2vec'] = word2vec_df['prepare'].apply(lambda x:np.array([model[i] for i in x if i in model]))
    word2vec_df['word2vec_mean'] = word2vec_df['word2vec'].apply(lambda x:np.mean(x,axis=0))
    return word2vec_df



df_spambots2 = pd.read_csv("./sample_dataset/cresci-2017.csv/datasets_full.csv/social_spambots_2.csv/tweets.csv",low_memory=False)
df_spambots3 = pd.read_csv("./sample_dataset/cresci-2017.csv/datasets_full.csv/social_spambots_3.csv/tweets.csv",low_memory=False)
df_traditional_spambots1 = pd.read_csv("./sample_dataset/cresci-2017.csv/datasets_full.csv/traditional_spambots_1.csv/tweets.csv",low_memory=False)
df_fake_follower = pd.read_csv("./sample_dataset/cresci-2017.csv/datasets_full.csv/fake_followers.csv/tweets.csv",low_memory=False)
df_russian = pd.read_csv("./sample_dataset/russian-troll-tweets/tweets.csv",low_memory=False)
df_genuine = pd.read_csv("./sample_dataset/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/tweets.csv",low_memory=False)

sample_number = 5 # 500 for whole dataset
sub_sample_number = 1 # 5 for whole dataset


df_spam2_tranformed = df_spambots2.copy()
df_spam2_tranformed = get_text_df(df_spam2_tranformed)
df_spam2_tranformed = clean_text(df_spam2_tranformed)
df_spam2_tranformed = clean_stop_word(df_spam2_tranformed)
df_spam2_tranformed = count(df_spam2_tranformed)
df_spam2_tranformed = back_to_string(df_spam2_tranformed)
df_spam2_tranformed.head()

df_spam3_tranformed = df_spambots3.copy()
df_spam3_tranformed = get_text_df(df_spam3_tranformed)
df_spam3_tranformed = clean_text(df_spam3_tranformed)
df_spam3_tranformed = clean_stop_word(df_spam3_tranformed)
df_spam3_tranformed = count(df_spam3_tranformed)
df_spam3_tranformed = back_to_string(df_spam3_tranformed)
df_spam3_tranformed.head()

df_traditional1_tranformed = df_traditional_spambots1.copy()
df_traditional1_tranformed = get_text_df(df_traditional1_tranformed)
df_traditional1_tranformed = clean_text(df_traditional1_tranformed)
df_traditional1_tranformed = clean_stop_word(df_traditional1_tranformed)
df_traditional1_tranformed = count(df_traditional1_tranformed)
df_traditional1_tranformed = back_to_string(df_traditional1_tranformed)
df_traditional1_tranformed.head()

df_genuine_tranformed = df_genuine.copy()
df_genuine_tranformed = get_text_df(df_genuine_tranformed)
df_genuine_tranformed = clean_text(df_genuine_tranformed)
df_genuine_tranformed = clean_stop_word(df_genuine_tranformed)
df_genuine_tranformed = count(df_genuine_tranformed)
df_genuine_tranformed = back_to_string(df_genuine_tranformed)
df_genuine_tranformed.head()

df_fake_follower_tranformed = df_fake_follower.copy()
df_fake_follower_tranformed = get_text_df(df_fake_follower_tranformed)
df_fake_follower_tranformed = clean_text(df_fake_follower_tranformed)
df_fake_follower_tranformed = clean_stop_word(df_fake_follower_tranformed)
df_fake_follower_tranformed = count(df_fake_follower_tranformed)
df_fake_follower_tranformed = back_to_string(df_fake_follower_tranformed)
df_fake_follower_tranformed.head()

df_russian_tranformed = df_russian.copy()
df_russian_tranformed = get_text_df(df_russian_tranformed)
df_russian_tranformed = clean_text(df_russian_tranformed)
df_russian_tranformed = clean_stop_word(df_russian_tranformed)
df_russian_tranformed = count(df_russian_tranformed)
df_russian_tranformed = back_to_string(df_russian_tranformed)
df_russian_tranformed.head()



text_spambots2_word2vec = df_spam2_tranformed.copy()
print('text_spambots2_word2vec')
group_to_list(text_spambots2_word2vec)
text_spambots2_word2vec = group_to_string(text_spambots2_word2vec)

text_spambots2_bow = text_spambots2_word2vec.copy()
text_spambots2_bow = text_spambots2_bow[['cleaned_text_string']]
text_spambots2_bow['target']=1
# print(text_spambots2_bow.shape)
text_spambots2_bow = text_spambots2_bow.sample(n=sample_number,random_state=0)
# print(text_spambots2_bow.shape)

text_spambots3_word2vec = df_spam3_tranformed.copy()
print('text_spambots3_word2vec')
group_to_list(text_spambots3_word2vec)
text_spambots3_word2vec = group_to_string(text_spambots3_word2vec)

# text_spambots3_word2vec = get_word2vec_mean(text_spambots3_word2vec)
text_spambots3_word2vec.head()
text_spambots3_bow = text_spambots3_word2vec.copy()
text_spambots3_bow = text_spambots3_bow[['cleaned_text_string']]
text_spambots3_bow['target']=1
# print(text_spambots3_bow.shape)

text_traditional1_word2vec = df_traditional1_tranformed.copy()
print('text_traditional1_word2vec')
group_to_list(text_traditional1_word2vec)
text_traditional1_word2vec = group_to_string(text_traditional1_word2vec)

# text_traditional1_word2vec = get_word2vec_mean(text_traditional1_word2vec)
text_traditional1_word2vec.head()
text_traditional1_bow = text_traditional1_word2vec.copy()
text_traditional1_bow = text_traditional1_bow[['cleaned_text_string']]
text_traditional1_bow['target']=1
# print(text_traditional1_bow.shape)
text_traditional1_bow = text_traditional1_bow.sample(n=sample_number,random_state=0)
# print(text_traditional1_bow.shape)

text_genuine_word2vec = df_genuine_tranformed.copy()
print('text_genuine_word2vec')
group_to_list(text_genuine_word2vec)
text_genuine_word2vec = group_to_string(text_genuine_word2vec)

# text_genuine_word2vec = get_word2vec_mean(text_genuine_word2vec)
text_genuine_word2vec.head()
text_genuine_bow = text_genuine_word2vec.copy()
text_genuine_bow = text_genuine_bow[['cleaned_text_string']]
text_genuine_bow['target']=0
# print(text_genuine_bow.shape)

text_russian_word2vec = df_russian_tranformed.copy()
print('text_russian_word2vec')
group_to_list(text_russian_word2vec)
text_russian_word2vec = group_to_string(text_russian_word2vec)

# text_genuine_word2vec = get_word2vec_mean(text_genuine_word2vec)
text_russian_word2vec.head()
text_russian_bow = text_russian_word2vec.copy()
text_russian_bow = text_russian_bow[['cleaned_text_string']]
text_russian_bow['target']=1
# print(text_russian_bow.shape)

text_fake_follower_word2vec = df_fake_follower_tranformed.copy()
print('text_fake_follower_word2vec')
group_to_list(text_fake_follower_word2vec)
text_fake_follower_word2vec = group_to_string(text_fake_follower_word2vec)

# text_genuine_word2vec = get_word2vec_mean(text_genuine_word2vec)
text_fake_follower_word2vec.head()
text_fake_follower_bow = text_fake_follower_word2vec.copy()
text_fake_follower_bow = text_fake_follower_bow[['cleaned_text_string']]
text_fake_follower_bow['target']=1
# print(text_fake_follower_bow.shape)
text_fake_follower_bow = text_fake_follower_bow.sample(n=sample_number,random_state=0)
# print(text_fake_follower_bow.shape)

text_clf_lr = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=5, random_state=0))
])

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
    X_train = df_train['cleaned_text_string']
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow]
    df_test = pd.concat(test_frames)
    X_test = df_test['cleaned_text_string']
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
    text_spambots3_bow_95 = text_spambots3_bow[~text_spambots3_bow.cleaned_text_string.isin(text_spambots3_bow_5.cleaned_text_string)]
    print(text_spambots3_bow_95.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5]
    df_train = pd.concat(train_frames)
    X_train = df_train['cleaned_text_string']
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_95]
    df_test = pd.concat(test_frames)
    X_test = df_test['cleaned_text_string']
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('5')
    train.append(training_set)
    test.append(adding_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))

    ########################################################

    text_spambots3_bow_10 = text_spambots3_bow_95.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_10.shape)
    text_spambots3_bow_90 = text_spambots3_bow_95[~text_spambots3_bow_95.cleaned_text_string.isin(text_spambots3_bow_10.cleaned_text_string)]
    print(text_spambots3_bow_90.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5,text_spambots3_bow_10]
    df_train = pd.concat(train_frames)
    X_train = df_train['cleaned_text_string']
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_90]
    df_test = pd.concat(test_frames)
    X_test = df_test['cleaned_text_string']
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('10')
    train.append(training_set)
    test.append(adding_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))

    ########################################################

    text_spambots3_bow_15 = text_spambots3_bow_90.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_15.shape)
    text_spambots3_bow_85 = text_spambots3_bow_90[~text_spambots3_bow_90.cleaned_text_string.isin(text_spambots3_bow_15.cleaned_text_string)]
    print(text_spambots3_bow_85.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5,text_spambots3_bow_10,text_spambots3_bow_15]
    df_train = pd.concat(train_frames)
    X_train = df_train['cleaned_text_string']
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_85]
    df_test = pd.concat(test_frames)
    X_test = df_test['cleaned_text_string']
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('15')
    train.append(training_set)
    test.append(adding_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))

    ########################################################

    text_spambots3_bow_20 = text_spambots3_bow_85.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_20.shape)
    text_spambots3_bow_80 = text_spambots3_bow_85[~text_spambots3_bow_85.cleaned_text_string.isin(text_spambots3_bow_20.cleaned_text_string)]
    print(text_spambots3_bow_80.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5,text_spambots3_bow_10,text_spambots3_bow_15,text_spambots3_bow_20]
    df_train = pd.concat(train_frames)
    X_train = df_train['cleaned_text_string']
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_80]
    df_test = pd.concat(test_frames)
    X_test = df_test['cleaned_text_string']
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('20')
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
    ########################################################
    genuine_train,genuine_test = train_test_split(text_genuine_bow_train, test_size=0.3, random_state=0)
    social2_train,social2_test = train_test_split(text_spambots2_bow, test_size=0.3, random_state=0)

    train_frames = [genuine_train,social2_train]
    df_train = pd.concat(train_frames)
    X_train = df_train['cleaned_text_string']
    y_train = df_train['target']

    test_frames = [genuine_test,social2_test]
    df_test = pd.concat(test_frames)
    X_test = df_test['cleaned_text_string']
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
    text_spambots3_bow_95 = df_test[~df_test.cleaned_text_string.isin(text_spambots3_bow_5.cleaned_text_string)]
    print(text_spambots3_bow_95.shape)

    train_frames = [df_train,text_spambots3_bow_5]
    df_train = pd.concat(train_frames)
    X_train = df_train['cleaned_text_string']
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_95]
    df_test = pd.concat(test_frames)
    X_test = df_test['cleaned_text_string']
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('5')
    train.append(training_set)
    test.append(training_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))

    ########################################################

    text_spambots3_bow_10 = text_spambots3_bow_95.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_10.shape)
    text_spambots3_bow_90 = text_spambots3_bow_95[~text_spambots3_bow_95.cleaned_text_string.isin(text_spambots3_bow_10.cleaned_text_string)]
    print(text_spambots3_bow_90.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5,text_spambots3_bow_10]
    df_train = pd.concat(train_frames)
    X_train = df_train['cleaned_text_string']
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_90]
    df_test = pd.concat(test_frames)
    X_test = df_test['cleaned_text_string']
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('10')
    train.append(training_set)
    test.append(training_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))


    ########################################################

    text_spambots3_bow_15 = text_spambots3_bow_90.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_15.shape)
    text_spambots3_bow_85 = text_spambots3_bow_90[~text_spambots3_bow_90.cleaned_text_string.isin(text_spambots3_bow_15.cleaned_text_string)]
    print(text_spambots3_bow_85.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5,text_spambots3_bow_10,text_spambots3_bow_15]
    df_train = pd.concat(train_frames)
    X_train = df_train['cleaned_text_string']
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_85]
    df_test = pd.concat(test_frames)
    X_test = df_test['cleaned_text_string']
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('15')
    train.append(training_set)
    test.append(training_set) #'social3'
    micro.append(f1_score(y_true, y_pred, average='micro'))
    macro.append(f1_score(y_true, y_pred, average='macro'))


    ########################################################

    text_spambots3_bow_20 = text_spambots3_bow_85.sample(n=sub_sample_number,random_state=0)
    print(text_spambots3_bow_20.shape)
    text_spambots3_bow_80 = text_spambots3_bow_85[~text_spambots3_bow_85.cleaned_text_string.isin(text_spambots3_bow_20.cleaned_text_string)]
    print(text_spambots3_bow_80.shape)

    train_frames = [text_genuine_bow_train,text_spambots2_bow,text_spambots3_bow_5,text_spambots3_bow_10,text_spambots3_bow_15,text_spambots3_bow_20]
    df_train = pd.concat(train_frames)
    X_train = df_train['cleaned_text_string']
    y_train = df_train['target']

    test_frames = [text_genuine_bow_test,text_spambots3_bow_80]
    df_test = pd.concat(test_frames)
    X_test = df_test['cleaned_text_string']
    y_test = df_test['target']

    text_clf_lr.fit(X_train, y_train)
    predicted = text_clf_lr.predict(X_test)

    y_true = y_test
    y_pred = predicted
    add.append('20')
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
plot_df.to_csv("text_feature_boosting_result.csv",index=False)
