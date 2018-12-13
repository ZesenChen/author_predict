
# coding: utf-8

# **Objective of the competition:**
#
# The competition dataset contains text from works of fiction written by spooky authors of the public domain:
#  1. Edgar Allan Poe (EAP)
#  2. HP Lovecraft (HPL)
#  3. Mary Wollstonecraft Shelley (MWS)
#
# The objective  is to accurately identify the author of the sentences in the test set.
#
# **Objective of the notebook:**
#
# In this notebook, let us try to create different features that will help us in identifying the spooky authors.
#
# As a first step, we will do some basic data visualization and cleaning before we delve deep into the feature engineering part.

# In[15]:




# CURRENT BEST : 0.29346821217, 0.291994132135(12h)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
from nltk.tokenize import RegexpTokenizer
import nltk.stem as stm
from nltk import WordNetLemmatizer, word_tokenize
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import re
import string
from time import time

from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

models = [('MultiNB', MultinomialNB(alpha=0.03)),
          ('Calibrated MultiNB', CalibratedClassifierCV(
              MultinomialNB(alpha=0.03), method='isotonic')),
          ('Calibrated BernoulliNB', CalibratedClassifierCV(
              BernoulliNB(alpha=0.03), method='isotonic')),
          ('Calibrated Huber', CalibratedClassifierCV(
              SGDClassifier(loss='modified_huber', alpha=1e-4,
                            max_iter=10000, tol=1e-4), method='sigmoid')),
          ('Logit', LogisticRegression(C=30))]

start_time = time()
color = sns.color_palette()
tqdm.pandas()
alphabet = 'abcdefghijklmnopqrstuvwxyz'
_punctuation = ['.', '..', '...', ',', ':', ';', '-', '*', '"', '!', '?']
embeddings_index = {}
f = open('glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# get_ipython().magic(u'matplotlib inline')

eng_stopwords = set(stopwords.words("english"))         # for wipe out some common words that hava no help such as "I", "you", "and"
pd.options.mode.chained_assignment = None               # prevent raise an exception
alpha_tokenizer = RegexpTokenizer('[A-Za-z]\w+')
lemmatizer = WordNetLemmatizer()

# In[2]:


## Read the train and test dataset and check the top few lines ##
train_df = pd.read_csv("newtrain.csv")
test_df = pd.read_csv("test.csv")
print("Number of rows in train dataset : ",train_df.shape[0])
print("Number of rows in test dataset : ",test_df.shape[0])


# In[31]:


def fraction_noun(row):
    """function to give us fraction of noun over total words """
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    return (noun_count/word_count)

def fraction_adj(row):
    """function to give us fraction of adjectives over total words in given text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    adj_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    return (adj_count/word_count)

def fraction_verbs(row):
    """function to give us fraction of verbs over total words in given text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    verbs_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return (verbs_count/word_count)

def fraction_adverbs(row):
    """function to give us fraction of verbs over total words in given text"""
    text = row['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    verbs_count = len([w for w in pos_list if w[1] in ('RB','RBR','RBS')])
    return (verbs_count/word_count)


def clean_text(x):
    x.lower()
    for p in _punctuation:
        x.replace(p, '')
    return x

def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in eng_stopwords]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


# In[17]:


train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train_df["num_unique_words"] = train_df["text"].apply(lambda x: len(set(str(x).split())))
test_df["num_unique_words"] = test_df["text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train_df["num_chars"] = train_df["text"].apply(lambda x: len(str(x)))
test_df["num_chars"] = test_df["text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train_df["num_stopwords"] = train_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test_df["num_stopwords"] = test_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
train_df["num_punctuations"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test_df["num_punctuations"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train_df["num_words_upper"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test_df["num_words_upper"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train_df["num_words_title"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test_df["num_words_title"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train_df["mean_word_len"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df["mean_word_len"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

train_df[","] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split(",")]))
test_df[","] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split(",")]))

train_df[";"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split(";")]))
test_df[";"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split(";")]))

train_df['\"'] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split('\"')]))
test_df['\"'] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split('\"')]))

train_df["..."] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split("...")]))
test_df["..."] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split("...")]))

train_df["?"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split("?")]))
test_df["?"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split("?")]))

train_df["!"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split("!")]))
test_df["!"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split("!")]))

train_df["."] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split(".")]))
test_df["."] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split(".")]))

train_df[":"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split(":")]))
test_df[":"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split(":")]))

train_df["*"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split("*")]))
test_df["*"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split("*")]))

train_df["-"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split("-")]))
test_df["-"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split("-")]))

train_df['fraction_noun'] = train_df.apply(lambda row: fraction_noun(row), axis =1)
test_df['fraction_noun'] = test_df.apply(lambda row: fraction_noun(row), axis =1)

train_df['fraction_adj'] = train_df.apply(lambda row: fraction_adj(row), axis =1)
test_df['fraction_adj'] = test_df.apply(lambda row: fraction_adj(row), axis =1)

train_df['fraction_verbs'] = train_df.apply(lambda row: fraction_verbs(row), axis =1)
test_df['fraction_verbs'] = test_df.apply(lambda row: fraction_verbs(row), axis =1)


most_words = ['strange', 'night', 'ancient', 'terrible', 'house', 'street', 'black', 'dark', 'city', 'remain', ''
          'moon', 'west', 'told', 'looked', 'dreams', 'door', 'stone', 'half', 'left','found', 'course', 'observe',
          'head', 'person', 'length', 'water', 'character', 'moment', 'manner', 'air', 'ider', 'speak', 'place',
            'hand', 'matter', 'de', 'feet', 'body', 'means', 'doubt','raymond', 'perdita', 'adrian', 'fall', 'come',
            'father', 'country', 'heart', 'idris', 'spirit', 'love', 'life', 'say', 'find', 'thing', 'long', 'dream',
          'idris', 'tears', 'passed', 'nature', 'fear', 'human', 'voice', 'dear', 'words', 'great', 'little', 'see',
          'the ', ' a ', 'appear', 'little', 'was ', 'one ', 'two ', 'three ', 'ten ', 'is ', 'are ', 'ed ', 'misery',
            'however', ' to ', 'into', 'about ', 'th', 'er', 'ex', 'an ', 'ground', 'any', 'silence', 'wall', 'look'
            , 'The ', 'I ', 'It ', 'He', 'Me', 'They ', 'She ', 'We ', 'You ', 'good', 'time', 'old', 'death', 'man']

_punctuation = ['.', '..', '...', ',', ':', ';', '-', '*', '"', '!', '?']

train_df['text_cleaned'] = train_df['text'].apply(lambda x: clean_text(x))
test_df['text_cleaned'] = test_df['text'].apply(lambda x: clean_text(x))

for word in most_words:
    train_df[word] = train_df["text_cleaned"].str.count(word)
    test_df[word] = test_df["text_cleaned"].str.count(word)

for char in alphabet:
    train_df['num_'+char] = train_df["text_cleaned"].str.count(char)
    test_df['num_'+char] = test_df["text_cleaned"].str.count(char)


# We can check the number of occurrence of each of the author to see if the classes are balanced.

# In[36]:


# train_df = train_df.drop(['fraction_adverbs'], axis=1)


xtrain_glove = [sent2vec(x) for x in tqdm(train_df.text)]
xtest_glove = [sent2vec(x) for x in tqdm(test_df.text)]
xtrain_glove = np.array(xtrain_glove)
xtest_glove = np.array(xtest_glove)
# print xtrain_glove.shape, xtest_glove.shape

train_df = pd.concat([train_df, pd.DataFrame(xtrain_glove)], axis=1)
test_df = pd.concat([test_df, pd.DataFrame(xtest_glove)], axis=1)
print train_df.info()


# This looks good. There is not much class imbalance. Let us print some lines of each of the authors to try and understand their writing style if possible.

# In[37]:


author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
train_y = train_df['author'].map(author_mapping_dict)
train_id = train_df['id'].values
test_id = test_df['id'].values


cols_to_drop = ['id', 'text', 'text_cleaned']
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)


# Only thing I can see is that there are quite a few special characters present in the text data. So count of these special characters might be a good feature. Probably we can create them later.
#
# Apart from that, I do not have much clue.. In case if you find any interesting styles (features which we can create), please add them in the comments.
#
# **Feature Engineering:**
#
# Now let us come try to do some feature engineering. This consists of two main parts.
#
#  1. Meta features - features that are extracted from the text like number of words, number of stop words, number of punctuations etc
#  2. Text based features - features directly based on the text / words like frequency, svd, word2vec etc.
#
# **Meta Features:**
#
# We will start with creating meta featues and see how good are they at predicting the spooky authors. The feature list is as follows:
# 1. Number of words in the text
# 2. Number of unique words in the text
# 3. Number of characters in the text
# 4. Number of stopwords
# 5. Number of punctuations
# 6. Number of upper case words
# 7. Number of title case words
# 8. Average length of the words
#

# In[38]:


def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 3
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = child
    param['subsample'] = 0.8
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 2000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model

def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def runBer(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.BernoulliNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model


# Let us now plot some of our new variables to see of they will be helpful in predictions.

# In[39]:

vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[2,3,3,1,1])

tfidf_vec = CountVectorizer(analyzer='word', ngram_range=(1, 5))
tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_test_y = clf.predict_proba(test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# add the predictions as new features #
train_df["nb_word2_eap"] = pred_train[:,0]
train_df["nb_word2_hpl"] = pred_train[:,1]
train_df["nb_word2_mws"] = pred_train[:,2]
test_df["nb_word2_eap"] = pred_full_test[:,0]
test_df["nb_word2_hpl"] = pred_full_test[:,1]
test_df["nb_word2_mws"] = pred_full_test[:,2]


vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[2,3,3,1,1])

tfidf_vec = CountVectorizer(analyzer='char_wb', ngram_range=(1, 5))
tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_test_y = clf.predict_proba(test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# add the predictions as new features #
train_df["nb_c2_eap"] = pred_train[:,0]
train_df["nb_c2_hpl"] = pred_train[:,1]
train_df["nb_c2_mws"] = pred_train[:,2]
test_df["nb_c2_eap"] = pred_full_test[:,0]
test_df["nb_c2_hpl"] = pred_full_test[:,1]
test_df["nb_c2_mws"] = pred_full_test[:,2]


# EAP seems slightly lesser number of words than MWS and HPL.

# In[40]:

vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[2,3,3,1,1])

tfidf_vec = CountVectorizer(analyzer='word', ngram_range=(1, 5))
tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runBer(dev_X, dev_y, val_X, val_y, test_tfidf)
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_test_y = clf.predict_proba(test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# add the predictions as new features #
train_df["ber_word2_eap"] = pred_train[:,0]
train_df["ber_word2_hpl"] = pred_train[:,1]
train_df["ber_word2_mws"] = pred_train[:,2]
test_df["ber_word2_eap"] = pred_full_test[:,0]
test_df["ber_word2_hpl"] = pred_full_test[:,1]
test_df["ber_word2_mws"] = pred_full_test[:,2]


vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[3,2,3,1,1])

tfidf_vec = CountVectorizer(analyzer='char_wb', ngram_range=(1, 5))
tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runBer(dev_X, dev_y, val_X, val_y, test_tfidf)
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_test_y = clf.predict_proba(test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# add the predictions as new features #
train_df["ber_c2_eap"] = pred_train[:,0]
train_df["ber_c2_hpl"] = pred_train[:,1]
train_df["ber_c2_mws"] = pred_train[:,2]
test_df["ber_c2_eap"] = pred_full_test[:,0]
test_df["ber_c2_hpl"] = pred_full_test[:,1]
test_df["ber_c2_mws"] = pred_full_test[:,2]


# This also seems to be somewhat useful. Now let us focus on creating some text based features.
#
# Let us first build a basic model to see how these meta features  are helping.

# In[41]:


tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

train_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)

del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd


# **Naive Bayes on Word Count Vectorizer:**

# In[22]:


### Fit transform the count vectorizer ###
vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[2,3,3,1,1])

tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_test_y = clf.predict_proba(test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# add the predictions as new features #
train_df["nb_cvec_eap"] = pred_train[:,0]
train_df["nb_cvec_hpl"] = pred_train[:,1]
train_df["nb_cvec_mws"] = pred_train[:,2]
test_df["nb_cvec_eap"] = pred_full_test[:,0]
test_df["nb_cvec_hpl"] = pred_full_test[:,1]
test_df["nb_cvec_mws"] = pred_full_test[:,2]


# We can train a simple XGBoost model.

# In[42]:


'''tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
'''
vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[3,2,3,1,1])

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runBer(dev_X, dev_y, val_X, val_y, test_tfidf)
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_test_y = clf.predict_proba(test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# add the predictions as new features #
train_df["ber_cvec_eap"] = pred_train[:,0]
train_df["ber_cvec_hpl"] = pred_train[:,1]
train_df["ber_cvec_mws"] = pred_train[:,2]
test_df["ber_cvec_eap"] = pred_full_test[:,0]
test_df["ber_cvec_hpl"] = pred_full_test[:,1]
test_df["ber_cvec_mws"] = pred_full_test[:,2]


vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[2,3,3,1,1])

tfidf_vec = CountVectorizer(ngram_range=(1,8), analyzer='char')
tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runBer(dev_X, dev_y, val_X, val_y, test_tfidf)
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_test_y = clf.predict_proba(test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# add the predictions as new features #
train_df["ber_cvec_char_eap"] = pred_train[:,0]
train_df["ber_cvec_char_hpl"] = pred_train[:,1]
train_df["ber_cvec_char_mws"] = pred_train[:,2]
test_df["ber_cvec_char_eap"] = pred_full_test[:,0]
test_df["ber_cvec_char_hpl"] = pred_full_test[:,1]
test_df["ber_cvec_char_mws"] = pred_full_test[:,2]
#----------------------------------------------------------------------------------------


### Fit transform the tfidf vectorizer ###
'''tfidf_vec = CountVectorizer(ngram_range=(1,8), analyzer='char')
tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
'''
vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[3,2,3,1,1])

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_test_y = clf.predict_proba(test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# add the predictions as new features #
train_df["nb_cvec_char_eap"] = pred_train[:,0]
train_df["nb_cvec_char_hpl"] = pred_train[:,1]
train_df["nb_cvec_char_mws"] = pred_train[:,2]
test_df["nb_cvec_char_eap"] = pred_full_test[:,0]
test_df["nb_cvec_char_hpl"] = pred_full_test[:,1]
test_df["nb_cvec_char_mws"] = pred_full_test[:,2]



# The cross val score is very high and is 3.75. But this might add some different information than word level features and so let us use this for the final model as well.
#
# **Naive Bayes on Character Tfidf Vectorizer:**
#
# Let us also get the naive bayes predictions on the character tfidf vectorizer.

# In[ ]:


### Fit transform the tfidf vectorizer ###
vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[3,2,3,1,1])

tfidf_vec = TfidfVectorizer(ngram_range=(1, 5), analyzer='char')
full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_test_y = clf.predict_proba(test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# add the predictions as new features #
train_df["nb_tfidf_char_eap"] = pred_train[:,0]
train_df["nb_tfidf_char_hpl"] = pred_train[:,1]
train_df["nb_tfidf_char_mws"] = pred_train[:,2]
test_df["nb_tfidf_char_eap"] = pred_full_test[:,0]
test_df["nb_tfidf_char_hpl"] = pred_full_test[:,1]
test_df["nb_tfidf_char_mws"] = pred_full_test[:,2]

#111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
'''tfidf_vec = TfidfVectorizer(ngram_range=(1, 5), analyzer='char')
full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
'''
vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[2,3,3,1,1])

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runBer(dev_X, dev_y, val_X, val_y, test_tfidf)
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_test_y = clf.predict_proba(test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# add the predictions as new features #
train_df["ber_tfidf_char_eap"] = pred_train[:,0]
train_df["ber_tfidf_char_hpl"] = pred_train[:,1]
train_df["ber_tfidf_char_mws"] = pred_train[:,2]
test_df["ber_tfidf_char_eap"] = pred_full_test[:,0]
test_df["ber_tfidf_char_hpl"] = pred_full_test[:,1]
test_df["ber_tfidf_char_mws"] = pred_full_test[:,2]

vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[2,3,3,1,1])

tfidf_vec = TfidfVectorizer(ngram_range=(1,5), analyzer='word')
full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runBer(dev_X, dev_y, val_X, val_y, test_tfidf)
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_test_y = clf.predict_proba(test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# add the predictions as new features #
train_df["ber_tfidf2_char_eap"] = pred_train[:,0]
train_df["ber_tfidf2_char_hpl"] = pred_train[:,1]
train_df["ber_tfidf2_char_mws"] = pred_train[:,2]
test_df["ber_tfidf2_char_eap"] = pred_full_test[:,0]
test_df["ber_tfidf2_char_hpl"] = pred_full_test[:,1]
test_df["ber_tfidf2_char_mws"] = pred_full_test[:,2]
#111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111

#-------------------------------------------------------------------------------------
'''tfidf_vec = TfidfVectorizer(ngram_range=(1,5), analyzer='word')
full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
'''
vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[3,2,3,1,1])

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_test_y = clf.predict_proba(test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# add the predictions as new features #
train_df["nb_tfidf2_char_eap"] = pred_train[:,0]
train_df["nb_tfidf2_char_hpl"] = pred_train[:,1]
train_df["nb_tfidf2_char_mws"] = pred_train[:,2]
test_df["nb_tfidf2_char_eap"] = pred_full_test[:,0]
test_df["nb_tfidf2_char_hpl"] = pred_full_test[:,1]
test_df["nb_tfidf2_char_mws"] = pred_full_test[:,2]
#-------------------------------------------------------------------------------------

# **SVD on Character TFIDF:**
#
# We could also create svd features on character tfidf features and used them for modeling.

# In[ ]:


n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

train_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)

del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd


# For the sake of kernel run time, we can just check the first fold in the k-fold cross validation for the scores. Please remove the 'break' line while running in local.

# In[45]:





##############################################################

vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[3,3,3,1,1])
X_train = vectorizer.fit_transform(train_df.text.values)
authors = ['MWS','EAP','HPL']
y_train = train_df.author.apply(authors.index).values
X_test = vectorizer.transform(test_df.text.values)

kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
for dev_index, val_index in kf.split(X_train):
    dev_X, val_X = X_train[dev_index], X_train[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    clf.fit(dev_X, dev_y)
    pred_val_y = clf.predict_proba(val_X)
    pred_full_test = pred_full_test + clf.predict_proba(X_test)
    #pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    break
print("cv scores : ", cv_scores)
pred_full_test /= 10

train_df["em_tfidf_eap"] = pred_train[:,0]
train_df["em_tfidf_hpl"] = pred_train[:,1]
train_df["em_tfidf_mws"] = pred_train[:,2]
test_df["em_tfidf_eap"] = pred_full_test[:,0]
test_df["em_tfidf_hpl"] = pred_full_test[:,1]
test_df["em_tfidf_mws"] = pred_full_test[:,2]
##############################################################

train_df.to_csv("train_26699.csv", index=False)
test_df.to_csv("test_26699.csv", index=False)

cols_to_drop = ['id', 'text', 'text_cleaned']
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)


kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.4)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("cv scores : ", cv_scores)
print sum(cv_scores) / 10
pred_full_test = pred_full_test / 10.

out_df = pd.DataFrame(pred_full_test)
out_df.columns = ['EAP', 'HPL', 'MWS']
out_df.insert(0, 'id', test_id)
out_df.to_csv("submission.csv", index=False)

end_time = time()
print 'totally time cost: %dm %.2fs' % ((end_time-start_time)/60, (end_time-start_time)%60)

# We are getting a mlogloss of '0.987' using just the meta features. Not a bad score. Now let us see which of these features are important.

# In[12]:


### Plot the important variables ###
'''
fig, ax = plt.subplots(figsize=(12,12))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()
'''

fig, ax = plt.subplots(figsize=(120,120))
xgb.plot_importance(model, max_num_features=500, height=0.8, ax=ax)
fig.savefig('test.png', dpi=100)
'''

# Naive bayes features are the top features as expected. Now let us get the confusion matrix to see the misclassification errors.

# In[ ]:


import itertools
from sklearn.metrics import confusion_matrix

### From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py #
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
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

cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y,axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, classes=['EAP', 'HPL', 'MWS'],
                      title='Confusion matrix, without normalization')
plt.show()
'''

# EAP and MWS seem to be misclassified more often than others. We could potentially create features which improves the predictions for this pair.
#
# **Next steps in this FE notebook:**
# * Using word embedding based features
# * Other meta features if any

# **Ideas for further improvements:**
# * Parameter tuning for tfidf and count vectorizer
# * Parameter tuning for naive bayes and XGB models
# * Ensembling / Stacking with other models

#
# **More to come. Stay tuned.!**
