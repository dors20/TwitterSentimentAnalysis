import pandas as pd
import numpy as np
import nltk
import spacy
import re
import string
import warnings
from IPython.core.display import display
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier, DMatrix, cv
warnings.filterwarnings('ignore')

df = pd.read('validation.csv')

df.dropna(inplace=True)

blanks = [] 
for i,txt,lb in df.itertuples():  # iterate over the DataFrame
    if type(txt)==str:            # avoid NaN values
        if txt.isspace():         # test 'text' for whitespace
            blanks.append(i)     # add matching index numbers to the list

df.drop(blanks, inplace=True)

nlp = spacy.load("en")
sp = spacy.load('en_core_web_md')
all_stopwords = sp.Defaults.stop_words

def process_tweets(text):    
    text = str(text).lower() #lower
    text = re.sub('\[.*?\]', '', text) #Remove text in square brackets
    text = re.sub('https?://\S+|www\.\S+', '', text) #Hyperlinks removal
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) #punctuations
    text = re.sub('\n', '', text) #newlines
    text = re.sub('\w*\d\w*', '', text) #word containing numbers
    tokens = word_tokenize(text) #tokenizing the tweet
    filtered_sentence = " ".join([w for w in tokens if not w in all_stopwords]) #removing stopwords
    clean_text = " ".join([w.lemma_ for w in nlp(filtered_sentence)]) #Lemmatization of words
    return clean_text

df['clean_text'] = df['text'].apply(lambda x:process_tweets(x))

X = df['clean_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

def output_metrics(predictions):
    ''' 
    Predictions are the predicted values for the test data 
    '''
    print("-------Accuracy Score--------")
    print(accuracy_score(y_test, predictions))
    accuracy_list.append(accuracy_score(y_test, predictions))
    print('\n')
    print("-------Classification Report--------")
    print(classification_report(y_test,predictions))
    print("-------Confusion Matrix--------")
    conf_mat = confusion_matrix(y_test,predictions)
    conf_mat_df = pd.DataFrame(data=conf_mat,columns=['negative','neutral','positive'],index=['negative','neutral','positive'])
    display(conf_mat_df)
    
    
def train_model(model,X_train,y_train,X_test,y_test):
    '''
    model            :  Model which is going to be used 
    X_train ,y_train :  features and labels of training data
    X_test, y_test   :  features and labels of testing data
    '''
    model_list.append(model)
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    tweet_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('model', model)])                    #doing Tf-idf on the data 
    scores = cross_val_score(tweet_clf, X_train, y_train, cv=cv)
    print("Cross Validation scores")
    print(scores)
    tweet_clf.fit(X_train,y_train)
    predictions = tweet_clf.predict(X_test)
    output_metrics(predictions)

xgb = joblib.load('xgb.pkl') 
train_model(xgb, X_train, y_train, X_test, y_test)
