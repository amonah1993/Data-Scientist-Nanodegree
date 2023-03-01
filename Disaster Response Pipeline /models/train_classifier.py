import sys
import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessage', engine)
    x = df['message']
    y = df.iloc[:,4:]
    categoryNames = list(df.columns[4:])

    return x, y, categoryNames


def tokenize(text):
    token = word_tokenize(text)
    wordLemmatizer = WordNetLemmatizer()
    tokenClean = []
    for t in token:
        tokenC= lemmatizer.lemmatize(t).lower().strip()
        tokenClean.append(tokenC)

    return tokenClean


def build_model():
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__criterion':['entropy'],
        'vect__smooth_idf':[True]
    }
    gridCv = GridSearchCV(pipeline,parameters)
    
    return gridCv


def evaluate_model(model, X_test, Y_test, category_names):
    yPredict = model.predict(X_test)



def save_model(model, model_filepath):
    pickTemp=open(model_filepath, 'wb')
    pickle.dump(model,pickTemp )


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()