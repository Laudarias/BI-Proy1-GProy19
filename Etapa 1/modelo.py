import pandas as pd
import numpy as np
import joblib
import re
import unicodedata
import inflect
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class TextPreprocessing:
    def __init__(self, stopwords=stopwords.words('spanish')):
        self.stopwords = stopwords
        self.stemmer = SnowballStemmer('spanish')
        self.lemmatizer = WordNetLemmatizer()

    def remove_non_ascii(self, words):
        return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]

    def to_lowercase(self, words):
        return [word.lower() for word in words]

    def remove_punctuation(self, words):
        return [re.sub(r'[^\w\s]', '', word) for word in words if re.sub(r'[^\w\s]', '', word) != '']

    def replace_numbers(self, words):
        p = inflect.engine()
        return [p.number_to_words(word) if word.isdigit() else word for word in words]

    def remove_stopwords(self, words):
        return [word for word in words if word not in self.stopwords]

    def stem_words(self, words):
        return [self.stemmer.stem(word) for word in words]

    def lemmatize_verbs(self, words):
        return [self.lemmatizer.lemmatize(word, pos='v') for word in words]

    def preproccesing(self, words):
        words = self.to_lowercase(words)
        words = self.replace_numbers(words)
        words = self.remove_punctuation(words)
        words = self.remove_non_ascii(words)
        words = self.remove_stopwords(words)
        return words

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.Series(X)
        X = X.apply(word_tokenize)
        X = X.apply(lambda x: self.preproccesing(x))
        X = X.apply(lambda x: self.stem_words(x))
        X = X.apply(lambda x: ' '.join(x))
        return X

def train_and_save_model(data_file):
    # Cargar datos
    data = pd.read_excel(data_file)

    # Preprocesar datos
    data['TextosT'] = data['Textos_espanol'].apply(word_tokenize)
    data['TextosT'] = data['TextosT'].apply(lambda x: ' '.join(TextPreprocessing().preproccesing(x)))

    train, test = train_test_split(data, test_size=0.2, random_state=33)
    X_train, y_train = train['TextosT'], train['sdg']

    estimators = [
        ('preprocess', TextPreprocessing()),
        ('transform', CountVectorizer(lowercase=False)),
        ('classifier', MultinomialNB())
    ]
    
    pipe_NB = Pipeline(estimators)
    pipe_NB.fit(X_train, y_train)

    # Guardar el modelo entrenado
    joblib.dump(pipe_NB, 'modelo_entrenado.joblib')
    print("Modelo guardado como modelo_entrenado.joblib")

if __name__ == "__main__":
    train_and_save_model("datos.xlsx") 
