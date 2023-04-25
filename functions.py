import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def generate_word_cloud(dataset, column1, mask=None):
    stopwords = set(STOPWORDS)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for i, label in enumerate([0, 1]):
        df_subset = dataset[dataset['label'] == label]
        text = " ".join(text for text in df_subset[column1])

        wordcloud = WordCloud(stopwords=stopwords,
                              background_color='white',
                              colormap='Dark2',
                              contour_color='#5d0f24',
                              contour_width=3,
                              max_words=200).generate(text)

        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'Word Cloud for df[\'real\'] == {label}', fontsize=20)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def remove_stopwords_and_punctuation(df, columns):
    stopwords_set = set(stopwords.words('english'))
    for col in columns:
        df[col] = df[col].str.lower().apply(nltk.word_tokenize)
        df[col] = df[col].apply(lambda x: [word for word in x if word not in stopwords_set and word.isalpha()])
        df[col] = df[col].apply(lambda x: [word for word in x if word not in punctuation])
        df[col] = df[col].apply(lambda x: ' '.join(x))


def tokenize(df, col):
    tokenized_title = []

    for title in df[col]:
        tokens_title = word_tokenize(title)
        tokenized_title.append(tokens_title)

    df[col] = tokenized_title

    return df


def stem_tokens(tokens):
    stemmer = SnowballStemmer('english')

    def stem_word(word):
        return stemmer.stem(word)

    def stem_words(words):
        return [stem_word(word) for word in words]

    return stem_words(tokens)


def create_bow_df(df, col):
    df[col] = df[col].apply(lambda x: ' '.join(x))
    vectorizer = CountVectorizer(min_df=2, max_features=1000)

    X = vectorizer.fit_transform(df[col])
    feature_names = vectorizer.get_feature_names_out()
    bow_df = pd.DataFrame(X.toarray(), columns=feature_names)

    bow_df = pd.DataFrame(X.toarray(), columns=feature_names)

    return bow_df


def create_b2v_df(df, col):
    # convert text into lists of words
    text = [[word for word in document] for document in df[col]]

    model = Word2Vec(text, vector_size=100, window=5, min_count=1, workers=4)

    # create a dictionary with each word and its corresponding vector
    word_vectors = {word: model.wv[word] for word in model.wv.key_to_index.keys()}

    # convert the texts into Bag of Words vectors
    vectors = []
    for doc in text:
        vector = np.zeros(model.vector_size)
        for word in doc:
            vector += word_vectors[word]
        vectors.append(vector)

    b2v_df = pd.DataFrame(vectors)

    return b2v_df


def split_data(X, y, test_size=0.15, val_size=0.15):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size / (1 - test_size))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
