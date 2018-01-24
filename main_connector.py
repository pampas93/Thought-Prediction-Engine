from word_predictor import WordPredictor
import nltk
nltk.data.path.append('nltk_data/')

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import emoji
import pickle

def predictWords(text):

    with open('WordPredictionFile', 'rb') as wf:
        wordObject = pickle.loads(wf.read())

    words = wordObject.predict(text).terms()[0:30]
    finalWords = []
    i = 0
    for word in words:
        if i == 4:
            break
        if word[0].isalpha():
            finalWords.append(word[0])
            i = i + 1

    return finalWords

en_stopwords = set(stopwords.words('english'))
snowball_stemmer = SnowballStemmer('english')

def linguistic_preprocess(tweet):
    without_stopwords = [w for w in tweet.split() if w not in en_stopwords]
    stemmed = [snowball_stemmer.stem(w) for w in without_stopwords]
    return ' '.join(stemmed)


all_emojis = emoji.EMOJI_ALIAS_UNICODE.keys()
emoji_id_mapper = {emoji: id for (id, emoji) in enumerate(all_emojis)}
id_emoji_mapper = {id: emoji for (id, emoji) in enumerate(all_emojis)}


def predict(text, vectorizer, classifier):
    cleaned = linguistic_preprocess(text)
    vector = vectorizer.transform([cleaned])
    prediction = classifier.predict(vector.toarray())[0]
    emoji_name = id_emoji_mapper[prediction]
    return emoji_name
