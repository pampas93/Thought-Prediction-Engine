from word_predictor import WordPredictor
import nltk
from nltk.corpus import wordnet


class Train_Word_Predictor:

    def __init__(self):
        self.wp = WordPredictor()
        for corpus in nltk.corpus.gutenberg.fileids():
            self.wp.learn_from_text(nltk.corpus.gutenberg.raw(corpus))
        print("Word prediction training ready")
    # def train_wordPredictor(self):
    #     #wp = WordPredictor()



    def predict_words(self, text):
        words = self.wp.predict(text).terms()[0:30]
        finalWords = []
        i=0
        for word in words :
            if i==4:
                break
            if word[0].isalpha():
                finalWords.append(word[0])
                i=i+1

        print(finalWords)
        return finalWords