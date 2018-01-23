from word_predictor import WordPredictor
import nltk
import pickle
from nltk.corpus import wordnet


class Train_Word_Predictor:

    def __init__(self):
        self.wp = WordPredictor()
        for corpus in nltk.corpus.gutenberg.fileids():
            self.wp.learn_from_text(nltk.corpus.gutenberg.raw(corpus))

        with open("WordPredictionFile", 'wb') as f:
            f.write(pickle.dumps(self.wp))

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

    def predict_fromDisk(self, text):

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

if __name__ == '__main__':

    # tfWord = Train_Word_Predictor()
    # finalWords = tfWord.predict_fromDisk("Good morning");
    finalWords = predictWords("Good morning to")
    print(finalWords)