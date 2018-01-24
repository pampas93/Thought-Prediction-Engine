import pickle
from flask import Flask, request
from flask import render_template
#
import emoji
import main_connector

app = Flask(__name__)


with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.loads(f.read())

with open('SGDClassifier', 'rb') as f:
    classifier = pickle.loads(f.read())

#tfWord = Train_Word_Predictor()

@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == 'GET':
        return render_template('index.html', emoji="")
    else:
        text = request.form['tweet']
        emo = str(main_connector.predict(text, vectorizer, classifier))
        #finalWords = tfWord.predict_words(text)
        #finalWords = []
        finalWords = main_connector.predictWords(text)
        finalWords.append("")
        finalWords.append("")
        finalWords.append("")


        return render_template('index.html', emoji=emoji.emojize(emo), content=text, word1=finalWords[0], word2=finalWords[1], word3=finalWords[2])


if __name__ == '__main__':
    app.run(debug=True)
