from flask import Flask, redirect, url_for, render_template, request
import requests
import pickle
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


DATA_JSON_FILE = 'SpamData/01_Processing/email-text-data.json'
data = pd.read_json(DATA_JSON_FILE)
vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit_transform(data.MESSAGE)
model = pickle.load(open('model/spam_email.pkl', 'rb'))


app = Flask(__name__)


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        mail = request.form['mail']
        data = [mail]
        print(data)
        data_trans = vectorizer.transform(data)
        a = model.predict(data_trans)
        prediction = a[0]
        if prediction == 1:
            return render_template('index.html', prediction_text="The email is spam")
        elif prediction == 0:
            return render_template('index.html', prediction_text="The email is not spam")
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run()
