from flask import Flask, render_template, request, redirect, url_for

from pipeline import vectorizer, textPreprocessing, getPrediction

from logger import logging

app = Flask(__name__,)

logging.info('App started')

data = dict()
reviews = []
positive = 0
negative = 0

@app.route('/')

def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative

    logging.info('Data sent to main.html')

    return render_template("main.html", data=data)

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']

    logging.info('Text received from user')

    preprocess_txt = textPreprocessing(text)
    logging.info('Text preprocessed')

    vectorized_txt = vectorizer(preprocess_txt)
    logging.info('Text vectorized')


    prediction = getPrediction(vectorized_txt)
    logging.info('Prediction made')

    if prediction == 'Positive':
        global positive
        positive += 1
    if prediction == 'Negative':
        global negative
        negative += 1

    reviews.append(text)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
