import numpy as np
import pandas as pd
import re    #Regular expression library
import string
import nltk
from collections import Counter
vocab=Counter()
import pickle

def removePunc(text):
    for punc in string.punctuation:
        text=text.replace(punc,' ')
    return text


with open('static/model/corpora/stopwords/english', 'r') as stp:  # Adjusted path
    sw=stp.read().splitlines()


from nltk.stem import PorterStemmer
st=PorterStemmer()

def textPreprocessing(txt):
    data=pd.DataFrame([txt],columns=['tweet'])    
    data['tweet']=data['tweet'].apply(lambda x: " " .join(x.lower() for x in x.split()))
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(re.sub(r'https?://\S+', '', word) for word in x.split()))
    data['tweet'] = data['tweet'].apply(removePunc)
    data['tweet'] = data['tweet'].str.replace(r'\d+', '', regex=True)
    data['tweet']=data['tweet'].apply(lambda x: " " .join(x.lower() for x in x.split() if x not in sw)) 
    data['tweet']=data['tweet'].apply(lambda x: " " .join(st.stem(x) for x in x.split()))

    return data['tweet']

with open('static/model/model.pickle','rb') as model:
    model=pickle.load(model)

vocab=pd.read_csv('static/model/vocabulary.txt',header=None)
token=vocab[0].tolist()

def vectorizer(dataset):
    vectorized_lst=[]  #create empty list
    for sentence in dataset:  #go sentence by sentence in x
        sentence_lst=np.zeros(len(token))   #initialize the 0 list with respect to vocab size
        for i in range(len(token)):    # go word by word in sentece in vocab size
            if token[i] in sentence.split():  #if found vocab's word then replace 0 value by 1
                sentence_lst[i] = 1
        vectorized_lst.append(sentence_lst)   #append into another list as a list 
    vectorized_lst_new=np.asarray(vectorized_lst, dtype=np.float32)   #create fully completed list into np array
    return vectorized_lst_new


def getPrediction(vectorized_txt):
    pred=model.predict(vectorized_txt)
    if pred==0:
        return "Positive"
    else:
        return "Negative"
    

