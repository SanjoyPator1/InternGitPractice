# -*- coding: utf-8 -*-

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np

import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json', encoding="utf8").read()
intents = json.loads(data_file)

def count():
    global counter
    

counter = 0
#a = True

'''try:
    for i in  range(8):
        counter=counter+1
        if (counter==3):
            raise Exception("manual exception and counter is - ")
        if (counter==5):
            a=False
except Exception as error:
    print("The counter now is - broken while -jcnsjdc the counter is - ",counter)'''

#print("The counter now is -jcnsjdc ", counter)

# try except for debugging - lol

try:
    for intent in intents['intents']:
        

        try:
            for pattern in intent['patterns']:
                
                try:
                    # take each word and tokenize it
                    w = nltk.word_tokenize(pattern)
                    words.extend(w)
                    # adding documents
                    documents.append((w, intent['tag']))
            
                    # adding classes to our class list
                    if intent['tag'] not in classes:
                        classes.append(intent['tag'])
                        
                    counter=counter+1
                    print("counter inside main - is - ", counter)
                    print("Present tag is - ",intent['tag'])
                    print("Present pattern question is - ",pattern)
                    
                except:
                    print("An exception occurred inside for loop main code in counter - ",counter)

        except Exception as error:
            print("An exception occurred in 2nd for loop patterns in counter = ",counter)
except:
  print("An exception occurred in 1st for loop tag part -", counter)




words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


print("Training data created")