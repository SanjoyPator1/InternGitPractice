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
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD



words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json', encoding="utf8").read()
intents = json.loads(data_file)


def count():
    counter
    

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
        print('The intent is - ', intents)

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
                    print("Present document list is - ",documents)
                    print("Present classes list is - ",classes)
                    
                except:
                    print("An exception occurred inside for loop main code in counter - ",counter)

        except Exception as error:
            print("An exception occurred in 2nd for loop patterns in counter = ",counter)
except:
  print("An exception occurred in 1st for loop tag part -", counter)




words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print("")
print (len(documents), "documents", documents)
print("")

print (len(classes), "classes", classes)
print("")

print (len(words), "unique lemmatized words", words)
print("")


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


print("Training data created")


#INITIALIZING THE TRAINING DATA

training = []
output_empty = [0] * len(classes)

print('Output_empty is - ', output_empty)

for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    
    print("")
    print("NEXT")
    print("present pattern_words is - ", pattern_words)
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        
    print("present bag is - ", bag)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    print("present output_row is - ", output_row)
    
    training.append([bag, output_row])
    print("present training is - ", training)
    
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

#print("training in np array is - ", training)

# create train_x and train_y  lists. X - patterns, Y - Labels and features
train_x = list(training[:,0])
print("train_x", train_x)

train_y = list(training[:,1])
print("train_y", train_y)

print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),kernel_initializer='he_uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='he_uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), kernel_initializer='he_uniform', activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), validation_split=0.33, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")
