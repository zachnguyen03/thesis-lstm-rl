import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import re
import matplotlib.pyplot as plt
import pandas as pd


# Read file
filename = './Dataset/wonderland.txt'
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
raw_text = re.sub(r'[^a-z0-9.,]', ' ', raw_text)

#Stopwords removal
import nltk
nltk.download('stopwords')
nltk.download('word_tokenize')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(raw_text)
raw_text = ' '.join([w for w in word_tokens if w not in stop_words or w != ' '])

# List appeared characters in the text
chars = sorted(list(set(raw_text)))

# Map integer to character and vice versa
int_to_char = dict((i,c) for i,c in enumerate(chars))
char_to_int = dict((c,i) for i,c in enumerate(chars))
num_classes = len(int_to_char)
n_chars = len(raw_text)
n_vocab = len(chars)

# Calculate patterns in the text
seq_length = 40
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total patterns: ", n_patterns)

# Number of units in specific layers
num_lstm_units = 128    
num_ff_units = 50


# Data transformation for Keras
#X = np.reshape(dataX, [n_patterns, seq_length, 1])
#X = X / float(n_vocab)

X = tf.keras.utils.to_categorical(dataX, num_classes=num_classes)
y = tf.keras.utils.to_categorical(dataY, num_classes=num_classes)


# Define the model function
def Model():
    model = tf.keras.models.Sequential()
    #model.add(tf.keras.layers.GRU(128, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
    model.add(tf.keras.layers.LSTM(num_lstm_units, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
#    model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.GRU(128, return_sequences = False))
    model.add(tf.keras.layers.LSTM(num_lstm_units, return_sequences = False))
#    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(num_ff_units, activation='relu'))
#    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(y.shape[1], activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001))
    return model


# Create model
model = Model()
model.summary()

# Define hyperparameters and callbacks
#filepath = "./Weights/weight_train.hdf5"
#checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]
# Train model and save loss to history and save weights to file
#history6 = model.fit(X, y, epochs=100, batch_size=256, callbacks=callbacks_list)


# Load weight file and recompile model
weights_file = './Weights/lstm2-17thmarch-2.hdf5'
model.load_weights(weights_file)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001))
    


# Generate Text using custom Text Generation function
def text_generation(length, diversity):
    # Pick random seed from input text
    outputted_text = ''
    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    # Loop through each pattern and predict using LSTM Model and print out the results
    for i in range(length):
        patternX = tf.keras.utils.to_categorical(pattern, num_classes=num_classes)
        patternX = np.reshape(patternX, (1, patternX.shape[0], patternX.shape[1]))
        prediction = model.predict(patternX, verbose=0)
#        index = np.argmax(prediction)
        index = sample(prediction, diversity)
        result = int_to_char[index] 
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        outputted_text += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return outputted_text
        

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds.T, 1)
    outputted_text = np.argmax(probas)
    return outputted_text
   
text_generation(1000, 0.2)




# Save model
#model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model


### returns a compiled model
### identical to the previous one
#model = tf.keras.models.load_model('my_model.h5')
