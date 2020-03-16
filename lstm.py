import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
#import spacy
import re

# line = re.sub('[!@#$]', '', line)

#nlp = spacy.load('en_core_web_md', disable=['parser', 'tagger', 'ner'])
# Read file
filename = 'wonderland.txt'
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()


# List appeared characters in the text
chars = sorted(list(set(raw_text)))

# Map integer to character and vice versa
int_to_char = dict((i,c) for i,c in enumerate(chars))
char_to_int = dict((c,i) for i,c in enumerate(chars))

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


# Data transformation for Keras
X = np.reshape(dataX, [n_patterns, seq_length, 1])
X = X / float(n_vocab)

X = tf.keras.utils.to_categorical(dataX)
y = tf.keras.utils.to_categorical(dataY)


# Define the model function
def Model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
#    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.LSTM(256, return_sequences = False))
#    model.add(tf.keras.layers.Dropout(0.25))
#    model.add(tf.keras.layers.LSTM(256, return_sequences = False))
#    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(y.shape[1], activation="softmax"))
    
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001))
    
    return model


# Create model
model = Model()


# Define hyperparameters and callbacks
filepath = "lstm2-15thmarch.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


	
history = model.fit(X, y, epochs=200, batch_size=64, callbacks=callbacks_list)


weights_file = 'lstm2-15thmarch.hdf5'
model.load_weights(weights_file)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001))
    
# Generate Text using custom Text Generation function in helper.py
text_generation(1000, 0.02)