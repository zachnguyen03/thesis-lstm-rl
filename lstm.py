import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import re
import matplotlib.pyplot as plt
# Read file
filename = './Dataset/wonderland.txt'
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# Extract symbols from text
raw_text = re.sub(r'^a-zA-z0-9.,', ' ', raw_text)


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
    model.add(tf.keras.layers.LSTM(256, return_sequences = False))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(y.shape[1], activation="softmax"))
    
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001))
    
    return model


# Create model
model = Model()


# Define hyperparameters and callbacks
filepath = "./Weights/lstm2-17thmarch.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# Train model and save loss to history
history2 = model.fit(X, y, epochs=200, batch_size=64, callbacks=callbacks_list)

# Load weight file and recompile model
weights_file = './Weights/lstm2-17thmarch.hdf5'
model.load_weights(weights_file)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001))
    
# Generate Text using custom Text Generation function
def text_generation(length, diversity):
    # Pick random seed from input text
    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    
    # Loop through each pattern and predict using LSTM Model and print out the results
    for i in range(length):
        patternX = tf.keras.utils.to_categorical(pattern, num_classes=32)
        patternX = np.reshape(patternX, (1, patternX.shape[0], patternX.shape[1]))
        prediction = model.predict(patternX, verbose=0)
#        index = np.argmax(prediction)
        index = sample(prediction, diversity)
        result = int_to_char[index] 
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        
        

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds.T, 1)
    return np.argmax(probas)

text_generation(1000, 0.1)