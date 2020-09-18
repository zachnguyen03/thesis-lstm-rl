# -*- coding: utf-8 -*-
import numpy as np
import sys


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

# Generate tokens (words) using spacy
def get_tokens(doc_text):
    # This pattern is a modification of the default filter from the
    # Tokenizer() object in keras.preprocessing.text. 
    # It just indicates which patters no skip.
    skip_pattern = '\r\n \n\n \n\n\n!"-#$%&()--.*+,-./:;<=>?@[\\]^_`{|}~\t\n\r '
    
    tokens = [token.text.lower() for token in nlp(doc_text) if token.text not in skip_pattern]
    
   return tokens


def plot():
    hist_df = pd.read_csv('history.csv')
    hist2_df = pd.read_csv('history2.csv')
    plt.plot(np.arange(200), hist_df['loss'][:200], label="128 LSTM")
    plt.plot(np.arange(200), hist2_df, label="256 LSTM")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    hist_csv_file = 'history2.csv' 
    with open(hist_csv_file, mode='w') as f:
        hist2_df.to_csv(f)