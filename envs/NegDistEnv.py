import numpy as np
import gym


class Environment(gym.Env):
    '''
    Description: Environment for text generation

    Observation:
        Type: Box(40)
        1-D Numpy array of sequence of 40 characters

    Action: 
        Type: Discrete(39)
        Possible outputted characters (A-Z, 0-9, whitespace)
    
    Reward:
        Negative distance of the last 2 outputted characters. Distance is defined as the difference between 2
        action numbers

    '''
    def __init__(self): 
        self.seed()
        self.state = None


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    
    def step(self, action):
        reward, done = 0,0
        self.state = np.append(self.state, [tf.keras.utils.to_categorical(action, num_classes=num_classes)], axis=0)
        self.state = self.state[1:]
        seq = ''.join([int_to_char[np.argmax(c)] for c in self.state])
        
        reward = -1*abs((char_to_int[seq[-1]] - char_to_int[seq[-2]]))         
        
        return self.state, reward, done, {}

    def reset(self):
        # Reset buffer to 40 first sequence and put them in LSTM
        start = np.random.randint(0, len(dataX) - 1)
        self.state = tf.keras.utils.to_categorical(dataX[start], num_classes=num_classes)
        return self.state

    # Render: return the sequences of text being processed
    def render(self):
        return ''.join([int_to_char[np.argmax(c)] for c in self.state])