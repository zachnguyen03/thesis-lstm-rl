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
        5 if the action is 18, otherwise 0

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
        
        if action == 18:
            reward = 5
        else:
            reward = 0          
        
        return self.state, reward, done, {}

    def reset(self):
        # Reset buffer to 40 first sequence and put them in LSTM
        start = np.random.randint(0, len(dataX) - 1)
        self.state = tf.keras.utils.to_categorical(dataX[start], num_classes=num_classes)
        return self.state

    # Render: return the sequences of text being processed
    def render(self):
        return ''.join([int_to_char[np.argmax(c)] for c in self.state])