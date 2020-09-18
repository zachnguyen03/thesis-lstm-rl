import tensorflow as tf 
import numpy as np 
import enchant
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
        1 for valid English word, 0 for invalid English word. Check
        the latest word of the sequence

    '''
    def __init__(self): 
        
        # Metadata
        self.observation_space = gym.spaces.Box(np.zeros(40), np.ones(40))
        self.action_space = gym.spaces.Discrete(39)

        self.seed()
        self.state = None

        start = np.random.randint(0, len(dataX)-1)
        self.buffer = tf.keras.utils.to_categorical(dataX[start], num_classes=num_classes) #Keep track of the 40 sequence
        self.lstm_output = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    
    def step(self, action):
        reward, done = 0,0

        self.buffer = np.append(self.buffer, [tf.keras.utils.to_categorical(action, num_classes=num_classes)], axis=0)
        self.buffer = self.buffer[1:]
        
        seq = ''.join([int_to_char[np.argmax(c)] for c in self.buffer])
        words = [word for word in seq.split() if word != '']
        d = enchant.Dict('en_US')
        if d.check(words[len(words)-1]) == True and len(words[len(words)-1]) > 1:
            reward = 1
        elif len(words[len(words)-1]) == 1:
            reward = 0
        else:
            reward = -1
        
#        print('Last word: ', words[len(words) - 1])

        return self.buffer, reward, done, {}

    def reset(self):
        # Reset buffer to 40 first sequence and put them in LSTM
        start = np.random.randint(0, len(dataX) - 1)
        self.buffer = tf.keras.utils.to_categorical(dataX[start], num_classes=num_classes)

    # Render: return the sequences of text being processed
    def render(self):
        return ''.join([int_to_char[np.argmax(c)] for c in self.buffer])

    
def training():
    env = Environment()
    env.reset()
    episode_reward = 0
    for i in range(1000):
        prediction = model.predict(np.reshape(env.buffer, (1, env.buffer.shape[0], env.buffer.shape[1])))
        action = np.argmax(prediction)
        state, reward, done, _ = env.step(action)
        print(env.render())
        episode_reward += reward
    print(episode_reward)
    env.close()
