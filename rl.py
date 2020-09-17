import tensorflow as tf 
import numpy as np 
import enchant
import gym

seed = 42
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 64
max_steps_per_episode = 10000


# Import Baseline Atari environment



def network():
    input = tf.keras.layers.input(shape=(40,1,))
    # Define LSTM layers
    layer1 = tf.keras.layers.LSTM(256, return_sequences=False)(input)
    layer2 = tf.keras.layers.LSTM(256, return_sequences=False)(layer1)
    fclayer1 = tf.keras.layers.Dense(50, activation='relu')(layer2)
    fclayer2 = tf.keras.layers.Dense(39, activation='softmax')(fclayer1)

    return tf.keras.models.Sequential(input=input, output=layer2), tf.keras.models.Sequential(input=fclayer1, output=fclayer2)

lstm, ff = network()

filepath = "./Weights/lstm2-17thmarch.hdf5"
# def FC_network():
#     input = tf.keras.layers.input(shape=(40,1,))

#     fclayer1 = tf.keras.layers.Dense(50, activation='relu')
#     fclayer2 = tf.keras.layers.Dense(39, activation='softmax')

#     return tf.keras.models.Sequential(input=lstm.output, output=fclayer2)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_function = tf.keras.losses.Huber()


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
        self.observation_space = gym.spaces.Box(40)
        self.action_space = gym.spaces.Discrete(39)

        self.seed()
        self.state = None

        self.buffer = [] #Keep track of the 40 sequence
        self.lstm_output = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    
    def step(self, action):
        reward, done = 0,0

        np.append(self.buffer, action)
        self.buffer = self.buffer[1:]

        seq = ''.join(self.buffer)
        words = [word for word in seq.split() if word != '']
        # if words[len(words)-1] is English reward += .1
        d = enchant.Dict('en_US')
        if d.check(words[len(words)-1]) == True:
            reward += 1
        else:
            reward -= 1
        

        return reward, state, done


    def sample_action(self):


    def reset(self):
        # Reset buffer to 40 first sequence and put them in LSTM
        self.buffer = []
        self.lstm_output = lstm.predict(self.buffer)

    # Render: return the sequences of text being processed
    def render(self):
        return ''.join(self.buffer)

    


env = gym.make(Environment)
env.reset()

for i in range(1000): #trials
    # env.reset() # LSTM
    for j in range(50): #iterations
        env.render() #FF
        env.step(env.action_space.sample()) # step