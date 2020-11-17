import tensorflow as tf 
import numpy as np 
import gym
from nltk.corpus import words as wrds
from nltk.corpus import wordnet as wn

np.seterr(divide='ignore')

weights_file = './Weights/lstm2-17thmarch-2.hdf5'
model.load_weights(weights_file)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

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

        self.buffer = None #Keep track of the 40 sequence
        self.lstm_output = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    
    def step(self, action):
        reward, done = 0,0
        self.buffer = np.append(self.buffer, [tf.keras.utils.to_categorical(action, num_classes=num_classes)], axis=0)
        self.buffer = self.buffer[1:]
        self.lstm_output = lstm.predict(np.reshape(self.buffer, (1, self.buffer.shape[0], self.buffer.shape[1])))
        
        seq = ''.join([int_to_char[np.argmax(c)] for c in self.buffer])
        words = [word for word in seq.split() if word != '']
        last_wrd = words[len(words)-1]
        if len(last_wrd) == 2:
            reward = 30
        else:
            reward = -1
#            try:
#                last_wrd_synset = wn.synset(last_wrd+'.n.01')
#                if last_wrd_synset.wup_similarity(wn.synset('cake.n.01')) > 0.75:
#                    reward = 10
#            except:
#                reward = -1
#        reward = -1
        
#        print('Last word: ', words[len(words) - 1])
            #d.check(words[len(words)-1]) == True 

        return self.buffer, reward, done, {}

    def reset(self):
        # Reset buffer to 40 first sequence and put them in LSTM
        start = np.random.randint(0, len(dataX) - 1)
        self.buffer = tf.keras.utils.to_categorical(dataX[start], num_classes=num_classes)
        self.lstm_output = lstm.predict(np.reshape(self.buffer, (1, self.buffer.shape[0], self.buffer.shape[1])))
        return self.buffer

    # Render: return the sequences of text being processed
    def render(self):
        return ''.join([int_to_char[np.argmax(c)] for c in self.buffer])

    


lstm = tf.keras.Model(inputs = model.layers[0].input, outputs = model.layers[1].output)

def create_ff_model():
    ff_input = tf.keras.layers.Input(model.layers[2].input_shape[1:])
    ff_model = ff_input
    for layer in model.layers[2:]:
        ff_model = layer(ff_model)
    return tf.keras.models.Model(inputs=ff_input, outputs=ff_model)

ff_model = create_ff_model()

def create_target_model():
    inputs = tf.keras.layers.Input(model.layers[2].input_shape[1:])
    layer1 = tf.keras.layers.Dense(50, activation='relu')(inputs)
    layer2 = tf.keras.layers.Dense(39, activation='softmax')(layer1)
    return tf.keras.models.Model(inputs=inputs, outputs=layer2)
ff_model_target = create_target_model()
ff_model_target.set_weights(ff_model.get_weights())


#Training
model_target = model
#Hyperparameters
gamma = 0.99
loss_function = tf.keras.losses.Huber()
kl_function = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

epsilon_greedy_frames = 1000000
epsilon_random_iters = 0
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
action_history = []
state_history = []
state_next_history = []
lstm_output_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
iters = 0
batch_size = 64

update_after_actions = 40
update_target_net = 1000

env = Environment()
while True:
    state = env.reset()
    episode_reward = 0
    
    for timestep in range(1000):
        iters += 1
        if iters < epsilon_random_iters and epsilon > np.random.rand(1)[0]:
            action = np.random.choice(39)
        else:
            prediction = ff_model(tf.convert_to_tensor(env.lstm_output), training=False)
#            action = np.argmax(prediction)
            action = sample(prediction, 0.2)
        
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)
        
        state_next,reward, done, _ = env.step(action)
        
        episode_reward += reward
        
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        rewards_history.append(reward)
        done_history.append(done)
        lstm_output_history.append(env.lstm_output)
        state = state_next
        
        #Update weights after 10th character outputted
        if iters % update_after_actions == 0 and len(done_history) > batch_size:
#            indices = np.random.choice(range(len(done_history)- batch_size))
#            state_sample = np.array(state_history[indices:indices+batch_size])
#            state_next_sample = np.array(state_next_history[indices:indices+batch_size])
#            lstm_output_sample = np.array(lstm_output_history[indices:indices+batch_size])
#            rewards_sample = rewards_history[indices:indices+batch_size]
#            action_sample = action_history[indices:indices+batch_size]
#            done_sample = tf.convert_to_tensor(
#                np.array(done_history[indices:indices+batch_size])
#            )
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            lstm_output_sample = np.array([lstm_output_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )
            #Build q-table and q-values
            future_rewards = ff_model_target.predict(tf.convert_to_tensor(np.reshape(lstm_output_sample,(batch_size, 128))))
            P_super = np.max(future_rewards, axis=1)
            updated_q_values = rewards_sample + gamma * tf.math.reduce_max(future_rewards, axis=1)
#            updated_q_values = updated_q_values* (1-done_sample) - done_sample
            P_new = softmax(updated_q_values)
            masks = tf.one_hot(action_sample, 39)
            
#            a = gamma * future_rewards
#            for i in range(len(rewards_sample)):
#                a[i] += rewards_sample[i]
#            a_P = np.array([softmax(row) for row in a])
#            a_P_max = np.max(a_P, axis=1)
                
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(ff_model.trainable_variables)
                q_values = ff_model(lstm_output_sample)
#                q_action = tf.math.reduce_sum(tf.math.multiply(np.reshape(q_values, (1, 64,39)), masks), axis=2)
                q_action = tf.math.reduce_sum(tf.math.multiply(tf.math.reduce_max(q_values, axis=1), masks), axis=1)
                alpha = 0.05
                loss = loss_function(updated_q_values, q_action) + alpha*0.5*(kl_function(P_new, P_super) + kl_function(P_super, P_new))  
            
            grads = tape.gradient(loss, ff_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, ff_model.trainable_variables))
        
        if iters % update_target_net == 0:
            ff_model_target.set_weights(ff_model.get_weights())
            
            print('running reward: {:.2f} at episode {}, iter {}'.format(running_reward, episode_count, iters))
            
        if len(rewards_history) > 100000:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]
            del lstm_output_history[:1]
        
        if done == 1:
            break
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    episode_count += 1
    running_reward = np.mean(episode_reward_history)
    print('Episode Reward: ', episode_reward)
    print('Running reward: ', running_reward)
    print('Loss: ', loss)
#    if running_reward > 0:
#        print('Updated at episode {}'.format(episode_count))
#        break
    # Stop if training episodes count 100 
    if episode_count == 300:
        print('Trained for 100 episodes')
        break
    
#plot
plt.plot(episode_reward_history)
plt.xlabel('Episode')
plt.ylabel('Episode reward')
plt.title('RL training')
plt.show()

