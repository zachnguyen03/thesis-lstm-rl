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
        d = enchant.Dict('en_US')
        if d.check(words[len(words)-1]) == True:
            reward = 10
        else:
            reward = -1
        
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

    
#def training():
#    env = Environment()
#    env.reset()
#    episode_reward = 0
#    for i in range(1000):
#        prediction = ff_model.predict(env.lstm_output)
#        action = np.argmax(prediction)
#        state, reward, done, _ = env.step(action)
#        print(env.render())
#        episode_reward += reward
#    print(episode_reward)
#    env.close()



lstm = tf.keras.Model(inputs = model.layers[0].input, outputs = model.layers[1].output)
#ff = model(inputs=model.layers[2].input, outputs = model.layers[3].output)

ff_input = tf.keras.layers.Input(model.layers[2].input_shape[1:])
ff_model = ff_input
for layer in model.layers[2:]:
    ff_model = layer(ff_model)
ff_model = tf.keras.models.Model(inputs=ff_input, outputs=ff_model)
ff_model_target = ff_model


#input_layer = tf.keras.layers.Input(shape=(X.shape[1], X.shape[2]))
#lstm_1 = tf.keras.layers.LSTM(128,return_sequences=True)(input_layer)
#lstm_2 = tf.keras.layers.LSTM(128,return_sequences=False)(lstm_1)
#ff_1 = tf.keras.layers.Dense(50, activation='relu')(lstm_2)
#ff_2 = tf.keras.layers.Dense(y.shape[1], activation='softmax')(ff_1)
#model_test = tf.keras.models.Model(input_layer, ff_2)
#model_test.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001))

#Training
model_target = model
#Hyperparameters
gamma = 0.995
loss_function = tf.keras.losses.Huber() 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0)

epsilon_greedy_frames = 1000000
epsilon_random_iters = 5000
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
batch_size = 128

update_after_actions = 4
update_target_net = 1000

env = Environment()
while True:
    state = env.reset()
    episode_reward = 0
    
    for timestep in range(1000):
        iters += 1
        if iters < epsilon_random_iters or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(39)
        else:
            prediction = ff_model(tf.convert_to_tensor(env.lstm_output), training=False)
            action = np.argmax(prediction)
        
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
            indices = np.random.choice(range(len(done_history)), size=batch_size)
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            lstm_output_sample = np.array([lstm_output_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )
            
            #Build q-table and q-values
            future_rewards = ff_model_target.predict(np.reshape(lstm_output_sample,(batch_size, 128)))
            
            updated_q_values = rewards_sample + gamma * tf.math.reduce_max(future_rewards, axis=1)
            updated_q_values = updated_q_values* (1-done_sample) - done_sample
            
            masks = tf.one_hot(action_sample, 39)
            
            with tf.GradientTape() as tape:
                tape.watch(ff_model.trainable_variables)
                q_values = ff_model(lstm_output_sample)
                q_action = tf.math.reduce_sum(tf.math.multiply(tf.math.reduce_max(q_values, axis=1), masks), axis=1)
                loss = loss_function(updated_q_values, q_action)
            
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
    if running_reward > 0:
        print('Updated at episode {}'.format(episode_count))
        break
    
    
# 0 -769 100 199 102 342 
# 0 -769 100 -713 158 7.16 227 1000