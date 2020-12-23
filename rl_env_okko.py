import tensorflow as tf 
import numpy as np 
import gym
from nltk.corpus import words as wrds
from nltk.corpus import wordnet as wn

np.seterr(divide='ignore')

#weights_file = './Weights/weight_lstm2-okko2.hdf5'
#model.load_weights(weights_file)
#model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001))

weights_file = './Weights/lstm2-17thmarch-2.hdf5'
model.load_weights(weights_file)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}

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

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    
    def step(self, action):
        reward, done = 0,0
        self.buffer = np.append(self.buffer, [tf.keras.utils.to_categorical(action, num_classes=num_classes)], axis=0)
        self.buffer = self.buffer[1:]
        
        seq = ''.join([int_to_char[np.argmax(c)] for c in self.buffer])
        words = [word for word in seq.split() if word != '']
        last_wrd = words[len(words)-1]
        sec_last_wrd = words[len(words)-2]
#        food_terms = ['food', 'cake', 'salt', 'sugar', 'said']
#        reward = -1*abs((char_to_int[seq[-1]] - char_to_int[seq[-2]])) #char_to_int is dict mapping character to action number (index)
#        if len(last_wrd) == 3:
#            reward = 10
#        else:
#            reward = -1
        if len(words) > 6 and last_wrd in wrds.words() and sec_last_wrd != last_wrd and last_wrd != 'the':
#        if len(words) > 7 and len(last_wrd) == 3:
            reward = 10
        else:
            reward = -1
#        if len(words) > 6 and last_wrd in wrds.words() and sec_last_wrd != last_wrd and last_wrd in nouns:
#            try:
#                last_wrd_synset = wn.synset(last_wrd+'.n.01')
#                if last_wrd_synset.wup_similarity(wn.synset('cake.n.01')) > 0.6:
#                    reward = 10
#            except:
#                reward = -1
#        else:
#            reward = -1
#        if last_wrd in wrds.words():
#            reward = 7
#        else:
#            reward = 0
            
        
        return self.buffer, reward, done, {}

    def reset(self):
        # Reset buffer to 40 first sequence and put them in LSTM
        start = np.random.randint(0, len(dataX) - 1)
        self.buffer = tf.keras.utils.to_categorical(dataX[start], num_classes=num_classes)
        return self.buffer

    # Render: return the sequences of text being processed
    def render(self):
        return ''.join([int_to_char[np.argmax(c)] for c in self.buffer])


#Training
model_target = Model()
model_ref = Model()
#Hyperparameters
gamma = 0.99 
alpha = 100000 #alpha for Jensen-SHannon loss
loss_function = tf.keras.losses.CategoricalCrossentropy()
kl_function = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)


action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
js_history = []
aux_loss = []
kl_loss = []
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
        prediction = model(np.reshape(env.buffer, (1, env.buffer.shape[0], env.buffer.shape[1])), training=False)
        action = sample(prediction, 0.2)
        
        state_next,reward, done, _ = env.step(action)
        
        episode_reward += reward
        
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        rewards_history.append(reward)
        done_history.append(done)
        state = state_next
        
        #Update weights after 10th character outputted
        if iters % update_after_actions == 0 and len(done_history) > batch_size:
            indices = np.random.choice(range(len(done_history)- batch_size))
            state_sample = np.array(state_history[indices:indices+batch_size])
            state_next_sample = np.array(state_next_history[indices:indices+batch_size])
            rewards_sample = rewards_history[indices:indices+batch_size]
            action_sample = action_history[indices:indices+batch_size]
            done_sample = tf.convert_to_tensor(
                np.array(done_history[indices:indices+batch_size])
            )
#            indices = np.random.choice(range(len(done_history)), size=batch_size)
#
#            # Using list comprehension to sample from replay buffer
#            state_sample = np.array([state_history[i] for i in indices])
#            state_next_sample = np.array([state_next_history[i] for i in indices])
#            rewards_sample = [rewards_history[i] for i in indices]
#            action_sample = [action_history[i] for i in indices]
#            done_sample = tf.convert_to_tensor(
#                [float(done_history[i]) for i in indices]
#            )
            #Build q-table and q-values
            future_rewards = model_target.predict(tf.convert_to_tensor(state_sample))
            P_super = future_rewards
            P_ref = model_ref.predict(tf.convert_to_tensor(state_sample))
            updated_q_values = rewards_sample + gamma * tf.math.reduce_max(future_rewards, axis=1)
#            P_new = softmax(updated_q_values)
            masks = tf.one_hot(action_sample, 39)
            
            #P_new
            a = gamma * future_rewards
            for i in range(len(rewards_sample)):
                a[i] += rewards_sample[i]
            P_new = np.array([softmax(row) for row in a])
            
            # Calculating the KL Divergence of P_super and P_new 
            M = 0.5*(P_ref+P_super)
            
            kl = np.mean([kl_function(P_ref[i], M[i]) for i in range(64)])
            kl_rev = np.mean([kl_function(P_super[i], M[i]) for i in range(64)])
                
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(model.trainable_variables)
                q_values = model(state_sample)
#                q_action = tf.math.reduce_sum(tf.math.multiply(np.reshape(q_values, (1, 64,39)), masks), axis=2)
                q_action = tf.math.reduce_sum(tf.math.multiply(q_values, masks), axis=1)
                loss = loss_function(updated_q_values, q_action) + alpha*0.5*(kl+kl_rev)
#                loss = loss_function(updated_q_values, q_action)
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if iters % update_target_net == 0:
            model_target.set_weights(model.get_weights())
            
            print('running reward: {:.2f} at episode {}, iter {}'.format(running_reward, episode_count, iters))
            
        if len(rewards_history) > 100000:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]
        
        if done == 1:
            break
    episode_reward_history.append(episode_reward)
    episode_count += 1
    running_reward = np.mean(episode_reward_history)
    print('Episode Reward: ', episode_reward)
    print('Running reward: ', running_reward)
    print('Loss: ', loss)
    print('Jensen-Shannon divergence: ', 0.5*(kl + kl_rev))
    aux_loss.append(loss)
    kl_loss.append(0.5*(kl + kl_rev))
    # Stop if training episodes count 100 
    if episode_count == 100:
        print('Trained for 100 episodes')
        break
    
# Plot episodes and loss curve
plt.plot(episode_reward_history)
plt.ylim(min(episode_reward_history), max(episode_reward_history))
plt.xlabel('Episode')
plt.ylabel('Episode reward')
plt.title('RL training')
plt.show()


aux, = plt.plot(aux_loss)
kl, = plt.plot(kl_loss)
plt.legend([aux, kl], ['Total loss', 'JS Loss'], loc='lower right')
plt.xlabel('Episode')
plt.ylabel('Episode loss')
plt.title('RL training')
plt.show()





