import torch.nn as nn
import torch.nn.functional as functional
from collections import deque
import gym
import random
import torch
import numpy as np
import torch.optim as optim
import time
import pickle


EARLY_STOPPING_THRESHOLD = 80 # stop training after average score of 80 achieved.
INPUT_DIMENSIONS = 4
OUTPUT_DIMENSIONS = 2
MAX_QUEUE_LENGTH = 1000000
EPSILON = 1
EPSILON_DECAY = .996
MIN_EPSILON = .05
EPOCHS =   2000
DISCOUNT_FACTOR = 0.995
TARGET_NETWORK_UPDATE_FREQUENCY = 5000
MINI_BATCH_SIZE = 32
PRETRAINING_LENGTH = 1000





class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIMENSIONS,12)
        self.fc2 = nn.Linear(12,12)
        self.fc3 = nn.Linear (12, OUTPUT_DIMENSIONS)


    def forward(self,x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ExperienceReplayBuffer():

    def __init__(self):
        self.experience_list = deque(maxlen = MAX_QUEUE_LENGTH)

    def sample_mini_batch(self):
    	return random.sample(self.experience_list, MINI_BATCH_SIZE)
        


    def append(self,experience):
        self.experience_list.append(experience)


if __name__ == "__main__":




    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    policy_net = Network()
    target_policy_net = Network()

    target_policy_net.load_state_dict(policy_net.state_dict()) # here we update the target policy network to match the policy network


    env = gym.envs.make("CartPole-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)


    queue = ExperienceReplayBuffer()

    optimizer = optim.Adam(policy_net.parameters(), lr=.001)

    step_counter = 0

    episode_reward_record = deque(maxlen=100)


    for i in range(EPOCHS):	# Epochs - number of unique episodes.
        episode_reward = 0
        done = False
        obs = env.reset()
        
        	
        while not done:
           
           # First, collect experience sample and add to experience replay buffer
           if random.uniform(0, 1) < EPSILON: # Random move
           	action = env.action_space.sample()
           else: # DQN predicts best move
           	with torch.no_grad():
           		prediction = np.array(policy_net(torch.tensor(np.array(obs)).float()))
           		action = np.argmax(prediction)
           new_state, reward, done, _ = env.step(action) # Perform action
           episode_reward += reward # Add reward
           experience_tuple = (obs, action, reward, new_state, done) # New experience tuple to be added
           queue.append(experience_tuple) # Append experience tuple to queue
           #env.render() # Render for visual (optional)
           obs = new_state # Move on to new state
           
           if step_counter >= PRETRAINING_LENGTH: # Must have a certain aount of experience before sampling
                experience = queue.sample_mini_batch()
                
                
                # Play over mini_batch_size
                batch_states = []
                batch_rewards = []
                for j in range(MINI_BATCH_SIZE):
                	current_experience = experience[j] # Contains (s, a, r, s', done)
                	batch_states.append(current_experience[0]) # Add state to  all states
                	with torch.no_grad():
                		current_estimate = policy_net(torch.tensor(np.array(current_experience[0])).float()).detach()
                	if current_experience[4] is True: # If current state is at end
                		current_estimate[current_experience[1]] = current_experience[2] # Set Q[action] = reward
                	else:
                		with torch.no_grad():
                			Q_sp_ap_prediction = np.array(target_policy_net(torch.tensor(np.array(current_experience[3])).float()))
                			action_sp_ap = np.argmax(Q_sp_ap_prediction)
                		current_estimate[current_experience[1]] = current_experience[2] + DISCOUNT_FACTOR*Q_sp_ap_prediction[action_sp_ap]
                	batch_rewards.append(current_estimate)
                
                batch_states_tensor = torch.FloatTensor(batch_states)
                batch_estimate = policy_net(batch_states_tensor)
                batch_rewards_tensor = torch.clone(batch_estimate).detach()
                
                for j in range(MINI_BATCH_SIZE):
                	batch_rewards_tensor[j] = batch_rewards[j]
                	
                
                loss = functional.smooth_l1_loss(batch_estimate,batch_rewards_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

           if step_counter % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                target_policy_net.load_state_dict(policy_net.state_dict()) # here we update the target policy network to match the policy network
           step_counter += 1

        EPSILON = EPSILON * EPSILON_DECAY
        if EPSILON < MIN_EPSILON:
            EPSILON = MIN_EPSILON
        
        episode_reward_record.append(episode_reward)
	
        if i%100 ==0 and i>0:
            last_100_avg = sum(list(episode_reward_record))/100
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(last_100_avg))
            print("EPSILON: " +  str(EPSILON))
            if last_100_avg > EARLY_STOPPING_THRESHOLD:
                break

    
    torch.save(policy_net.state_dict(), "DQN.mdl")
    pickle.dump([EPSILON], open("DQN_DATA.pkl",'wb'))





