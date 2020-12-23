"""
Created on Fri Dec 18 13:35:51 2020

@author: MYCoskun

https://github.com/pythonlessons/Reinforcement_Learning/blob/master/01_CartPole-reinforcement-learning/Cartpole_DQN.py
"""

import numpy as np
import gym

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'DFCU_EHS-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
import gym_DFCU_EHS
# env = gym.make('DFCU_EHS-v0')

import gym.spaces

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt


# env = gym.make('CartPole-v1')

# env.reset()
# env.step(np.array([1, 1, 1, 1,3]),3)

# print(env.action_space.n)

# %% 
def OurModel(input_shape,action_space):
    X_input = Input(input_shape)
    
    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu",
              kernel_initializer='he_uniform')(X_input)
    
    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    
    # Output Layer with # of actions: 49 nodes
    X = Dense(action_space, activation="softmax",
              kernel_initializer='he_uniform')(X)
    
    model = Model(inputs = X_input, outputs = X, name = "dfcu")
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
                  metrics=["accuracy"])
    
    model.summary()
    return model
# %%
class DQNAgent:
    def __init__(self):
        self.env = gym.make('DFCU_EHS-v0')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000
        
        self.tau = 1e-4 # sample time
        self.time = 0
        self.simTime = 0.001
        self.env._max_episode_steps = (self.simTime/self.tau)+1
        self.fig, self.axs = plt.subplots(2)
        
        # create main model
        self.model = OurModel(input_shape=(self.state_size,),
                              action_space = self.action_size)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
    def act(self,state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))
        
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []
        
        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])
            
        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        
        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
        
    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)
            
    def run(self):
        for e in range(self.EPISODES):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            self.time = 0
            # plt.cla()
            self.axs[1].clear
            self.axs[0].clear
            while not done:
                # self.env.render()
                self.time += self.tau
                self.axs[0].scatter(self.time,state[0,0])
                self.axs[1].scatter(self.time,state[0,3])
                plt.pause(self.tau)
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action,0.23)
                print(next_state[0], next_state[2], reward, done)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1: # 100
                    reward = reward
                    # print(reward)
                else:
                    reward = -500
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:                   
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as dfcuEHS-dqn.h5")
                        self.save("dfcuEHS-dqn.h5")
                        return
                self.replay()

    def test(self):
        self.load("dfcuEHS-dqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break
# %%
if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
    # agent.test()
