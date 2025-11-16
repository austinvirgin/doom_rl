from vizdoom import *

import random

import time

import numpy as np

from gymnasium import Env

from gymnasium.spaces import Discrete, Box

import cv2

from matplotlib import pyplot as plot

# actions = np.identity(3, dtype=np.uint8)

# episodes = 10
# for episode in range(episodes):
#     # starting a new game
#     game.new_episode()

#     # Keeps going until character dies or whatever metric is used
#     while not game.is_episode_finished():
#         state = game.get_state()
#         img = state.screen_buffer
#         info = state.game_variables
#         reward = game.make_action(random.choice(actions))
#         print(f'reward: {reward}')
#         time.sleep(0.02)
    
#     print(f"result: {game.get_total_reward()}")
#     time.sleep(2)

class VizDoomGym(Env):
    # Initalizing function
    def __init__(self, render=False):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config('ViZDoom/scenarios/basic.cfg')

        # Render frame logic
        self.game.set_window_visible(render)

        self.game.init()

        self.observation_space = Box(low = 0, high = 255, shape = (100, 160, 1), dtype=np.uint8)

        self.action_space = Discrete(3)

    # What is done when taking a step
    def step(self, action):
        # Specify action then take step
        actions = np.identity(3)
        reward = self.game.make_action(actions[action], 4)

        if self.game.get_state():
            state = self.game.get_state()
            img = state.screen_buffer
            img = self.grayscale(img)
            info = state.game_variables

        else:
            img = np.zeros(self.observation_space.shape)
            info = 0
        
        done = self.game.is_episode_finished()

        return img, reward, done, info

    # Define how to render the game or environment
    def render():
        pass
    
    # What happens when a new game is started
    def reset(self):
        self.game.new_episode()

        return self.grayscale(self.game.get_state().screen_buffer)

    # Grayscale the game frame and resize it
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state

    # Will be called to close the game
    def close(self):
        self.game.close()

env = VizDoomGym()

state = env.reset()

# print(env.observation_space.sample().shape)

plot.imshow(state)

plot.show()

env.close()