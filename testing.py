import gym
import numpy as np
import keras
import random
from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import time
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
GAME = 'bird'
max_ep=1000
train = True


class dqnagent():
    def __init__(self, lr = 0.00025, ob_size = (84,84,4), action_size = 4):
        self.state_size = (84,84,4)
        self.action_size = 2
        self.model = self._build_model(lr)
        self.game_state = game.GameState()

    def choose_action(self,s):
        a = self.model.predict(s)
        return np.argmax(a[0])

    def _build_model(self,lr):
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        shape_image=(84,84,4)
        model = Sequential()
        model.add(keras.layers.Lambda(lambda x: x / 255.0,input_shape = shape_image))
        model.add(Conv2D(32,(8,8), strides=4,use_bias =True,bias_initializer='zeros',kernel_initializer = init,activation = 'relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(64,(4,4), strides = 2, use_bias = True, bias_initializer = 'zeros',kernel_initializer = init, activation='relu'))
        model.add(Conv2D(64,(3,3),use_bias= True, bias_initializer = 'zeros', kernel_initializer = init, activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer='he_uniform' ))
        model.add(Dense(24, activation = 'relu',kernel_initializer='he_uniform' ))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer = 'he_uniform'))
        model.compile(optimizer=keras.optimizers.RMSprop(lr,rho=0.95), loss = 'mse')
        return model

    def env_re(self):
        s,r,d = self.game_state.frame_step([0,1])
        return s[40:,0:407,:]

    def step(self,a):
        return self.game_state.frame_step(a)

    def model_load(self):
        self.model.load_weights("model_flappy_dqn.h5")

record = []
brain = dqnagent()
st = time.time()

model_saved = True
if model_saved == True:
    brain.model_load()

if train == True:
    for i in range(100):
        s = brain.env_re()
        s =cv2.resize(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY),(84,84))
        s = np.reshape(s,(1,84,84))
        s = [s for _ in range(4)]
        s = np.stack(s,axis=3)
        s = np.array(s)
        d = False
        R = 0
        while not d:
            a = brain.choose_action(s)
            sts= [0,0]
            sts[a] = 1
            s2, r, d = brain.step(sts)
            s2=s2[40:,0:407,:]
            s2 =cv2.resize(cv2.cvtColor(s2, cv2.COLOR_RGB2GRAY),(84,84))
            s2= np.reshape(s2, (1,84,84,1) )
            s2=np.concatenate((s2,s[:,:,:,0:3]),axis=3)
            if r ==1:
                R+=r
            s = s2
            if d == True:
                record.append(R)
                print(i, R)
                break
record = np.array(record)
plt.plot(record)
plt.xlabel('no of episode')
plt.ylabel('score')
plt.show()
