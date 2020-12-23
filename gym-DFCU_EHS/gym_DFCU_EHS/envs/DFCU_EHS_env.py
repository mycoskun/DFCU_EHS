# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 20:21:34 2020

@author: MYCoskun
"""

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt


class DFCU_EHSEnv(gym.Env):
    """
    Description:
        
    Source:
        
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.Aa = 0.00125663; # Piston side area [m^2]
        self.Ab = 0.00094247; # Rod side area [m^2]
        self.Ba = 1500e6; # Piston side bulk Moduli [Pa]
        self.Bb = 1500e6; # Rod side bulk Moduli [Pa]
        self.V0A = 0.001; # Piston side dead-volume [m^3]
        self.V0B = 0.001; # Rod side dead-volume [m^3]
        self.xMin = 0; # Min. piston stroke [m]
        self.xMax = 2; # 0.5->2 | Max. piston stroke [m] 
        self.b = 0.3; # Critical pressure ratio [-]
        self.ex = 0.53; # Exponent of valve model [-]
        self.Kv = 6.3387e-09; # 1.3 lpm @ 3.5 MPa
        self.Ps = 10e6+101325; # Supply pressure [Pa]
        self.Pt = 101325; # 1 atm Tank pressure [Pa]
        self.addMass = 250 # Additional mass [Kg]
        self.mPiston = 8; # Piston mass [Kg]
        self.m = self.addMass + self.mPiston # Total mass [Kg]
        self.tau = 0.0001  # seconds between state updates
        self.FmassLoad = 1*9.805*self.addMass # The MASS force exerted on the 
                                      # piston in the direction of gravity (N)
        self.Fload = 200 + self.FmassLoad; # The FORCE and MASS force exerted 
                               # on the piston in the direction of gravity (N)

        # Position at which to fail the episode
        self.pos_range = np.array([0, self.xMax])

        # Observation space limit
        low = np.array([self.xMin,
                        np.finfo(np.float32).min,
                        -self.xMax,
                        np.finfo(np.float32).min,
                        -1e6,
                        -1e6],
                       dtype=np.float32)
        
        high = np.array([self.xMax,
                         np.finfo(np.float32).max,
                         self.xMax,
                         np.finfo(np.float32).max,
                         self.Ps,
                         self.Ps],
                        dtype=np.float32)


        self.action_space = spaces.Discrete(50)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.t = 0
        # Pa = 0
        # Pb = 0
        # self.fig, self.ax = plt.subplots()
        # plt.show()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action,ref):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg
       
        def flow_vector(P_1,P_2,Kv,b,x):
            # Based on Linjama, M., Huova, M., and Karvonen, M., 2012.
            # Modelling of flow characteristics of on/off valves.
            # P = Pressure [Pa]
            # Q = Flowrate [m^3/s]
            # Kv = Vector of flow coefficient of valve model of DFCU [m^3 s^-1 Pa^x]
            # b = Vector of critical pressure ratios of valve model of DFCU [-]
            # x = Vector of exponent of valve model of DFCU [-]
            
            # Define valve count at each DFCU
            # valCount = Kv.size
            valCount = 1
            Q = np.zeros((valCount,1))
            
            # % Check b and x vector or scalar,
            if valCount != 1:
                if b.size == 1:
                    b_ = b*np.ones((valCount,1))
                    x_ = x*np.ones((valCount,1))
            else:
                b_ = b
                x_ = x
                    
                # Assume that parameters are identical for all valves,
                Kv1 = Kv; Kv2 = Kv;
                b1 = b_; b2 = b_;
                x1 = x_; x2 = x_;
                    
                for i in range(valCount):
                    if (b1*P_1 < P_2) and (P_2 <= P_1):
                        Q[i] = Kv1*(P_1-P_2)**x1
                    elif  P_2 <= b1*P_1:
                        Q[i] = Kv1*((1-b1)*P_1)**x1
                    elif (b2*P_2 < P_1) and (P_1 <P_2):
                        Q[i] = -Kv2*(P_2 - P_1)**x2
                    elif P_1 <= b2*P_2:
                        Q[i] = -Kv2*((1-b2)*P_2)**x2
                Qsum = Q.sum()
                                        
                return Qsum
            
        x, x_dot, error, e_dot, Pa, Pb = self.state

        def valve_seq(action):
            valve = np.arange(0,5)

            c = np.array(np.meshgrid(valve,valve)).T.reshape(-1,2)
            d = np.zeros((len(c),2))

            action_1 = np.concatenate((c[:,0].reshape(-1,1),d[:,0].reshape(-1,1),
                               d[:,0].reshape(-1,1),c[:,1].reshape(-1,1)), 
                              axis=1)

            action_2 = np.concatenate((d[:,0].reshape(-1,1),c[:,0].reshape(-1,1),
                               c[:,1].reshape(-1,1),d[:,0].reshape(-1,1)),
                              axis=1)

            action_space = np.vstack((action_1,action_2))
            
            return action_space[action-1,:]
            
            
        valve_decode_seq = valve_seq(action)
        cPA = valve_decode_seq[0]
        cAT = valve_decode_seq[1]
        cPB = valve_decode_seq[2]
        cBT = valve_decode_seq[3]
        
        # Friction Force
        Ffric = np.tanh(2000*x_dot) * (280+70*np.exp(-1e4*(x_dot**2))) + 100*x_dot
        
        # if self.Pa is None:
        #     Pa = 0
        # if self.Pb is None:
        #     Pb = 0
        
        
        # Flow [m^3/s]
        QPA = flow_vector(self.Ps,Pa,cPA*self.Kv,self.b,self.ex)
        QAT = flow_vector(Pa,self.Pt,cAT*self.Kv,self.b,self.ex)
        QPB = flow_vector(self.Ps,Pb,cPB*self.Kv,self.b,self.ex)
        QBT = flow_vector(Pb,self.Pt,cBT*self.Kv,self.b,self.ex)

        # Semi implicit euler
        x_2dot = (Pa*self.Aa-Pb*self.Ab-Ffric-self.Fload)/self.m # Accel. [m/s^2]
        x_dot = x_dot + self.tau * x_2dot # Velocity [m/s]
        x = x + self.tau * x_dot # Position [m]
        Pad_1 = (self.Ba*(QPA-QAT-self.Aa*x_dot));
        Pad_2 = x*self.Aa+self.V0A;
        Pa_dot = Pad_1/Pad_2; # Piston side pressure differential [Pa/s]
        Pa = Pa + self.tau * Pa_dot # Piston side pressure [Pa]
        Pbd_1 = (self.Bb*(QPB-QBT+self.Ab*x_dot));
        Pbd_2 = (self.xMax-x)*self.Ab+self.V0B;
        Pb_dot = Pbd_1/Pbd_2; # Rod side pressure differential [Pa/s]
        Pb = Pb + self.tau * Pb_dot # Rod side pressure [Pa]

        error = ref - x # Error
        error = round(error,5)

        # Termination criteria, isDone
        done = bool(
            x < self.xMin
            or x > self.xMax
            or np.abs(error) > 0.01
        )
 

        
        # # self.ePrev = None
        # if self.ePrev is None:
        #     self.ePrev = 0
        # else:
        #     self.ePrev = error
        
        try:
            e_dot = (abs(error)-abs(self.ePrev))/self.tau
        except:
            e_dot = (abs(error)-0)/self.tau
            
        self.ePrev = error
        
        x = round(x,5)
        x_dot = round(x_dot,5)
        error = round(error,5)
        e_dot = round(e_dot,5)
        Pa = round(Pa,5)
        Pb = round(Pb,5)
        
        self.state = (x, x_dot, error, e_dot, Pa, Pb)
        
        # Define reward
        # coef = 1;
        if error < 1e-3:
            error = 1e-3
        
        r1 = 1e-4/error**2
        if valve_decode_seq.sum() == 0:
            r2 = 1
        else:
            r2 = 1/valve_decode_seq.sum()**2
        reward = r1+r2
        
        # a = 0.005-np.tanh(abs(e));  
        # if a > 0:
        #     r1 = 50*a*coef;
        # else:
        #     r1 = a*coef;
                    
        # if valve_decode_seq.sum() > 0:
        #     r2 = -0.01*(1-(1/valve_decode_seq.sum()**2))*coef;
        # else:
        #     r2 = 0;
                
        # if e_dot >= 0:
        #     r3 = -1e-3*coef
        # else:
        #     r3 = 1e-2*coef; # 0

        # r4 = -500*done*coef

        # reward = (r1+r2+r3+r4)*coef


        return np.array(self.state), reward, done, {}
    

    def reset(self):
        x_init = self.np_random.uniform(low=0.2, high=0.25, size=(1,))
        x_init[0] = round(x_init[0],5)
        self.state = [x_init[0], 0, 0, 0, 0, 0]
        self.ePrev = None
        # self.time = 0
        Pa = 0
        Pb = 0
        return np.array(self.state)
    
    # def render(self):
        # self.t += self.tau
        # self.ax.plot(self.t,self.state[0])
    

    
