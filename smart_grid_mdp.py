# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:55:50 2024

@author: hma2
I assume we use a fixed time length
"""

class smart_grid:
    def __init__(self, temp_in, temp_ex, time_int):
        self.temp_in = temp_in
        self.temp_ex = temp_ex
        self.time_int = time_int
        
        self.states = self.create_state()
        self.actions = [0, 1]  #[off, on]
        self.transition = self.getTransition()
        self.reward = self.reward_f()
    
    def create_state(self):
        #Initiate state space
        states = []
        for t_i in self.temp_in.state:
            for t_e in self.temp_ex.state:
                for t in self.time_int:
                    states.append((t_i, t_e, t))
        return states
    
    def getTransition(self):
        #Calculate transition function
        trans = {}
        for st in self.states:
            trans[st] = {}
            t_i = st[0]
            t_e = st[1]
            t = st[2]
            
            for act in self.actions:
                trans[st][act] = {}
                next_t_i = self.temp_in.next_inside_temp(t_i, t_e, t)
                next_t_e_dist = self.temp_ex.next_extrenal_temp(t)
                next_t = t + 1  #Assume the time interval is 1
                
                for next_t_e, pro in next_t_e_dist.items():
                    trans[st][act][(next_t_i, next_t_e, next_t)] = pro
                    
        self.check_trans(trans)
        return trans
    
    def check_trans(self, trans):
        #Check if the transition system is correct
        for st in trans.keys():
            for act in trans[st].keys():
                if abs(sum(trans[st][act].values())-1) > 0.01:
                    print("st is:", st, " act is:", act, " sum is:", sum(self.stotrans[st][act].values()))
                    return False
        # print("Transition is correct")
        return True
                
    def reward_f(self):
        reward = {}
        for st in self.states:
            reward[st] = {}
            for act in self.actions:
                reward[st][act] = self.reward_single(st, act)
        return reward
    
    def reward_single(self, state, act):
        """
        

        Parameters
        ----------
        state : tuple
            DESCRIPTION. current state
        act : 0 or 1
            DESCRIPTION. action taken at state

        Returns
        -------
        reward value given state and action

        """
        return 
        

class inside_temp:
    def __init__(self, start, end):
        self.state = [i for i in range(start, end + 1)]
    
    def next_inside_temp(temp_in, temp_ex, heater):
        """
        Xinyi Finish this part
        Parameters
        ----------
        temp_in : int
            DESCRIPTION. inside temperature
        temp_ex : int
            DESCRIPTION. external temperature
        heater : bool
            DESCRIPTION. heater on or not

        Returns
        -------
        inside temperature of next time step

        """
        
        return
    
class external_temp:
    def __init__(self, start, end):
        self.state = [i for i in range(start, end + 1)]
        
    def next_extrenal_temp(time):
        """
        Xinyi Finish this part        
        
        Parameters
        ----------
        time : int
            DESCRIPTION. current time

        Returns
        -------
        a distribution of temperature at this time. key is the temperature, value is the probability

        """
        
        distribution = {}
        return distribution