# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:55:50 2024

@author: hma2
I assume we use a fixed time length
"""

import numpy as np
from scipy.stats import norm

class smart_grid:
    def __init__(self, temp_in, temp_ex, time, gamma, tau):
        self.temp_in = temp_in
        self.temp_ex = temp_ex
        self.time_int = time
        self.gamma = gamma
        self.tau = tau

        self.states = self.create_state()
        self.actions = [0, 1]  #[off, on]
        self.transition = self.getTransition()
        self.reward = self.reward_f()
        self.reward_l = self.reward_leader()

        self.nextSt_list, self.nextPro_list = self.stotrans_list()

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
                if next_t not in self.time_int:
                    trans[st][act]['Sink'] = 1
                else:
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

    def stotrans_list(self):
        #Prepare data to generate samples
        transition_list = {}
        transition_pro = {}
        for st in self.transition:
            transition_list[st] = {}
            transition_pro[st] = {}
            for act in self.transition[st]:
                transition_list[st][act] = {}
                transition_pro[st][act] = {}
                st_list = []
                pro_list = []
                for st_, pro in self.transition[st][act].items():
                    st_list.append(st_)
                    pro_list.append(pro)
                transition_list[st][act] = st_list
                transition_pro[st][act] = pro_list
        return transition_list, transition_pro

    def getcore(self, V, st, act):
        core = 0
        for st_, pro in self.transition[st][act].items():
            if st_ != 'Sink':
                core += pro * V[self.states.index(st_)]
        return core

    def get_policy_entropy(self, reward, flag):
        threshold = 0.0001
        if not flag:
            reward = self.reward_l
        else:
            self.update_reward(reward)
            reward = self.reward
        V = self.init_value()
        V1 = V.copy()
        policy = {}
        Q = {}
        for st in self.states:
            policy[st] = {}
            Q[st] = {}
        itcount = 1
        while (
            itcount == 1
            or np.inner(np.array(V) - np.array(V1), np.array(V) - np.array(V1))
            > threshold
        ):
            V1 = V.copy()
            for st in self.states:
                Q_theta = []
                for act in self.actions:
                    core = (reward[st][act] + self.gamma * self.getcore(V1, st, act)) / self.tau
                    # Q[st][act] = np.exp(core)
                    Q_theta.append(core)
                Q_sub = Q_theta - np.max(Q_theta)
                p = np.exp(Q_sub)/np.exp(Q_sub).sum()
                # Q_s = sum(Q[st].values())
                # for act in self.actions:
                    # policy[st][act] = Q[st][act] / Q_s
                for i in range(len(self.actions)):
                    policy[st][self.actions[i]] = p[i]
                V[self.states.index(st)] = self.tau * np.log(np.exp(Q_theta).sum())
            itcount += 1
        return V, policy

    def policy_evaluation(self, reward, flag, policy):
        threshold = 0.00001
        if not flag:
            reward = self.reward_l
        else:
            self.update_reward(reward)
            reward = self.reward
        V = self.init_value()
        delta = np.inf
        while delta > threshold:
            V1 = V.copy()
            for st in self.states:
                temp = 0
                for act in self.actions:
                    if act in policy[st].keys():
                        temp += policy[st][act] * (reward[st][act] + self.gamma * self.getcore(V1, st, act))
                V[self.states.index(st)] = temp
            delta = np.max(abs(V-V1))
        return V

    def init_value(self):
        #Initial the value to be all 0
        return np.zeros(len(self.states))

    def reward_f(self):
        # price is an array storing electricity price in each time interval
        reward = {}
        for st in self.states:
            reward[st] = {}
            for act in self.actions:
                reward[st][act] = self.reward_single(st, act)
        return reward

    def reward_single(self, state, act):
        """
        Define the follower's reward function

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
        T_star = 20 #ideal temperature of the user
        b = 1 #sensitivity varys among different users

        T_int = state[0]
        reward = - b * (T_int-T_star) ** 2
        return reward

    def reward_leader(self):
        """
        Define the leader's reward function

        Returns
        -------
        None.

        """
        reward = {}
        for state in self.states:
            t = state[2]
            reward[state] = {}
            for act in self.actions:
                c_t = 0 #unit is cents/kWh
                if 0 <= t < 5 or 20 <= c_t < 24:
                    c_t = 10 * act
                elif 5 <= t < 14:
                    c_t = 20 * act
                elif 14 <= t < 20:
                    c_t = 40 * act
                reward[state][act] = - c_t
        return reward

    def get_initial(self, external_temp): #return a dictionary of intial distribution

        """


        Parameters
        ----------
        initial_dist : dictionary
            DESCRIPTION. The initial state distribution

        Returns
        -------
        The distribution of initial states in a list form

        """
        t_0 = 0
        t_e = external_temp
        t_e_dict = t_e.next_extrenal_temp(0) # a distribution dict
        dict = {}
        for key in t_e_dict.keys():
            state = (key, key+2, t_0)
            dict[state] = t_e_dict[key]
        return dict

    def generate_sample(self, pi):
        #pi here should be pi[st] = [pro1, pro2, ...]
        traj = []
        st_index = np.random.choice(len(self.states), 1, p = self.init)[0]
        st = self.states[st_index]
        act_index = np.random.choice(len(self.actions), 1, p = pi[st])[0]
        act = self.actions[act_index]
        traj.append(st)
        traj.append(act)
        next_st = self.one_step_transition(st, act)
        while next_st != "Sink":
            st = next_st
            # st_index = self.states.index(st)
            act_index = np.random.choice(len(self.actions), 1, p = pi[st])[0]
            act = self.actions[act_index]
            traj.append(st)
            traj.append(act)
            next_st = self.one_step_transition(st, act)
        traj.append(next_st)
        return traj

    def one_step_transition(self, st, act):
        st_list = self.nextSt_list[st][act]
        pro_list = self.nextPro_list[st][act]
        next_st = np.random.choice(len(st_list), 1, p = pro_list)[0]
        return st_list[next_st]

    def reward_traj(self, traj, flag):
        #Flag is used to identify whether it is leader's reward or follower
        #Flag = 0 represents leader, Flag = 1 represents follower
        if not flag:
            reward = self.reward_l
        else:
            reward = self.reward
        st = traj[0]
        act = traj[1]
        if len(traj) >= 4:
            r = reward[st][act] + self.gamma * self.reward_traj(traj[2:], flag)
        else:
            return reward[st][act]
        return r

class inside_temp:
    def __init__(self, start, end):
        self.state = [i for i in range(start, end + 1)]

    def next_inside_temp(self, temp_in, temp_ex, heater):
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
        COP = 2.5
        m_air = 1250
        c_air = 1000
        lam = 90
        r_h = 1500
        delta_t = 3600 #Delta T = 3600s

        Q = heater * r_h * COP - lam * (temp_in - temp_ex)
        temp_in_next = temp_in + Q/(m_air * c_air) * delta_t
        return temp_in_next

class external_temp:
    def __init__(self, start, end):
        self.state = [i for i in range(start, end + 1)]

    def next_extrenal_temp(self, time):
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
        #assume time is from 0-23 hour
        T_sub_high, t1 = 13, 0
        T_low, t2 = 10, 6
        T_high, t3 = 25, 14
        t4 = 23
        if t1 <= time < t2:
            T = T_sub_high - (T_sub_high - T_low) / (t2 - t1) * (time - t1)
        elif t2 <= time < t3:
            T = T_high - (T_high - T_low) / (t3 - t2) * (t3 - time)
        else:
            T = T_high - (T_high - T_sub_high) / (t4 - t3) * (time - t3)

        distribution = {}

        sigma = 2
        list_p = []
        for st in self.state:
            st_l = st - 0.5
            st_r = st + 0.5
            cdf_st_l = norm.cdf(st_l, T, sigma)
            cdf_st_r = norm.cdf(st_r, T, sigma)
            p = cdf_st_r - cdf_st_l
            list_p.append(p)

        #normalize
        array_p = np.array(list_p)
        array_p = array_p/np.sum(array_p)

        count = 0
        for st in self.state:
            distribution[st] = array_p[count]
            count = count + 1

        return distribution


def test():
    e_temp = external_temp(10, 25)
    i_temp = inside_temp(20, 26)
    # print(e_temp.next_extrenal_temp(1))
    print(i_temp.next_inside_temp(20, 10, 1))
    

if __name__ == "__main__":
    test()

