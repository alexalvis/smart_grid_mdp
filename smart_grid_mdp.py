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
        self.trans_kernal_Q()
#        self.initial = self.get_initial()
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
                next_t_i = self.temp_in.next_inside_temp(t_i, t_e, act)
                next_t_e_dist = self.temp_ex.next_extrenal_temp(t)
                next_t = t + 1  #Assume the time interval is 1
                if next_t not in self.time_int:
                    trans[st][act]['Sink'] = 1
                elif next_t_i == 'P':
                    trans[st][act]['Penalty'] = 1
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
            if st_ == 'Penalty':
                return -300
            if st_ != 'Sink':
                core += pro * V[self.states.index(st_)]
        return core

    def get_policy_entropy(self, reward, flag):
        threshold = 0.0001
        if not flag:
            reward = self.reward_l
        else:
#            self.update_reward(reward)
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
            print("itcount:", itcount)
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
        T_star = 23 #ideal temperature of the user
        b = 1 #sensitivity varys among different users, may need further adjustment

        T_int = state[0]
        price = 2
        r_h = 1.50 #unit is kW
        cost = act * r_h * price


        reward = - b * (T_int-T_star) ** 2 - cost
#        reward/= 10
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
        initial_dist = {}
        for key in t_e_dict.keys():
            state = (key, key+2, t_0) # to be dicussed
            initial_dist[state] = t_e_dict[key]
        return initial_dist


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
    
    def stVisitFre(self, policy):
        threshold = 0.00001
        gamma = 0.95
        Z0 = self.initial
        Z_new = Z0.copy()
        Z_old = Z_new.copy()
        itcount = 1

        while itcount == 1 or np.inner(np.array(Z_new)-np.array(Z_old), np.array(Z_new)-np.array(Z_old)) > threshold:
#            print(itcount)
            Z_old = Z_new.copy()
            Z_new = Z0.copy()
            for st in self.states:
                index_st = self.states.index(st)
                for act in self.actions:
                    for st_ in self.states:
                        if st in self.stotrans[st_][act].keys():
                            Z_new[index_st] += gamma * Z_old[self.states.index(st_)] * policy[st_][act] * self.stotrans[st_][act][st]
            
            itcount += 1
        return Z_new
    
    def stactVisitFre(self, policy):
        Z = self.stVisitFre(policy)
        st_act_visit = {}
        for i in range(len(self.states)):
            st_act_visit[self.states[i]] ={}
            for act in self.actions:
                st_act_visit[self.states[i]][act] = Z[i] * policy[self.states[i]][act]
        return st_act_visit

    def transition_kernel(self, policy):
        trans = np.zeros((len(self.states), len(self.states)))
        for st in self.states:
            st_index = self.states.index(st)
            for act in self.actions:
                for next_st, pro in self.transition[st][act].items():
                    if next_st in self.states:
                        next_st_index = self.states.index(next_st)
                        trans[st_index][next_st_index] += pro * policy[st][act]
        
        self.transKernel = trans
        
    def trans_kernal_Q(self):
        len_act = len(self.actions)
        trans_Q = np.zeros((len(self.states)*len(self.actions), len(self.states)))
        for i in range(len(self.states)):
            for j in range(len(self.actions)):
                for next_st, pro in self.transition[self.states[i]][self.actions[j]].items():
                    if next_st in self.states:
                        trans_Q[i * len_act + j][self.states.index(next_st)] = pro
        self.trans_matrix = trans_Q
        
    def successor(self):
        I = np.identity(len(self.states))
        suc = np.linalg.inv(I -self.gamma * self.transKernel)
        return suc
    
    def suc_v(self, suc, policy):
        len_act = len(self.actions)
        reward = self.reward_f()
        R = np.zeros(len(self.states))
        for i in range(len(self.states)):
            R[i] = sum(policy[self.states[i]][act] * reward[self.states[i]][act] for act in self.actions)
        suc_v = suc.dot(R)
        R_Q = np.zeros(len(self.states)*len(self.actions))
        for i in range(len(self.states)):
            for j in range(len(self.actions)):
                R_Q[i * len_act + j] = reward[self.states[i]][self.actions[j]]
        suc_Q = R_Q.T + self.gamma * self.trans_matrix.dot(suc_v)
        
        return suc_v, suc_Q, R_Q
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
        m_air = 2000
        c_air = 2000
        lam = 400
        r_h = 900
        delta_t = 3600 #Unit is second

        Q = heater * r_h * COP - lam * (temp_in - temp_ex)
        temp_in_next = round(temp_in + Q/(m_air * c_air) * delta_t)
        
        if heater:
            temp_in_next = self.state[0] if temp_in_next < self.state[0] else self.state[-1] if temp_in_next > self.state[-1] else temp_in_next
        else:
            if temp_in_next < self.state[0]:
                temp_in_next = 'P'
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
        start = self.state[0]
        end = self.state[-1]
        #assume time is from 0-23 hour
        T_sub_high, t1 = start + (end - start)/3, 0
        T_low, t2 = start, 6
        T_high, t3 = end, 14
        t4 = 23
        if t1 <= time < t2:
            T = T_sub_high - (T_sub_high - T_low) / (t2 - t1) * (time - t1)
        elif t2 <= time < t3:
            T = T_high - (T_high - T_low) / (t3 - t2) * (t3 - time)
        else:
            T = T_high - (T_high - T_sub_high) / (t4 - t3) * (time - t3)

        sigma = 2
        dict_p = {}
        for st in self.state:
            st_l = st - 0.5
            st_r = st + 0.5
            cdf_st_l = norm.cdf(st_l, T, sigma)
            cdf_st_r = norm.cdf(st_r, T, sigma)
            p = cdf_st_r - cdf_st_l
            if p > 0.05:
                dict_p[st] = p

        #normalize
        sum_values = sum(dict_p.values())
        normalized_dict_p = {key: value / sum_values for key, value in dict_p.items()}

        return normalized_dict_p


def test():
    e_temp = external_temp(13, 26)
    i_temp = inside_temp(20, 26)
#    print(e_temp.next_extrenal_temp(23))
    print(i_temp.next_inside_temp(20, 16, 1))
    time = [i for i in range(24)]
    gamma = 0.95
    tau = 2
    sg = smart_grid(i_temp, e_temp, time, gamma, tau)
    # sg = None
    return sg

if __name__ == "__main__":
    sg = test()
    # state_1 = (23, 13, 23)
    # state_2 = (22, 13, 23)
    # state_3 = (24, 13, 23)
    V, policy = sg.get_policy_entropy([], 1)
    sg.transition_kernel(policy)
    suc = sg.successor()
    suc_v, suc_Q, R_Q = sg.suc_v(suc, policy)
    trans_matrix = sg.trans_matrix
    # index_1 = sg.states.index(state_1)
    # index_2 = sg.states.index(state_2)
    # index_3 = sg.states.index(state_3)
    # print(V[index_1], V[index_2], V[index_3])
  

