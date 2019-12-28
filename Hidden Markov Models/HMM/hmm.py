# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict
        self.alpha = None
        self.beta = None

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        for s in range(S):
            alpha[s][0] = self.pi[s] * self.B[s][self.obs_dict[Osequence[0]]]
        
        for t in range(1,L):
            for s in range(S):
                temp = 0
                for s_prev in range(S):
                    temp += alpha[s_prev][t-1] * self.A[s_prev][s]
                alpha[s][t] = self.B[s][self.obs_dict[Osequence[t]]] * temp
        
        #self.alpha = alpha
        ###################################################
        
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        for s in range(S):
            beta[s][L-1] = 1
           
        
        for t in range(L-2, -1, -1):
            for s in range(S):
                temp = 0
                for s_next in range(S):
                    temp += self.A[s][s_next] * self.B[s_next][self.obs_dict[Osequence[t+1]]] * beta[s_next][t+1]
                beta[s][t] = temp
                    
        #self.beta = beta
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        T = len(Osequence)
        prob = 0
        ###################################################
        S = len(self.pi)
        alpha = self.forward(Osequence)
        
        for i in range(S):
            prob += alpha[i][T-1]
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        sequence_prob_value = self.sequence_prob(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        
        for i in range(S):
            for t in range(L):
                prob[i][t] = (alpha[i][t] * beta[i][t])/sequence_prob_value
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        sequence_prob_val = self.sequence_prob(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        
        for s in range(S):
            for s_next in range(S):
                for t in range(L-1):
                    prob[s][s_next][t] = (self.A[s][s_next] * alpha[s][t] * self.B[s_next][self.obs_dict[Osequence[t+1]]] * beta[s_next][t+1])/sequence_prob_val
                    
        
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        S = len(self.pi)
        L = len(Osequence)
        delta = [[0 for t in range(L)] for s in range(S)]
        paths = [[None for t in range(L)] for s in range(S)]
        s_dict = {}
        
        
        for key, value in self.state_dict.items():
            s_dict[value] = key
            
        for s in range(S):
            delta[s][0] = self.pi[s] * self.B[s][self.obs_dict[Osequence[0]]]
   
        
        for t in range(1,L):
            for s in range(S):
                deltas = []
                for s_prev in range(S):
                    deltas.append(self.A[s_prev][s] * delta[s_prev][t-1])
                   
                delta[s][t] = self.B[s][self.obs_dict[Osequence[t]]] * np.max(deltas)
                paths[s][t] = s_dict[np.argmax(deltas)]
        
        
        path = [None] * L
        
        
        
        
        """
        for t in range(L-1, 0, -1):
            idx = self.state_dict[path[0]]
            path = [paths[idx][t]] + path
       """
        delta = np.asarray(delta)
        path[L-1] = s_dict[np.argmax(delta[:, L-1])]
        
        
        for i in range(L-2, -1, -1):
            id = self.state_dict[path[i+1]] 
            #print(paths[id][i-1])
            path[i] = paths[id][i+1] 
            
        
                         
            
        
        
        
        ###################################################
        return path
