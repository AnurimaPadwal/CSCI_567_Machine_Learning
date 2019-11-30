import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
    
	model = None
	##################################################
	
	S = len(tags)
	pi = [0] * S
	A = [[0 for i in range(S)] for j in range(S)]
	obs_dict = {}
	state_dict = {}
	for i,tag in enumerate(tags):
		#pi.append(1/len(tags))
		state_dict[tag] = i
			
	unique_words = set()
	idx = 0
	sequence_start_s = [0] * S
	state_count = [0] * S

	for line in train_data:
		for i, tag in enumerate(line.tags):
			pi[state_dict[tag]] += 1
	
	sum_pi = sum(pi)
	for i in range(S):
		pi[i] /= sum_pi

		
	
	for line in train_data:
		for word in line.words:
			if word not in unique_words:
				unique_words.add(word)
				obs_dict[word] = idx
				idx+=1
					
		for i in range(len(line.tags)-1):
			s_id = state_dict[line.tags[i]]
			s_next_id = state_dict[line.tags[i+1]]
			A[s_id][s_next_id] += 1
			sequence_start_s[s_id] += 1
			state_count[s_id] += 1
		
		last_state_id = state_dict[line.tags[len(line.tags) - 1]]
		state_count[last_state_id] += 1
		
	B = [[0 for j in range(len(obs_dict))] for i in range(S)]
	for line in train_data:
		for word, tag in zip(line.words, line.tags):
			tag_idx = state_dict[tag]
			obs_idx = obs_dict[word]
			B[tag_idx][obs_idx] += 1
	
	for i in range(S):
		d = np.sum(B[i])
		for j in range(len(obs_dict)):
			B[i][j] /= d if d!=0 else B[i][j]
			
	for s in range(S):
		d = np.sum(A[s])
		for s_next in range(S):
			A[s][s_next] /= d if d!=0 else A[s][s_next]
	pi = np.asarray(pi)
	A = np.asarray(A)
	B = np.asarray(B)
		
	model = HMM(pi, A, B, obs_dict, state_dict)        
				
										
	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	#############################################
	obs_dict = model.obs_dict
	A = model.A
	B = model.B
	state_dict = model.state_dict
	pi = model.pi
	S = len(pi)
	new_prob = np.full((S,1), 1e-6)
    
	for line in test_data:
		for i, word in enumerate(line.words):
			if word not in obs_dict:
				model.B = np.append(model.B, new_prob, axis = 1)
				obs_idx = len(model.B[0]) - 1
				model.obs_dict[word] = obs_idx
	#model.B = B
	#model.obs_dict = obs_dict
	
	
		tagged = model.viterbi(line.words)
		tagging.append(tagged)	
	
	return tagging

	###################################################
	