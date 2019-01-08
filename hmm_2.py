"""Hidden Markov Model sequence tagger

"""
from classifier import Classifier
import numpy
import math
from collections import Counter

class HMM(Classifier):
        
    def get_model(self): return None
    def set_model(self, model): pass

    model = property(get_model, set_model)

    def __init__(self):
        self.features = 0
        self.labels = 0
        self.feature_list = {}
        self.label2index = {}
        self.index2label = {}
        
    def _collect_counts(self, instance_list):
        """Collect counts necessary for fitting parameters

        This function should update self.transtion_count_table
        and self.feature_count_table based on this new given instance
        
        Add your docstring here explaining how you implement this function

        Returns None
        """
        features = Counter()

        for instance in instance_list:
            features.update(instance.features())

        sig_features = set(feature for feature, count in features.items() if count > 0)
            
        self.transition_count_table = numpy.ones((1, 1))
        self.feature_count_table = numpy.ones((1, 1))
        self.initial_prob_table = numpy.zeros(1)
        self.final_prob_table = numpy.zeros(1)

        self.feature_list['UNK'] = self.features
        self.features += 1
        
        for instance in instance_list:
            instance.feature_vector = []
            
            for feature in instance.features():
                if feature in self.feature_list:
                    continue
                elif feature in sig_features:
                    self.feature_list[feature] = self.features
                    instance.feature_vector.append(feature)
                    self.features += 1
                        
            for label in instance.label:
                if label in self.label2index:
                    continue
                else:
                    self.label2index[label] = self.labels
                    self.labels += 1

            x, y = self.transition_count_table.shape
            x = self.labels - x
            y = self.labels - y

            self.initial_prob_table = numpy.pad(self.initial_prob_table, (0,x), 'constant', constant_values=(0))
            self.final_prob_table = numpy.pad(self.final_prob_table, (0,x), 'constant', constant_values=(0))
            self.transition_count_table = numpy.pad(self.transition_count_table, ((0,x),(0,y)), 'constant', constant_values=(1))
            self.fill_transition_count_table(instance)

            x, y = self.feature_count_table.shape
            x = self.features - x
            y = self.labels - y

            self.feature_count_table = numpy.pad(self.feature_count_table, ((0,x),(0,y)), 'constant', constant_values=(1))
            self.fill_feature_count_table(instance)

        self.index2label = {v: k for k, v in self.label2index.items()}
        
    def fill_transition_count_table(self, instance):
        for ii in range(0, len(instance.label)):
            a = self.label2index[instance.label[ii-1]]
            b = self.label2index[instance.label[ii]]
            if ii == (len(instance.label)-1):
                self.final_prob_table[b] += 1
            if ii == 0:
                self.initial_prob_table[b] += 1
            else:
                self.transition_count_table[a][b] += 1 

    def fill_feature_count_table(self, instance):
        for ii in range(0, len(instance.features())):
            if instance.features()[ii] in self.feature_list:
                a = self.feature_list[instance.features()[ii]]
            else:
                a = self.feature_list['UNK']

            b = self.label2index[instance.label[ii]]
            self.feature_count_table[a][b] += 1

    def train(self, instance_list):
        """Fit parameters for hidden markov model

        Update codebooks from the given data to be consistent with
        the probability tables 

        Transition matrix and emission probability matrix
        will then be populated with the maximum likelihood estimate 
        of the appropriate parameters

        Add your docstring here explaining how you implement this function

        Returns None
        """
        self.transition_matrix = numpy.zeros((1,1))
        self.emission_matrix = numpy.zeros((1,1))

        #TODO: estimate the parameters from the count tables
        self._collect_counts(instance_list)
        self.populate_transition_matrix()
        self.populate_emission_matrix()
        
        self.initial_prob_table = self.initial_prob_table/self.initial_prob_table.sum(axis=0)
        self.final_prob_table = self.final_prob_table/self.final_prob_table.sum(axis=0)

    def populate_transition_matrix(self):
        self.transition_matrix = self.transition_count_table/self.transition_count_table.sum(axis=0)[None,:]

    def populate_emission_matrix(self):
        self.emission_matrix = self.feature_count_table/self.feature_count_table.sum(axis=1)[:,None]
        
    def classify(self, instance):
        """Viterbi decoding algorithm

        Wrapper for running the Viterbi algorithm
        We can then obtain the best sequence of labels from the backtrace pointers matrix

        Add your docstring here explaining how you implement this function

        Returns a list of labels e.g. ['B','I','O','O','B']
        """
        instance.feature_vector = instance.features()
        trellis, backtrace_pointers = self.dynamic_programming_on_trellis(instance, False)
        a = len(trellis[0][:]) - 1
        l = [x for x in self.index2label]
        best_sequence = [max(l, key=lambda x: trellis[x][a])]
        
        for t in range(len(instance.feature_vector)-1, 0, -1):
            best_sequence.append(backtrace_pointers[int(best_sequence[-1]), t])
            
        best_sequence = [self.index2label[x] for x in best_sequence[::-1]]

        return best_sequence

    def make_feature_vector(self, instance):
        output = []
        
        for feature in instance.features():
            if feature in self.feature_list:
                output.append(self.feature_list[feature])
            else:
                output.append(self.feature_list['UNK'])
        return output

    def compute_observation_loglikelihood(self, instance):
        """Compute and return log P(X|parameters) = loglikelihood of observations"""
        forward, trellis = self.dynamic_programming_on_trellis(instance, True)
        loglikelihood = math.log(forward)
            
        return loglikelihood * -1

    def dynamic_programming_on_trellis(self, instance, run_forward_alg=True):
        """Run Forward algorithm or Viterbi algorithm

        This function uses the trellis to implement dynamic
        programming algorithm for obtaining the best sequence
        of labels given the observations

        Add your docstring here explaining how you implement this function

        Returns trellis filled up with the forward probabilities 
        and backtrace pointers for finding the best sequence
        """
        #TODO:Initialize trellis and backtrace pointers
        trellis = numpy.zeros((self.labels, len(instance.feature_vector)+1))
        backtrace_pointers = numpy.zeros((self.labels, len(instance.feature_vector)))

        #TODO:Traverse through the trellis here
        if run_forward_alg:
            trellis = numpy.zeros((self.labels, len(instance.feature_vector)+1))
            for ii in self.index2label:
                if instance.feature_vector[0] not in self.feature_list:
                    trellis[ii,0] = self.initial_prob_table[ii] * self.emission_matrix[self.feature_list['UNK'],ii]
                elif instance.feature_vector[0] in self.feature_list:
                    trellis[ii,0] = self.initial_prob_table[ii] * self.emission_matrix[self.feature_list[instance.feature_vector[0]],ii]
                                        
            for t in range(1, len(instance.feature_vector)):
                for state in self.index2label:
                    if instance.feature_vector[t] not in self.feature_list:
                        trellis[state,t] = numpy.sum([trellis[ii,t-1] * self.transition_matrix[ii,state] * self.emission_matrix[self.feature_list['UNK']][state] for ii in self.index2label])
                    elif instance.feature_vector[t] in self.feature_list:
                        trellis[state,t] = numpy.sum([trellis[ii,t-1] * self.transition_matrix[ii,state] * self.emission_matrix[self.feature_list[instance.feature_vector[t]],state] for ii in self.index2label])

            trellis[:,-1] = trellis[:,len(instance.feature_vector)-2] * self.final_prob_table[:]
            
            return (numpy.sum(trellis[:,-1]), trellis)
        
        else:
            for ii in self.index2label:
                if instance.feature_vector[0] not in self.feature_list:
                    trellis[ii][0] = self.initial_prob_table[ii] * self.emission_matrix[self.feature_list['UNK']][ii]
                elif instance.feature_vector[0] in self.feature_list:
                    trellis[ii][0] = self.initial_prob_table[ii] * self.emission_matrix[self.feature_list[instance.feature_vector[0]]][ii]
                                        
            for t in range(1, len(instance.feature_vector)):
                for state in self.index2label:
                    if instance.feature_vector[t] not in self.feature_list:
                        trellis[state][t] = max([trellis[ii][t-1] * self.transition_matrix[ii][state] * self.emission_matrix[self.feature_list['UNK']][state] for ii in self.index2label])
                    elif instance.feature_vector[t] in self.feature_list:
                        trellis[state][t] = max([trellis[ii][t-1] * self.transition_matrix[ii][state] * self.emission_matrix[self.feature_list[instance.feature_vector[t]]][state] for ii in self.index2label])
                    backtrace_pointers[state][t] = max(self.index2label, key=lambda x: trellis[x][t-1] * self.transition_matrix[state][x])

            trellis[:,-1] = trellis[:,len(instance.feature_vector)-2] * self.final_prob_table[:]
                  
        return (trellis, backtrace_pointers)

    def train_semisupervised(self, unlabeled_instance_list, labeled_instance_list=None):
        """Baum-Welch algorithm for fitting HMM from unlabeled data

        The algorithm first initializes the model with the labeled data if given.
        The model is initialized randomly otherwise. Then it runs 
        Baum-Welch algorithm to enhance the model with more data.

        Add your docstring here explaining how you implement this function

        Returns None
        """
        if labeled_instance_list is not None:
            self.train(labeled_instance_list)
        else:
            #TODO: initialize the model randomly
            features = Counter()

            self.labels = 3
            self.label2index['B'] = 0
            self.label2index['I'] = 1
            self.label2index['O'] = 2
            self.index2label = {v: k for k, v in self.label2index.items()}
            self.initial_prob_table = numpy.ones(self.labels)
            self.final_prob_table = numpy.ones(self.labels)
            self.transition_matrix = numpy.ones((self.labels, self.labels))

            self.final_prob_table = self.final_prob_table / self.final_prob_table.sum(axis=0)
            self.initial_prob_table = self.initial_prob_table / self.initial_prob_table.sum(axis=0)
            self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1)[None,:]

            for instance in unlabeled_instance_list:
                features.update(instance.features())

            sig_features = set(feature for feature, count in features.items() if count > 0)
                
            self.emission_matrix = numpy.ones((1, self.labels))

            self.feature_list['UNK'] = self.features
            self.features += 1
            
            for instance in unlabeled_instance_list:                
                for feature in instance.features():
                    if feature in self.feature_list:
                        continue
                    elif feature in sig_features:
                        self.feature_list[feature] = self.features
                        self.features += 1

                x, y = self.emission_matrix.shape
                x = self.features - x

                self.emission_matrix = numpy.pad(self.emission_matrix, ((0,x),(0,0)), 'constant', constant_values=(1))

            self.emission_matrix = self.emission_matrix / self.emission_matrix.sum(axis=1)[:,None]

        old_likelihood = float("inf")
        while True:
            #E-Step
            likelihood = 0.0
            for instance in unlabeled_instance_list:
                if len(instance.features()) > 1: 
                    instance.feature_vector = instance.features()
                    (alpha_table, beta_table, forward, backward) = self._run_forward_backward(instance)
                    #TODO: update the expected count tables based on alphas and betas
                    #also combine the expected count with the observed counts from the labeled data
                #M-Step
                #TODO: reestimate the parameters
                    prev_emission_matrix = self.emission_matrix
                    prev_transition_matrix = self.transition_matrix
                    pi = numpy.zeros_like(self.initial_prob_table)
                    Xi = numpy.zeros_like(self.transition_matrix)
                    gamma = numpy.zeros_like(self.emission_matrix)
                    
                    for ii in range(0, len(instance.feature_vector)):
                        if instance.feature_vector[ii] in self.feature_list:
                            gamma[self.feature_list[instance.feature_vector[ii]],:] = alpha_table[:,ii] * beta_table[:,ii] / forward

                    for ii in range(0, len(instance.feature_vector)-1):
                        for state1 in self.index2label:
                            for state2 in self.index2label:
                                if instance.feature_vector[ii+1] in self.feature_list:
                                    Xi[state1,state2] += (alpha_table[state1,ii] * self.transition_matrix[state1,state2] * self.emission_matrix[self.feature_list[instance.feature_vector[ii+1]],state2] * beta_table[state2,ii+1] / forward)
                                
                    for ii in range(0, len(instance.feature_vector)):
                        if instance.feature_vector[ii] in self.feature_list:
                            self.emission_matrix[self.feature_list[instance.feature_vector[ii]],:] = gamma[self.feature_list[instance.feature_vector[ii]],:] / numpy.sum(gamma[self.feature_list[instance.feature_vector[ii]],:])
                                      
                    for state in self.index2label:
                        self.transition_matrix[:,state] = Xi[:,state] / numpy.sum(Xi[:,state])

                    print(self.transition_matrix)
                    likelihood = self.compute_observation_loglikelihood(instance)
            
            if self._has_converged(old_likelihood, likelihood):
                print("New Log Likelihood: " + str(likelihood))
                break
            else:
                old_likelihood = likelihood
                print("Log Likelihood: " + str(old_likelihood))
                    
    def _has_converged(self, old_likelihood, likelihood):
        """Determine whether the parameters have converged or not
        Returns True if the parameters have converged.    
        """
        if likelihood >= old_likelihood:
            return True
        
        return False

    def _run_forward_backward(self, instance):
        """Forward-backward algorithm for HMM using trellis
    
        Fill up the alpha and beta trellises (the same notation as 
        presented in the lecture and Martin and Jurafsky)
        You can reuse your forward algorithm here

        return a tuple of tables consisting of alpha and beta tables
        """
        #TODO: implement forward backward algorithm right here
        forward, alpha_table = self.dynamic_programming_on_trellis(instance, True)
        trellis = numpy.zeros((self.labels, len(instance.feature_vector)+1))

        for ii in self.index2label:
            if instance.feature_vector[-1] not in self.feature_list:
                trellis[ii,-1] = self.final_prob_table[ii] * self.emission_matrix[self.feature_list['UNK'],ii]
            elif instance.feature_vector[-1] in self.feature_list:
                trellis[ii,-1] = self.final_prob_table[ii] * self.emission_matrix[self.feature_list[instance.feature_vector[-1]],ii]
            
        for t in range(len(instance.feature_vector)-1, 0, -1):
            for state in self.index2label:
                if instance.feature_vector[t] not in self.feature_list:
                    trellis[state,t] = numpy.sum([trellis[ii,t+1] * self.transition_matrix[ii,state] * self.emission_matrix[self.feature_list['UNK'],state] for ii in self.index2label])
                elif instance.feature_vector[t] in self.feature_list:
                    trellis[state,t] = numpy.sum([trellis[ii,t+1] * self.transition_matrix[ii,state] * self.emission_matrix[self.feature_list[instance.feature_vector[t]],state] for ii in self.index2label])

        if instance.feature_vector[0] not in self.feature_list:
            trellis[:,0] = numpy.sum([trellis[ii,1] * self.initial_prob_table[:] * self.emission_matrix[self.feature_list['UNK'],:] for ii in self.index2label])
        elif instance.feature_vector[0] in self.feature_list:
            trellis[:,0] = numpy.sum([trellis[ii,1] * self.initial_prob_table[:] * self.emission_matrix[self.feature_list[instance.feature_vector[0]],:] for ii in self.index2label])               
        
        beta_table = trellis
        backward = numpy.sum(trellis[:,0])
                                                                                            
        return (alpha_table, beta_table, forward, backward)
