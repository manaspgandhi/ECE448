'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here
epsilon = 0.0005

def baseline(test, train):
    '''
    Implementation for the baseline tagger.
    input:  test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
            training data (list of sentences, with tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    '''
    [['START', 'appointment', 'of', 'william', 's.', 'pfaff', 'jr.', ',', '41', ',', 'as', 'promotion', 'manager', 'of', 'the', 'times-picayune', 'publishing', 'company', 'was', 'announced', 'saturday', 'by', 'john', 'f.', 'tims', ',', 'president', 'of', 'the', 'company', '.', 'END'], ['START', 'pfaff', 'succeeds', 'martin', 'burke', ',', 'who', 'resigned', '.', 'END'], ['START', 'the', 'new', 'promotion', 'manager', 'has', 'been', 'employed', 'by', 'the', 'company', 'since', 'january', ',', '1946', ',', 'as', 'a', 'commercial', 'artist', 'in', 'the', 'advertising', 'department', '.', 'END'], ['START', 'he', 'is', 'a', 'native', 'of', 'new', 'orleans', 'and', 'attended', 'allen', 'elementary', 'school', ',', 'fortier', 'high', 'school', 'and', 'soule', 'business', 'college', '.', 'END'],
    '''
    trained_data = []
    
    # tag : how many times it came up
    tagset = {}
    # this is a coutner that is going to store the counts of how many times certain 
    # word-tag pairs occur
    c = Counter()
        
    # parse through test data to get the word
    for train_sentence in train:
        curr_sentence_output = []
        c.update(train_sentence)
        for train_word_pair in train_sentence:
            if(train_word_pair[1] not in tagset):
                tagset[train_word_pair[1]] = 1
            else:
                tagset[train_word_pair[1]] += 1
                
    # now we need to find the tag that has the highest frequency
    # this will be used for words we don't see
    most_common_tag = ""
    most_common_tag_frequency = 0
    for tag in tagset :
        curr_frequency = tagset[tag]
        if curr_frequency > most_common_tag_frequency :
            most_common_tag = tag
            most_common_tag_frequency = curr_frequency

    # now let's iterate through test and label all the words with tags
    for test_sentence in test:
        curr_sentence = []
        
        for test_word in test_sentence:
            best_tag = ''
            highest_frequency_of_tag = 0
            
            # assign tags now
            # account for START and END tags
            if(test_word == 'START'):
                best_tag = 'START'    
            elif(test_word == 'END'):
                best_tag = 'END'
            # if it is a word and not START or END, then do this
            for tag in tagset:    
                potential_train_word_pair = (test_word, tag)
                curr_word_pair_occur_count = c[potential_train_word_pair]
                if(curr_word_pair_occur_count > highest_frequency_of_tag):
                    highest_frequency_of_tag = curr_word_pair_occur_count
                    best_tag = tag
            
            if(highest_frequency_of_tag == 0):
                curr_sentence.append((test_word, most_common_tag))
            else:
                curr_sentence.append((test_word, best_tag))

        trained_data.append(curr_sentence) 
    
    return trained_data


def viterbi(test, train):
    '''
    Implementation for the viterbi tagger.
    input:  test data (list of sentences, no tags on the words)
            training data (list of sentences, with tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    '''
    3 PROBABILITIES NEEDED:
    1. Initial probabilities (How often does each tag occur at the start of a sentence?)
    2. Transition probabilities (How often does tag 𝑡𝑏 follow tag 𝑡𝑎?)
    3. Emission probabilities (How often does tag t yield word w?)

    STEPS:
    1. Count occurrences of tags, tag pairs, tag/word pairs.
    Compute smoothed probabilities
    2. Take the log of each probability
    3. Construct the trellis. Notice that for each tag/time pair, you must store not only the probability of the best path but also a 
       pointer to the previous tag/time pair in that path.
    4. Return the best path through the trellis.
    
    '''
   
    # let's start by counting the occurences of tags, tag pairs, and tag/word pairs 
    initial_probabilities = Counter()               # tags and frequencies
    emission_probabilities = defaultdict(Counter)   # curr/previous tag pairs
    transition_probabilities = defaultdict(Counter) # tag/word pairs
    
    # count the tags and tag word pairs 
    for i in range(len(train)):
        sentence = train[i]
        
        for bigram_index in range(len(sentence)):
            word, tag = sentence[bigram_index]
            
            emission_probabilities[tag][word] += 1
            if bigram_index == 0:
                initial_probabilities[tag] += 1
    
    for i in range(len(train)):
        sentence = train[i]
        for (primary_word, primary_tag), (secondary_word, secondary_tag) in zip(sentence, sentence[1:]):
            transition_probabilities[primary_tag][secondary_tag] += 1
    
    # now let's log all these probabilities with laplace smoothing
    initial_probabilities = laplaceSmoothing(initial_probabilities, epsilon)
    emission_probabilities = laplaceSmoothing(emission_probabilities, epsilon)
    transition_probabilities = laplaceSmoothing(transition_probabilities, epsilon)
    
    tags = list(emission_probabilities.keys())
    
    def constructTrellis(sentence):
        
        # Format of each trellis element: (primary_tag_index, secondary_tag_index, probability)
        trellis = np.zeros((len(tags), len(sentence)), dtype=object)
       
        # get the starting position of the trellis and fill it in
        startData = (0, None, log(initial_probabilities['START']))      
        for tag in range(len(trellis)):
            trellis[tag, 0] = startData
        
        # go through each of the words in the sentence
        for word_index in range(len(trellis[0])):
            #first one is start
            if(word_index != 0):
                
                word = sentence[word_index]

                for primary_tag_index in range(len(trellis)): # Go through each possible part of speech/tag
                    primary_tag = tags[primary_tag_index]
                    highest_tag_probability = (primary_tag_index, -1, -np.inf)

                    for secondary_tag_index in range(len(tags)):
                        secondary_tag = tags[secondary_tag_index]

                        # make sure that the tag is known
                        if(primary_tag in transition_probabilities[secondary_tag]): 
                            # check if the word exists within the primary tag's words
                            # if it's not, it is 
                            if word in emission_probabilities[primary_tag]:
                                emission_probability = log(emission_probabilities[primary_tag][word])
                            else:
                                emission_probability = log(emission_probabilities[primary_tag]['UNKNOWN'])
                            transition_probability = log(transition_probabilities[secondary_tag][primary_tag])

                            tag_probability = (primary_tag_index, secondary_tag_index, transition_probability + emission_probability + trellis[secondary_tag_index, word_index-1][2])
                            # get the max probability for the given tag word pair
                            if(tag_probability[2] > highest_tag_probability[2]):
                                highest_tag_probability = tag_probability

                    # Best tag probablity based on transition from all possible previous tags to current tag 
                    trellis[primary_tag_index, word_index] = highest_tag_probability
        
        # print(trellis)
        return trellis
        
    def backtrace(trellis, sentence):
        # let's start by getting the highest probability tag in the last column of the trellis first
        # then work backward
        primary_tag_index, secondary_tag_index, probability = max(trellis[:,-1], key=(lambda x: x[2]))
        
        
        
        #path of the trellis, we will look for the optimal version of this
        trellis_path = []
        
        #start with the end of the sentence and make our way backwards
        trellis_path.append((sentence[-1], tags[primary_tag_index]))
        
        primary_tag_index, secondary_tag_index, probability = trellis[primary_tag_index, len(sentence)-1]
        
        for back_word_index in range(len(sentence) - 2, -1, -1):
            prev_trellis_elem = (sentence[back_word_index], tags[secondary_tag_index])
            trellis_path.append(prev_trellis_elem)
            
            primary_tag_index, secondary_tag_index, probability = trellis[secondary_tag_index, back_word_index]
        
        # make sure the start of the path starts with a start
        trellis_path[-1] = (trellis_path[-1][0], 'START')
        
        # we need to return the reversed path, as we check in reverse order
        return trellis_path[::-1]
        
    return [backtrace(constructTrellis(sentence), sentence) for sentence in test]

def laplaceSmoothing(probability_array, epsilon):
    
    smoothed_probabilities = defaultdict(Counter)
    if(type(probability_array) is defaultdict):
    
        for frequency, bigram in probability_array.items():
            classifications = len(bigram) + 1
            class_tokens = sum(bigram.values())
            
            for key2, bigram_tokens in bigram.items():
                smoothed_probabilities[frequency][key2] = (bigram_tokens + epsilon) / (class_tokens + (epsilon * (classifications)))
            
            smoothed_probabilities[frequency]['UNKNOWN'] = epsilon / (class_tokens + (epsilon * (classifications)))
        
    else:
        classifications = len(probability_array)
        class_tokens = sum(probability_array.values())
        for frequency, total_tokens in probability_array.items():
            smoothed_probabilities[frequency] = (total_tokens) / (class_tokens)        
   
    return smoothed_probabilities



def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
#     tag_counts = defaultdict(int)
#     transition_counts = defaultdict(int)
#     emission_counts = defaultdict(int)
#     hapax_counts = defaultdict(int)

#     # Count word occurrences and identify hapax words
#     for sentence in train:
#         for word, tag in sentence:
#             tag_counts[tag] += 1
#             emission_counts[(tag, word)] += 1
#             if word not in hapax_counts:
#                 hapax_counts[word] = 0
#             hapax_counts[word] += 1

#     hapax_tag_probs = defaultdict(float)
#     for word, count in hapax_counts.items():
#         for tag in tag_counts:
#             if (tag, word) in emission_counts:
#                 hapax_tag_probs[tag] += 1

#     # Normalize hapax tag probabilities
#     for tag in tag_counts:
#         hapax_tag_probs[tag] /= sum(hapax_tag_probs.values())

#     # Laplace smoothing constant
#     alpha = 0.001

#     # Compute smoothed emission probabilities
#     num_tags = len(tag_counts)
#     num_words = sum(tag_counts.values())

#     emission_probs = {}
#     for (tag, word), count in emission_counts.items():
#         hapax_prob = hapax_tag_probs[tag]
#         emission_probs[(tag, word)] = (count + alpha) / (tag_counts[tag] + alpha * num_words) * hapax_prob

#     # Construct trellis
#     tagged_sentences = []
#     for sentence in test:
#         trellis = [{}]

#         # Initialize trellis for the first word
#         for tag in tag_counts:
#             if (tag, sentence[0]) in emission_probs:
#                 trellis[0][tag] = {
#                     'prob': math.log(emission_probs[(tag, sentence[0])]),
#                     'prev': None
#                 }
#             else:
#                 trellis[0][tag] = {
#                     'prob': math.log(alpha / (tag_counts[tag] + alpha * num_words)),
#                     'prev': None
#                 }

#         # Fill in the trellis for subsequent words
#         for t in range(1, len(sentence)):
#             trellis.append({})
#             for tag in tag_counts:
#                 max_prob = float('-inf')
#                 prev_tag_selected = None
#                 for prev_tag in tag_counts:
#                     if (prev_tag, tag) in transition_counts and (tag, sentence[t]) in emission_probs:
#                         prob = trellis[t - 1][prev_tag]['prob'] + math.log(transition_counts[(prev_tag, tag)]) + math.log(emission_probs[(tag, sentence[t])])
#                     else:
#                         prob = trellis[t - 1][prev_tag]['prob'] + math.log(alpha / (tag_counts[prev_tag] + alpha * num_tags))

#                     if prob > max_prob:
#                         max_prob = prob
#                         prev_tag_selected = prev_tag

#                 trellis[t][tag] = {'prob': max_prob, 'prev': prev_tag_selected}

#         # Find the best path through the trellis
#         best_path = []
#         max_prob = float('-inf')
#         best_tag = None
#         for tag in tag_counts:
#             if trellis[-1][tag]['prob'] > max_prob:
#                 max_prob = trellis[-1][tag]['prob']
#                 best_tag = tag

#         for t in range(len(sentence) - 1, -1, -1):
#             best_path.insert(0, (sentence[t], best_tag))
#             best_tag = trellis[t][best_tag]['prev']

#         tagged_sentences.append(best_path)

#     return tagged_sentences
