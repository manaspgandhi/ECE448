'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
'''
Note:
For grading purpose, all bigrams are represented as word1*-*-*-*word2

Although you may use tuple representations of bigrams within your computation, 
the key of the dictionary itself must be word1*-*-*-*word2 at the end of the computation.
'''

import numpy as np
import math
from collections import Counter

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    '''
    frequency = {}
    
    # parse through train
    for key, val in train.items():
        temp = []
        # get the words zipped, into the right format, and append them to the temp array
        # parse through list of lists
        for i in range(len(val)):
            array_of_zipped_words = zip(val[i][0:], val[i][1:])
            
            # parse through the tuples now and concatenate them with the right formatting
            for tup in array_of_zipped_words:
                temp.append(tup[0] + "*-*-*-*" + tup[1])
                
        # dict of Counters
        frequency[key] = Counter(temp)
    
    return frequency
    

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    stopwords (set of str):
        - Set of stopwords to be excluded

    Output:
    nonstop (dict of Counters): 
        - nonstop[y][x] = frequency of bigram x in texts of class y,
          but only if neither token in x is a stopword. x is in the format 'word1*-*-*-*word2'
    '''
    
    for classification, bigram in frequency.items():

        for word_pair in bigram:
            index_first_star = word_pair.index("*")
            index_last_star = index_first_star + 7
            word1 = word_pair[0 : index_first_star]
            word2 = word_pair[index_last_star : ]
        
            if(word1 and word2 in stopwords):
                del(word_pair)
    
    nonstop = frequency
    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of bigram x in y, where x is in the format 'word1*-*-*-*word2'
          and neither word1 nor word2 is a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary bigram given y


    Important: 
    Be careful that your vocabulary only counts bigrams that occurred at least once
    in the training data for class y.
    '''
    likelihood = {}
    
    # y = class, x = bigram, which is a counter
    for classification, bigram_counter in nonstop.items():
        num_all_tokens = sum(bigram_counter.values())
        
        num_bigram_types = len(bigram_counter)
        likelihood[classification] = {}
        denomenator = num_all_tokens + smoothness*(num_bigram_types + 1)

        for bigram, count in bigram_counter.items():
            #calculate
            num_x_tokens = count
            numerator = num_x_tokens + smoothness
            p_bigram_given_class = numerator/denomenator
            
            #set likelihood
            likelihood[classification][bigram] = p_bigram_given_class
            
        #OOV
        p_oov_given_class = smoothness/denomenator
        likelihood[classification]["OOV"] = p_oov_given_class
        
    
    return likelihood

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    hypotheses = []
    
    
    # loop thru all texts and form bigrams again, same way we did before
    # check the likelihood of each bigram with the likelihood param
    # add the log of the likelihood to the positive and negative
    # now we have pos and neg total, so we compare the total
    # to the final array add the greater one
    # add to out of vocabulary if one the words is a stop word
    # if both stop words then skip
    
    #load the positive and negative values with the logs to make sure we dont zero out
    # neg = 1 - pos
    
    # recreate the bigrams
    for i in range(len(texts)) :
        classifications = {"pos":math.log(prior), "neg":math.log(1-prior)}
        for j in range (len(texts[i]) - 1) :
            curr_bigram = (texts[i][j] + "*-*-*-*" + texts[i][j+1])
            
            for classification in classifications :    
                # add to both classes
                # make sure to log it when adding
                if curr_bigram in likelihood[classification] :
                    classifications[classification] += math.log(likelihood[classification][curr_bigram])
                elif not (texts[i][j + 1] in stopwords and texts[i][j] in stopwords):
                    classifications[classification] += math.log(likelihood[classification]["OOV"])
        
        # now figure out which one is greater and append the greater one or undecided to hypothesis
        if classifications["pos"] > classifications["neg"] :
            hypotheses.append("pos")
        elif classifications["pos"] < classifications["neg"] :
            hypotheses.append("neg")
        else :
            hypotheses.append("undecided")
            
    return hypotheses


def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    accuracies = np.zeros(shape=(len(priors), len(smoothnesses)))
    
    # parse through all the priors and smoothnesses to get the likelihood ratios and hypotheses
    for prior_val in range(len(priors)):
        for smoothness_val in range(len(smoothnesses)):
            hits = 0
            likelihood = laplace_smoothing(nonstop, smoothnesses[smoothness_val])
            hypotheses = naive_bayes(texts, likelihood, priors[prior_val])
            num_hypotheses = len(hypotheses)
            
            #find hits
            for k in range(num_hypotheses) :
                if labels[k] == hypotheses[k]  :
                    hits += 1
            
            # calculate accuracy based on hits
            accuracies[prior_val][smoothness_val] = hits/num_hypotheses
            
    return accuracies
            