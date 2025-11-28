'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''
    # find dimensions
    elems = 0
    for i in range(len(texts)):
        max_text = Counter(texts[i])
        
        #max for word1
        if(max_text[word0] > elems):
            elems = max_text[word0]
        
    Pmarginal = np.zeros(elems + 1)
    
    #loop through all texts and find number of instances of word0
    for i in range(len(texts)):
        curr_text = Counter(texts[i])
        curr_count = curr_text[word0]
        
        # add 1 to the index corresponding to the count
        # for now, we collect the number of texts with a given count, corresponding to the index
        # later will convert to percentages
        curr_count_num = int(curr_count)
        Pmarginal[curr_count_num] += 1
    
    #now we convert to percentages
    for j in range(len(Pmarginal)):
        Pmarginal[j] = (Pmarginal[j]/len(texts))
   
    return Pmarginal
    
def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''
    # start by finding the dimensions of Pcond
    rows = 0
    cols = 0
    for i in range(len(texts)):
        max_text = Counter(texts[i])
        
        #max for word1
        if(max_text[word0] > rows):
            rows = max_text[word0]
        
        #max for word1
        if(max_text[word1] > cols):
            cols = max_text[word1]
        
    Pcond = np.zeros((rows + 1, cols + 1))
    
    for i in range(len(texts)):
        curr_text = Counter(texts[i])
        curr_count_0 = curr_text[word0]
        curr_count_1 = curr_text[word1]
                
        # add 1 to the index corresponding to the count
        # for now, we collect the number of texts with a given count, corresponding to the index
        # later will convert to percentages
        curr_count_num_0 = int(curr_count_0)
        curr_count_num_1 = int(curr_count_1)
        
        Pcond[curr_count_num_0][curr_count_num_1] += 1
    
    Px0 = marginal_distribution_of_word_counts(texts, word0) 

    #at this point we have the counts for both words occuring x amount of times
    #now we convert to conditional probability
    for row in range(len(Pcond)):
        for col in range(len(Pcond[0])):
            Pcond[row][col] = (Pcond[row][col]/len(texts))
            if (Px0[row] == 0):
                Pcond[row][col] = np.nan
            else:
                Pcond[row][col] = Pcond[row][col]/Px0[row]
                
    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''

    cols = Pmarginal.shape
    Pjoint = np.zeros(Pcond.shape)
    for row in range(len(Pcond)):
        for col in range(len(Pcond[0])):
            if(np.isnan(Pcond[row][col])):
                Pjoint[row][col] = Pmarginal[row]
            else:
                Pjoint[row][col] = Pcond[row][col] * Pmarginal[row]
    
   
    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''
    mu = np.zeros(2)
    
    for row in range(len(Pjoint)):
        for col in range(len(Pjoint[0])):
            
            mu[0] += Pjoint[row][col] * row
            mu[1] += Pjoint[row][col] * col
    
    for i in range(len(mu)):
        mu[i]/(Pjoint.shape[i])
    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    
    Sigma = np.zeros((2, 2))
    #first get the two variances
    denom1 = 0
    numer1 = 0
    word_count = 0
    for i in range(len(Pjoint)):
        word_count = 0
        for j in range(len(Pjoint[0])):
            word_count += Pjoint[i][j]
            
        word_count *= len(Pjoint)    
        denom1 += word_count
        numer1 += ((i - mu[0])**2) * word_count
    
    Sigma[0][0] = numer1/denom1
    
    numer2 = 0
    denom2 = 0
    for i in range(len(Pjoint[0])):
        word_count = sum(Pjoint[:,i]) * len(Pjoint[0])    
        denom2 += word_count
        numer2 += ((i - mu[1])**2) * word_count
    
    Sigma[1][1] = numer2/denom2
    
    #now we can get the covariances
    for i in range(len(Pjoint)):
        for j in range(len(Pjoint[0])):
            Sigma[0][1] += (i - mu[0]) * (j - mu[1]) * Pjoint[i][j]
            Sigma[1][0] += (j - mu[1]) * (i - mu[1]) * Pjoint[i][j]
    
    return Sigma

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    Pfunc = Counter()
    for i in range(len(Pjoint)):
        for j in range(len(Pjoint[0])):
            Pfunc[(f(i, j))] += Pjoint[i][j]

    return Pfunc
    
