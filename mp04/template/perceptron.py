# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020



import numpy as np

def trainPerceptron(train_set, train_labels,  max_iter):
    #Write code for Mp4
    W = np.zeros(len(train_set[0]))
    b = 0
    eta = 0.01 #learning rate
    
    for i in range(max_iter):
        for j in range(len(train_set)):
            over_under_estimate = 0
            x = np.array(train_set[j])
            
            # Step 1: compute the classifier output y_hat = argmax(wTkx + bk)
            y_hat = np.add(np.dot(W, x), b)
            
            # we need to know if it is an over or under estimate
            if (y_hat < 0): # meaning underestimate
                over_under_estimate = 0 
            else: # meaning overestimate
                over_under_estimate = 1
            
            # Step 2: Update the weight vectors
            if (over_under_estimate == 0 and train_labels[j] == 1):
                W = np.add(W, np.multiply(eta, x))
                b = np.add(b, eta)
            elif (over_under_estimate == 1 and train_labels[j] == 0):
                W = np.subtract(W, np.multiply(eta, x))
                b = np.subtract(b, eta)
                    
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4
    W, b = trainPerceptron(train_set, train_labels,  max_iter)
    classification_set = []
    
    for i in range(len(dev_set)):
        x = dev_set[i]
        y_hat = np.add(np.dot(W, np.array(x)), b)
        
        over_under_estimate = 0
        if (y_hat < 0):
            over_under_estimate = 0 
        else:
            over_under_estimate = 1
        classification_set.append(over_under_estimate)

    return classification_set