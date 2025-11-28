'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
   
    
    P = np.zeros((model.M, model.N, 4, model.M, model.N))
    
    for r in range(model.M):
        for c in range(model.N):
            # make sure we're not at the end
            if not model.TS[r, c]:
                # we need to set the indexes for where to move based on walls

                #initial vals for where it will go if no walls
                up = r
                down = r
                left = c
                right = c

                # now check for walls using W
                if (not model.W[max(r - 1, 0), c]):
                    up = max(r - 1, 0)

                if not model.W[min(r + 1, model.M - 1), c]:
                    down = min(r + 1, model.M - 1)

                if (not model.W[r, max(c - 1, 0)]):
                    left = max(c - 1, 0)

                if (not model.W[r, min(c + 1, model.N - 1)]):
                    right = min(c + 1, model.N - 1)

                for a in range(4):
                    # now for the actions
                    # calculate the probability of going in one direction to (r', c'), 
                    # and then both options for directions that are one adjascent to (r', c') as well

                    # move left   
                    if a == 0:
                        P[r, c, a, r, left] += model.D[r, c, 0]
                        P[r, c, a, down, c] += model.D[r, c, 1]
                        P[r, c, a, up, c] += model.D[r, c, 2]

                    # move up
                    elif a == 1:
                        P[r, c, a, up, c] += model.D[r, c, 0]
                        P[r, c, a, r, left] += model.D[r, c, 1]
                        P[r, c, a, r, right] += model.D[r, c, 2]

                    # move right
                    elif a == 2:
                        P[r, c, a, r, right] += model.D[r, c, 0]
                        P[r, c, a, up, c] += model.D[r, c, 1]
                        P[r, c, a, down, c] += model.D[r, c, 2]

                    # move down
                    elif a == 3:
                        P[r, c, a, down, c] += model.D[r, c, 0]
                        P[r, c, a, r, right] += model.D[r, c, 1]
                        P[r, c, a, r, left] += model.D[r, c, 2]
    
    return P


def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    # value iteration formula = u_i(s) = r(s) + gamma * max(sum(P(s' | s, a) * u_i-1(s')))
    
    U_next = np.zeros((model.M, model.N))
    
    # all moves
    for r in range(model.M):
        for c in range(model.N):
            utility = float('-inf')
            
            # all four actions
            for a in range(4):
                probability_option = P[r, c, a].flatten()
                ui = U_current.flatten()
                utility = max(np.dot(probability_option, ui), utility)

            U_next[r, c] = model.R[r, c] + model.gamma * utility
    
    return U_next

def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    # value iteration formula = u_i(s) = r(s) + gamma * max(sum(P(s' | s, a) * u_i-1(s')))
    
    U = np.zeros((model.M, model.N))
    U_prime = np.zeros((model.M, model.N))
    P_transition = compute_transition(model)
    
    for i in range(100): # needs 100 iterations
        U = compute_utility(model, U_prime, P_transition)
        
        # check the convergence criterion
        # convergence criterion: |U_{i+1}(s) - U_i(s)| < epsilon
        convergence_criterion = np.max(abs(U - U_prime))
        if convergence_criterion < epsilon:
            break # means it has converged and we are done
            
        U_prime = U
    return U
    
def policy_evaluation(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP();
    
    Output:
    U - The converged utility function, which is an M x N array
    '''
    # policy iteration forumla: u_pi(s) = r(s) + gamma * sum(P(s' | s, pi(s)) * u_pi(s'))
    
    U_pi = np.zeros((model.M, model.N))
    U = np.zeros((model.M, model.N))
    
    for i in range(10000):
        for r in range(model.M):
            for c in range(model.N):
                # policy evaluation formula
                probability = model.FP[r, c].flatten()
                upi = U_pi.flatten()
                U[r, c] = model.R[r, c] + model.gamma * np.dot(probability, upi)
                
        # check the convergence criterion
        # convergence criterion: |U_{i+1}(s) - U_i(s)| < epsilon
        convergence_criterion = np.max(abs(U - U_pi))
        if convergence_criterion < epsilon:
            break # means it has converged and we are done
            
        U_pi = U.copy()
        
    return U
