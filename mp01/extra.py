import numpy as np

def estimate_geometric(PX):
    '''
    @param:
    PX (numpy array of length cX): PX[x] = P(X=x), the observed probability mass function

    @return:
    p (scalar): the parameter of a matching geometric random variable
    PY (numpy array of length cX): PY[x] = P(Y=y), the first cX values of the pmf of a
      geometric random variable such that E[Y]=E[X].
    '''
    ex = 0
    p = 0
    PY = 0
    i = 0
    cx = PX.shape
    
    for pmf in PX:
        ex += i * pmf
        i += 1
    
    p = 1/(ex+1)
    PY = np.zeros(cx)
    
    for j in range(len(PX)):
        PY[j] = p * ((1 - p)**j)
        
    return p, PY
