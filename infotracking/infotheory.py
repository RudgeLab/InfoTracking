import numpy as np

def entropy(hgram, ax=None):
    '''
    Entropy H(X) of one variable given joint histogram

    hgram = joint histogram (2d array)

    ax = axis over which to sum histogram to compute marginal distribution

    returns: entropy value
    '''
    # Convert bins counts to probability values
    px = hgram / float(np.sum(hgram))
    if ax:
        px = np.sum(px, axis=ax) # marginal for x over y
    nzs = px > 0 # Only non-zero pxy values contribute to the sum
    ex = -np.sum(px[nzs] * np.log2(px[nzs]))
    return ex

def mutual_information(hgram):
    '''
    Mutual information I(X,Y) for joint histogram

    hgram = joint histogram (2d array) 
    '''
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
    
def joint_entropy(hgram):
    '''
    Joint entropy H(X,Y) for joint histogram

    hgram = joint histogram (2d array) 

    returns: joint entropy value
    '''
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    #print pxy[nzs]
    return -np.sum(pxy[nzs] * np.log2(pxy[nzs]))

def conditional_entropy(hgram, ax):
    '''
    Conditional entropy H(Y|X) for joint histogram

    hgram = joint histogram (2d array) 

    ax = axis over which to sum to compute marginal distribution of X

    returns: joint entropy value
    '''
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=ax) # marginal for x over y
    je = joint_entropy(hgram)
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = px > 0 # Only non-zero pxy values contribute to the sum
    ex = -np.sum(px[nzs] * np.log2(px[nzs]))
    return je - ex 


def normalised_conditional_entropy(hgram, ax):
    '''
    Conditional entropy H(Y|X)/H(Y) for joint histogram

    hgram = joint histogram (2d array) 

    ax = axis over which to sum to compute marginal distribution of X

    returns: joint entropy value
    '''
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=ax) # marginal for x over y
    py = np.sum(pxy, axis=1-ax) # marginal for y over x
    je = joint_entropy(hgram)
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzsx = px > 0 # Only non-zero pxy values contribute to the sum
    nzsy = py > 0 # Only non-zero pxy values contribute to the sum
    ex = -np.sum(px[nzsx] * np.log2(px[nzsx]))
    ey = -np.sum(py[nzsy] * np.log2(py[nzsy]))
    return (je - ex) / ey 
