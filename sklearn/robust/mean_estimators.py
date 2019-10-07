
import numpy as np

def blockMOM(x,K):
    '''Sample the indices of K blocks for data x using a random permutation

    Parameters
    ----------

    K : int
        number of blocks

    x : array like, length = n_sample
        sample whose size correspong to the size of the sample we want to do blocks for.

    Returns
    -------

    list of size K containing the lists of the indices of the blocks, the size of the lists are contained in [n_sample/K,2n_sample/K]
    '''
    b=int(np.floor(len(x)/K))
    nb=K-(len(x)-b*K)
    nbpu=len(x)-b*K
    perm=np.random.permutation(len(x))
    blocks=[[(b+1)*g+f for f in range(b+1) ] for g in range(nbpu)]
    blocks+=[[nbpu*(b+1)+b*g+f for f in range(b)] for g in range(nb)]
    return [perm[b] for  b in blocks]

def MOM(x,blocks):
    '''Compute the median of means of x using the blocks blocks

    Parameters
    ----------

    x : array like, length = n_sample
        sample from which we want an estimator of the mean

    blocks : list of list, provided by the function blockMOM.

    Return
    ------

    The median of means of x using the block blocks, a float.
    '''
    means_blocks=[np.mean([ x[f] for f in ind]) for ind in blocks]
    indice=np.argsort(means_blocks)[int(np.floor(len(means_blocks)/2))]
    return means_blocks[indice],indice

def mom(x,K):
    blocks=blockMOM(K,x)
    return MOM(x,blocks)[0]


def huber(x,c,T=100):
    mu=np.median(x)
    def psisx(x,c):
        return 1 if np.abs(x)<c else (2*(x>0)-1)*c/x

    def weight(x,mu,c):
        if x-mu==0:
            return 1
        else:
            return psisx(x-mu,c)
    for t in range(T):
        w=[weight(xx,mu,c) for xx in x]
        mu=np.sum(np.array(w)*x)/np.sum(w)
    return mu
