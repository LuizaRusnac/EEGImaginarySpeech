import numpy as np
import numpy.matlib
import preprocessing

"""
This module computes the Adaptive filter Least Mean Square using Mean Squared Error algoritm

This module contains:

my_lms - filter the 1D x signal using LMS adaptive filter

my_lms_onerec - filter the x 2D signal ([nr. channels x nr. samples]) using LMS adaptive filter

my_lms_data - filter the x 3D signal ([nr.records x nr. channels x nr. samples]) using LMS adaptive filter

"""

def sgn_norm(sgn):
    return (sgn-sgn.min())/(sgn.max()-sgn.min())

def my_lms(x, d, L = 100, mu = 0.01, flag = 0, repeat = 0):
    """
    This function filter the x 1D signal using LMS adaptive filter.

    Input data:
        x - 1D signal corresponding to one channel of EEG. Dimension: [1 x nr. samples]
        d - 1D signal corresponding to EOG. Dimension: [1 x nr. samples]
        L - number of points of the filter. DEFAULT = 100
        mu - the learning rate. DEFAULT = 0.01
        flag - may take values of 0 and 1. If flag = 0 the signal remains the same. If flag = 1 the signal is normalized 
        between [0,1]. DEFAULT = 0
        repeat - the number of times to repreat the signals in order to obtain better ceofficients. DEFAULT = 0

    IMPORTANT: The dimension of x ust be the same of d!!!
            Number of samples of x must be higher than L!!!

    Output data:
        y - the signal estimating the reference signal (EOG)
        e - the difference between the signal x and the estimated reference signal y (the desired signal)
        w - coeficients of the filter

    """

    dim = len(x)

    dimd = len(d)

    if dim!=dimd:
        raise AttributeError("The length of x MUST be the same as d!")

    if dim <= L:
        raise AttributeError("The length of x MUST be the higher than L value!")

    x = x.reshape(dim,1)
    d = d.reshape(dimd,1)

    if flag==1:
        x = sgn_norm(x)
        d = sgn_norm(d)
    
    if repeat > 0:
        d = np.concatenate((d, np.tile(d,(repeat,1))))
        x = np.concatenate((x, np.tile(x,(repeat,1))))

    w = np.zeros((L,1))
    y = np.zeros(d.shape)
    y[:L] = d[:L]
    e = np.zeros(x.shape)
    e[:L] = x[:L]

    for n in range(L,len(x)):
        y[n] = d[n-L:n].T @ w
        e[n] = x[n] - y[n]
       
        dw = mu * e[n] * d[n-L:n]
        w = w + dw  

        y[n] = d[n-L:n].T @ w 
        e[n] = x[n] - y[n]

    if repeat>0:
        return y[-dim:], e[-dim:], w
    else:
        return y, e, w

def my_lms_onerec(X, d, L = 100, mu = 0.01, flag = 0, repeat = 0):
    """
    This function filter the x 2D signal ([nr. channels x nr. samples]) using LMS adaptive filter.

    Input data:
        x - 2D signal corresponding to one EEG record. Dimension: [nr. channels x nr. samples]
        d - 1D signal corresponding to EOG. Dimension: [1 x nr. samples]
        L - number of points of the filter. DEFAULT = 100
        mu - the learning rate. DEFAULT = 0.01
        flag - may take values of 0 and 1. If flag = 0 the signal remains the same. If flag = 1 the signal is normalized 
        between [0,1]. DEFAULT = 0
        repeat - the number of times to repreat the signals in order to obtain better ceofficients. DEFAULT = 0

    Output data:
        y - the signal estimating the reference signal (EOG)
        e - the difference between the signal x and the estimated reference signal y (the desired signal)
        w - coeficients of the filter

    """

    if len(X.shape)!=2:
        raise AttributeError("X has %d dimensions. It has to be 2D"%len(X.shape))

    if len(d.shape)!=1:
        raise AttributeError("d has %d dimensions. It has to be 1D"%len(d.shape))

    if X.shape[0] > X.shape[1]:
        X = X.T

    dim = X.shape

    dimd = len(d)

    d = d.reshape(1,dimd)

    if dim[0]!=dimd[0]:
        raise AttributeError("The number of x records MUST be the same as the number of d records!")

    if dim[1]!=dimd:
        raise AttributeError("The length of x MUST be the same as d!")

    if dim[1] <= L:
        raise AttributeError("The length of x MUST be the higher than L value!")

    X = preprocessing.sgnNorm(X)
    d = sgn_norm(d)

    if repeat > 0:
        d = np.tile(d,(1,repeat))
        X = np.tile(X,(1,repeat))

    dim2 = X.shape
    w = np.zeros((L,dim2[0]))
    y = np.zeros((dim2[0],dim2[1]))
    y[:,:L] = numpy.matlib.repmat(d[0,:L],dim2[0],1)
    e = np.zeros(X.shape)
    e[:,:L] = X[:,:L]

    for n in range(L,dim2[1]):
        y[:,n] = (d[0,n-L:n] @ w).T
        e[:,n] = X[:,n] - y[:,n]
       
        dw = mu * numpy.matlib.repmat(e[:,n].reshape(dim2[0],1),1,L) * numpy.matlib.repmat(d[0,n-L:n].reshape(1,L),dim[0],1)
        w = w + dw.T  

        y[:,n] = (d[0,n-L:n] @ w).T
        e[:,n] = X[:,n] - y[:,n]

    if repeat > 0:
        return y[:,-dim[1]:], e[:,-dim[1]:], w
    else:
        return y, e, w

def my_lms_data(X, d, L = 100, mu = 0.01, flag = 0, repeat = 0):
    """
    This function filter the x 3D signal ([nr.records x nr. channels x nr. samples]) using LMS adaptive filter.

    Input data:
        x - 3D signal corresponding to EEG records. Dimension: [nr. records x nr. channels x nr. samples]
        d - 2D signal corresponding to EOG records. Dimension: [nr. records x nr. samples]
        L - number of points of the filter. DEFAULT = 100
        mu - the learning rate. DEFAULT = 0.01
        flag - may take values of 0 and 1. If flag = 0 the signal remains the same. If flag = 1 the signal is normalized 
        between [0,1]. DEFAULT = 0
        repeat - the number of times to repreat the signals in order to obtain better ceofficients. DEFAULT = 0

    Output data:
        y - the signal estimating the reference signal (EOG)
        e - the difference between the signal x and the estimated reference signal y (the desired signal)
        w - coeficients of the filter

    """

    dim = X.shape
    xfilt = np.zeros(dim)
    yfilt = np.zeros(dim)
    wfilt = np.zeros((dim[0],L,dim[1]))

    dimd = d.shape

    if dim[2]!=dimd[1]:
        raise AttributeError("The length of x MUST be the same as d!")

    if dim[2] <= L:
        raise AttributeError("The length of x MUST be the higher than L value!")

    for i in range(dim[0]):
        print("LMS: Wait... It might takes several minutes! You are at %d/%d"%(i+1,dim[0]))
        Xrec = preprocessing.sgnNorm(X[i,:,:])
        drec = sgn_norm(d[i,:])

        drec = drec.reshape(1,dimd[1])

        if repeat > 0:
            drec = np.tile(drec,(1,repeat))
            Xrec = np.tile(Xrec,(1,repeat))

        dim2 = Xrec.shape
        w = np.zeros((L,dim2[0]))
        y = np.zeros((dim2[0],dim2[1]))
        y[:,:L] = numpy.matlib.repmat(drec[0,:L],dim2[0],1)
        e = np.zeros(Xrec.shape)
        e[:,:L] = Xrec[:,:L]

        for n in range(L,dim2[1]):
            y[:,n] = (drec[0,n-L:n] @ w).T
            e[:,n] = Xrec[:,n] - y[:,n]
           
            dw = mu * numpy.matlib.repmat(e[:,n].reshape(dim2[0],1),1,L) * numpy.matlib.repmat(drec[0,n-L:n].reshape(1,L),dim2[0],1)
            w = w + dw.T  

            y[:,n] = (drec[0,n-L:n] @ w).T
            e[:,n] = Xrec[:,n] - y[:,n]

        if repeat > 0:
            yfilt[i,:,:] = y[:,-dim[2]:]
            xfilt[i,:,:] = e[:,-dim[2]:]
            wfilt[i,:,:] = w
        else:
            yfillt[i,:,:] = y
            xfilt[i,:,:] = e
            wfilt[i,:,:] = w

    return yfilt, xfilt, wfilt