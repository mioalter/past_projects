from numpy import *

def knn_search(x,D,K):
    '''
    x - a row vector whose K nearest neighbors we want to find
    D - data array in which each row is a data vector
    so the number of rows is the number of data points and
    the number of columns is the dimension of the data space (number of features)
    K - the "K" of "K-nearest neighbors"
    '''
    ndata=D.shape[0]    #number of rows in D = number of data points
    dim=D.shape[1]      #number of columns of D = dimension of data space
    if K>ndata:         #if K happens to be larger than ndata, set K to ndata
        K=ndata
    sqd = ((D-x)**2).sum(axis=1)
    #Let x_i be the ith row of D, then D-x is an array whose ith row is
    #the vector x_i-x, squaring is entrywise (not matrix multiplication)
    #then summing over columns (axis=1) produces a 1 x ndata array whose ith entry is
    #the square of the distance from x_i to x.
    ind=argsort(sqd)    #produces an array of the indices of sqd in increasing order of value
    return ind[:K]      #returns the first K indices in ind = the indices of the K nearest x_i

def knn_classifier(x,D,labels,K):
    '''
    x - vector to classify
    D - data array, each row is a data point, number of columns is dimension of data space
    labels - a 1 x ndata array whose ith entry is the class of the ith data point
    K - the "K" of "knn"
    '''
    nearest=knn_search(x,D,K)
    #finds indices of K nearest neighbors
    counts=bincount(labels[nearest])
    #makes an array whose ith entry is the number of times i appears
    #in the array of labels of the neighbors
    return argmax(counts),nearest
    #argmax returns the label which appears the greatest number of times in counts
    #so the nearest neighbors vote and the majority wins

train=genfromtxt("train.csv",dtype=int,delimiter=',',skip_header=1)

def digit_recognizer(train_size,test_size,K):
    '''
    train_size - the size of the training set
    test_size - the size of the test set
    K - number of nearest neighbors to use
    Both train and test sets are taken from the MNIST training set,
    the first from rows 0-39999 and the second from the last 2000 rows.
    This way we have the answers for the test vectors and we can check our prediction.
    '''
    D=train[:train_size,1:]
    #the data array
    labels=train[:train_size,0]
    #the labels are the 0th column
    test=train[40000:40000+test_size,1:]
    #the test vectors are taken from the last 2000 rows
    answers=train[40000:40000+test_size,0]
    #the correct answers from the label column in the original file
    predictions=zeros(test_size,dtype=int)
    #the array of predictions, we start with zeros and fill in the answers in the next loop
    for i in range(test_size):
        x=test[i]
        predict,neig=knn_classifier(x,D,labels,K)
        predictions[i]=predict
    #we have now filled in the predictions array with our predictions
    evaluation=absolute(answers-predictions)
    #an array of non-negative numbers: the 0s are where our predictions were correct,
    #the nonzero numbers are where we were wrong
    count=bincount(evaluation)
    #we count the number of times each value appears in the evaluations array.
    #we are only interested in the number of zeros = the number of correct predictions
    score=float(count[0])/test_size*100
    #this is the percentage of correct answers.
    return 'For a training set of %d, %d test vectors, and K=%d, the percentage correct is %f'%(train_size,test_size,K,score)
                                    
