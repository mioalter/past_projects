from numpy import *


def gradMatJ(theta,Data):
    '''
    computes the gradient of the cost function J at the point theta
    theta - an m+1-vector
    Data - an n x (m+1) array
    '''
    n=Data.shape[0]
    #number of data points = number of rows
    m=Data.shape[1]-1
    #dim of feature space
    y=Data[:,-1]
    #y is the last column of Data, the vector of labels
    gradMatrixJ=ones(n*(m+1))
    gradMatrixJ=gradMatrixJ.reshape(n,m+1)
    if m==1:
        for j in range(n):
            gradMatrixJ[j,0]=1/float(m)*(theta[0]+(theta[1]*Data[j,0:m])-Data[j,-1])
            gradMatrixJ[j,1]=1/float(m)*(theta[0]+(theta[1]*Data[j,0:m])-Data[j,-1])*Data[j,0]
    else:
        for j in range(n):
            for i in range(m+1):
                if i==0:
                    gradMatrixJ[j,i]=1/float(m)*(theta[0]+dot(theta[1:],Data[j,0:m])-Data[j,-1])
                else:
                    gradMatrixJ[j,i]=1/float(m)*(theta[0]+dot(theta[1:],Data[j,0:m])-Data[j,-1])*Data[j,i-1]
    #print 'gradMatrix is', gradMatrixJ                
    gradJ=sum(gradMatrixJ,axis=0)
    return gradJ

def costMatJ(theta,Data):
    '''
    computes the cost J as a function of theta
    '''
    n=Data.shape[0]
    #number of data points = number of rows
    m=Data.shape[1]-1
    #dim of feature space
    y=Data[:,-1]
    #y is the last column of Data, the vector of labels
    costMatrixJ=ones(n)
    if m==1:
        for j in range(n):
            costMatrixJ[j]=1/float(2*m)*(theta[0]+(theta[1]*Data[j,0:m])-Data[j,-1])**2
    else:
        for j in range(n):
            costMatrixJ[j]=1/float(2*m)*(theta[0]+dot(theta[1:],Data[j,0:m])-Data[j,-1])**2                           
    costJ=sum(costMatrixJ)
    return costJ


           
def lm(Data,alpha,error,theta=None):
    '''
    Data - an n x (m+1) array
    n rows
    each row of the form (x,y) where
    x is an m-vector of features, and
    y is the value
    alpha - the learning rate, a small real number
    error - how close the gradient should be to zero before stopping
    '''
    n=Data.shape[0]
    #number of data points = number of rows
    m=Data.shape[1]-1
    #dim of feature space
    y=Data[:,-1]
    #y is the last column of Data, the vector of labels
    if theta==None:
        theta=ones(m+1)
        #if theta is unspecified, initialize it to ones
    gradJ=ones(m)
    gradJ=gradMatJ(theta,Data)
    numIterations=0
    while dot(gradJ,gradJ)>error:
        theta = theta - alpha*gradJ
        gradJ=gradMatJ(theta,Data)
        numIterations+=1
    print 'hypothsis found in %d iterations'%numIterations
    return theta

def learnRate(Data,alpha,error,theta=None):
    '''
    computes the cost function
    '''
    n=Data.shape[0]
    #number of data points = number of rows
    m=Data.shape[1]-1
    #dim of feature space
    y=Data[:,-1]
    #y is the last column of Data, the vector of labels
    if theta==None:
        theta=ones(m+1)
        #if theta is unspecified, initialize it to ones
    gradJ=ones(m)
    gradJ=gradMatJ(theta,Data)
    numIterations=0
    numVec=numIterations
    Jvals=1.
    while dot(gradJ,gradJ)>error:
        theta = theta - alpha*gradJ
        gradJ=gradMatJ(theta,Data)
        numIterations+=1
        #print costMatJ(theta,Data)
        if numIterations==1:
            Jvals=array([costMatJ(theta,Data)])
            numVec=array([1])
        else:
            Jvals=concatenate((Jvals,array([costMatJ(theta,Data)])))
            numVec=concatenate((numVec,array([numIterations])))
    return numVec,Jvals
        

    

        
        
