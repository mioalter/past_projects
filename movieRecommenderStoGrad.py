from numpy import *
from numpy.linalg import *
from pylab import plot,scatter,show

######################
### Set-up ###########
######################

def loadData():
    data=genfromtxt('u.data',dtype=int,delimiter='\t')
    return data

def loadData1():
    data=genfromtxt('u1.base',dtype=int,delimiter='\t')
    return data

def loadData2():
    data=genfromtxt('u2.base',dtype=int,delimiter='\t')
    return data

def loadData3():
    data=genfromtxt('u3.base',dtype=int,delimiter='\t')
    return data

def loadData4():
    data=genfromtxt('u4.base',dtype=int,delimiter='\t')
    return data

def loadData5():
    data=genfromtxt('u5.base',dtype=int,delimiter='\t')
    return data

def loadTest1():
    data=genfromtxt('u1.test',dtype=int,delimiter='\t')
    return data

def loadTest2():
    data=genfromtxt('u2.test',dtype=int,delimiter='\t')
    return data

def loadTest3():
    data=genfromtxt('u3.test',dtype=int,delimiter='\t')
    return data

def loadTest4():
    data=genfromtxt('u4.test',dtype=int,delimiter='\t')
    return data

def loadTest5():
    data=genfromtxt('u5.test',dtype=int,delimiter='\t')
    return data

def loadMovies():
    movies=genfromtxt('u.item',dtype='string',delimiter='|')
    nm=movies.shape[0]
    movieList=movies[:,1]
    return movieList


def predictionMatrix(data,nu,nm):
    '''
    makes a prediction matrix for data with
    nu - # users
    nm - # movies ranked
    '''
    #the predictions matrix will be an nm x nu matrix whose (i,j)th entry
    #is the score that user j+1 gave to movie i+1
    Y=ones(nu*nm,dtype=int)
    Y=Y.reshape(nm,nu)
    Y=Y*-999
    #we will use -999 to denote missing values (unranked movies)
    #we initialize Y to a matrix of -999s and fill in the (movie,user)
    #pairs for which we have scores
    for i in range(len(data)):
        Y[data[i,1]-1,data[i,0]-1]=data[i,2]
    return Y

def mask(Y):
    '''
    Y - nm x nu matrix prediction matrix with missing values
    returns a list of non-missing entries of Y
    '''
    nm=Y.shape[0]
    nu=Y.shape[1]
    mask=array([],dtype=int)
    for i in range(nm):
        for j in range(nu):
            if Y[i,j]>-50:
                if len(mask)==0:
                    mask=array([i,j],dtype=int)
                else:
                    mask=vstack((mask,array([i,j])))
    return mask

    
####################################################################
####################################################################
## Preprocessing: row mean, mean normalizing + feature scaling #####
####################################################################
####################################################################

def rowMean(Y):
    '''
    Y - nm x nu INTEGER array
    returns an nm-vector whose ith entry is the average of the ith row of the matrix
    '''
    #we allow for Y to have missing values which we denote by -999
    nm=Y.shape[0]
    nu=Y.shape[1]
    total=zeros(nm)
    count=zeros(nm)
    for i in range(nm):
        for j in range(nu):
            if Y[i,j]!=-999:
                total[i]+=Y[i,j]
                count[i]+=1
    mean=total
    for i in range(nm):
        if count[i]!=0:
            mean[i]=mean[i]/float(count[i])
    mean=mean.reshape(nm,1)
    return mean

def meanCenter(Y):
    '''
    Y - nm x nu INTEGER array
    returns Y mean centered, leaves missing values as -999
    '''
    nm=Y.shape[0]
    nu=Y.shape[1]
    Y=Y.reshape(nm,nu)
    M=rowMean(Y).reshape(nm)
    Z=ones(nm*nu)
    Z=Z.reshape(nm,nu)
    Z=Z*(-99.9)
    for i in range(nm):
        for j in range(nu):
            if Y[i,j]!=-999:
                Z[i,j]=float(Y[i,j])-M[i]
    return Z
    
def prePCA(X):
    '''
    X - n x k matrix
    X is a matrix of data vectors on which we will perform PCA
    so we do not allow for missing values
    returns a mean normalized and feature scaled matrix
    '''
    n=X.shape[0]
    k=X.shape[1]
    Sigma=zeros(n)
    for i in range(n):
        Sigma[i]=std(X[i,:])
    X=X-rowMean(X)
    for i in range(n):
        for j in range(k):
            X[i,j]=X[i,j]/float(Sigma[i])
    return X
    
###############################################################
###### Computions: use YM=meanCenter(Y) #######################
###############################################################


data1=loadData1()
Y=predictionMatrix(data1,943,1682)
Y=Y[0:300,0:300]
M=rowMean(Y)
YM=meanCenter(Y)
t=loadTest1()
T=predictionMatrix(t,943,1682)
T=T[0:300,0:300]



def costJ(Y,X,Theta,lam,i,j):
    '''
    the cost function J
    Y - an nm x nu prediction matrix
    X - an n x nm array whose columns are the feature vectors x^(i)
    Theta - an n x nu array whose columns are the weight vectors theta^(j)
    lam - the regularization constant lambda
    i - index in range(nm)
    j - index in range(nu)
    '''
    nm=Y.shape[0] #number of movies
    nu=Y.shape[1] #number of users
    cost=1/float(2)*(dot(Theta[:,j],X[:,i])-Y[i,j])**2+lam/float(2)*(dot(X[:,i],X[:,i])+dot(Theta[:,j],Theta[:,j]))
    return cost

def stoGrad(Y,X,Theta,lam,i,j):
    '''
    the gradient of the cost function
    Y - an nm x nu predictions matrix
    X
    Theta
    lam - the regularization constant lambda
    i - index in range(nm)
    j - index in range(nu)
    '''
    n=X.shape[0]
    FX=zeros(n)
    FT=FX
    for k in range(n):
        FX[k]=(dot(Theta[:,j],X[:,i])-Y[i,j])*Theta[k,j]+lam*X[k,i]
        FT[k]=(dot(Theta[:,j],X[:,i])-Y[i,j])*X[k,i]+lam*Theta[k,j]
    FX=FX.reshape(n,1)
    FT=FT.reshape(n,1)
    return FX,FT


def sgTest(Y,n,al,lam,X=None,Theta=None):
    '''
    Y - a predictions matrix
    n - the number features (dimension of feature space) to use
    al - the learning rate alpha, a small real number
    lam - the regularization constant lambda
    error - how close the gradient must be to zero before stopping
    returns matrices X and Theta of minimizing feature vectors and weight (user preference) vectors
    and vectors J and count with the cost J for each iteration
    '''
    nm=Y.shape[0]
    nu=Y.shape[1]
    #if X and Theta are unspecified, initialize them to random values.
    if X==None:
        X=random.rand(n,nm)
    if Theta==None:
        Theta=random.rand(n,nu)
    train=mask(Y)
    i=train[-1][0]
    j=train[-1][1]
    J=array([costJ(Y,X,Theta,lam,i,j)])
    FX,FT=stoGrad(Y,X,Theta,lam,i,j)
    X_old=X
    Theta_old=Theta
    X=X-al*FX
    Theta=Theta-al*FT
    numIter=0
    count=zeros(1,dtype=int)
    l=len(train)
    r=numIter%l
    while numIter<20:
        s=train[r]
        i=s[0]
        j=s[1]
        J=concatenate((J,array([costJ(Y,X,Theta,lam,i,j)])))
        FX,FT=stoGrad(Y,X,Theta,lam,i,j)
        X_old=X
        Theta_old=Theta
        X=X-al*FX
        Theta=Theta-al*FT
        numIter+=1
        r=numIter%l
        count=concatenate((count,array([numIter])))
        print J[-1]
    print 'converges in %d iterations' %numIter
    return X,Theta,J,count


def sgDescent(Y,n,al,lam,step,iterations,X=None,Theta=None):
    '''
    Y - a predictions matrix
    n - the number features (dimension of feature space) to use
    al - the learning rate alpha, a small real number
    lam - the regularization constant lambda
    error - how close the gradient must be to zero before stopping
    returns matrices X and Theta of minimizing feature vectors and weight (user preference) vectors
    and vectors J and count with the cost J for each iteration
    '''
    nm=Y.shape[0]
    nu=Y.shape[1]
    #if X and Theta are unspecified, initialize them to random values.
    if X==None:
        X=random.rand(n,nm)
    if Theta==None:
        Theta=random.rand(n,nu)
    train=mask(Y)
    i=train[-1][0]
    j=train[-1][1]
    J=array([costJ(Y,X,Theta,lam,i,j)])
    Jav=J
    FX,FT=stoGrad(Y,X,Theta,lam,i,j)
    X_old=X
    Theta_old=Theta
    X=X-al*FX
    Theta=Theta-al*FT
    numIter=0
    count=zeros(1,dtype=int)
    countAv=count
    l=len(train)
    r=numIter%l
    while numIter<=iterations: ##sum((X-X_old)**2)+sum((Theta-Theta_old)**2)>error:
        s=train[r]
        i=s[0]
        j=s[1]
        J=concatenate((J,array([costJ(Y,X,Theta,lam,i,j)])))
        if numIter>=step and numIter%step==0:
            print numIter
            Jav=concatenate((Jav,array([average(J[numIter-step:numIter])])))
            countAv=concatenate((countAv,array([numIter])))
        FX,FT=stoGrad(Y,X,Theta,lam,i,j)
        X_old=X
        Theta_old=Theta
        X=X-al*FX
        Theta=Theta-al*FT
        numIter+=1
        count=concatenate((count,array([numIter])))
        r=numIter%l
        ##al=float(c)/(numIter+c)*al
    ##print 'converges in %d iterations' %numIter
    return X,Theta,Jav,countAv


#######################################################
#######  Evaluation   #################################
#######################################################
### Add row mean and compute again with original Y ####
#######################################################

def predictionError(Y,Z):
    '''
    Y - the original prediction matrix
    Z - the prediction matrix generated by us
    '''
    nm=Y.shape[0]
    nu=Y.shape[1]
    error=zeros(nm*nu)
    error=error.reshape(nm,nu)
    count=0
    for i in range(nm):
        for j in range(nu):
            if Y[i,j]!=-999:
                error[i,j]=abs(Y[i,j]-Z[i,j])
                count+=1
    total=sum(error)
    err=total/float(count)
    return err
                
def rankRound(Z):
    '''
    Z - nm x nu matrix of generated predictions
    returns the corresponding integer matrix
    '''
    nm=Z.shape[0]
    nu=Z.shape[1]
    W=zeros(nm*nu,dtype=int)
    W=W.reshape(nm,nu)
    for i in range(nm):
        for j in range(nu):
            if abs(int(Z[i,j])-Z[i,j])>0.5:
                W[i,j]=int(Z[i,j])+1
            else:
                W[i,j]=int(Z[i,j])
    return W
                
def errorCount(Y,W):
    '''
    Y - nm x nu prediction matrix (with missing values)
    W - integer prediction matrix generated by us
    computes the number of misrankings
    '''
    nm=Y.shape[0]
    nu=Y.shape[1]
    count=0
    total=0
    for i in range(nm):
        for j in range(nu):
            if Y[i,j]!=-999:
                total+=1
                if Y[i,j]-W[i,j]!=0:
                    count+=1
    return count,total

###################################
#### Analyisis ####################
###################################
        
def pca(X):
    '''
    X - a n x k matrix
    the columns of X are k n-dimensional data vectors
    returns a 2 x k matrix whose columns are the images
    of the data vectors in R^2
    '''
    n=X.shape[0]
    k=X.shape[1]
    U=svd(X)[0]
    #svd gives X=UDV^T so for Sigma=XX^T, Sigma=UD^2U^T
    #where U is the orthogonal matrix of eigenvectors of Sigma
    #svd puts the eigenvectors in order of decreasing eigenvalue
    #we want the eigenvectors corresponding to the largest
    #eigenvalues so we take the first two columns of U
    Ured=U[:,0:2] #Ured n x 2 matrix of first two eigenvectors
    P=transpose(Ured) #2 x n matrix R^n --> R^2
    Z=dot(P,X)
    #matrix whose columns are the projections of the
    #data vectors (columns of X) onto R^2
    return Z

def knn_search(x,X):
    '''
    x - an n-vector (the feature vector of a movie)
    X - an n x nm array, columns are feature vectors of movies
    '''
    n=X.shape[0]
    nm=X.shape[1]
    x=x.reshape(3,1)
    sqd=sum((X-x)**2,axis=0)
    ind=argsort(sqd)
    return ind[:5]

def movieRec(x,X,movies):
    '''
    x - n-vector (movie you like)
    X - n x nm array (columns are feature vectors of movies)
    movies - nm-vector of movie titles
    '''
    nearest=knn_search(x,X)
    print 'Since you like ',movies[nearest[0]]
    print 'You might also like '
    print movies[nearest[1]],', '
    print movies[nearest[2]],', '
    print movies[nearest[3]], 'and, '
    print movies[nearest[4]]
    return

def findNear(Q,movies,x0,x1,y0,y1):
    '''
    Q - 2 x k array to plot with pca
    returns movies whose projections lie
    in the rectangle [x0,x1] x [y0,y1]
    '''
    k=Q.shape[1]
    nearby=[]
    for i in range(k):
        if x0< Q[0,i] < x1 and y0< Q[1,i] <y1:
            nearby.append(movies[i])
    return nearby
                            

###TO DO:in sgDescent, compute (i,j)th cost, plot every 1000 iterations.
###TRY:gradually descreasing learning rate by setting alpha=c1/(numIter+c2) so that sgd converges
