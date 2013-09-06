from numpy import *
from numpy.linalg import *

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

def costJ(Y,X,Theta,lam):
    '''
    the cost function J
    Y - an nm x nu prediction matrix
    X - an n x nm array whose columns are the feature vectors x^(i)
    Theta - an n x nu array whose columns are the weight vectors theta^(j)
    lam - the regularization constant lambda
    '''
    #we make an nm x nu matrix of zeros
    #fill in the cost of  each (i,j) pair for which we have a ranking
    #then get the total cost by adding all the entries
    #that is, summing over all (i,j) pairs for which we have rankings
    #since the missing values are 0 in this matrix, these do not
    #affect the answer
    #we form the unregularized cost matrix and add the regularization
    #terms at the end
    nm=Y.shape[0] #number of movies
    nu=Y.shape[1] #number of users
    matJ=zeros(nu*nm)
    matJ=matJ.reshape(nm,nu)
    for i in range(nm):
        for j in range(nu):
            if Y[i,j]!=-99.9: #Y is an array of FLOATS
                matJ[i,j]=1/float(2)*(dot(Theta[:,j],X[:,i])-Y[i,j])**2
    #the regularization terms are the sums of the norm-squareds of the columns of the X and Theta matrices
    regX=dot(X,transpose(X))
    regTheta=dot(Theta,transpose(Theta))
    #regX and regTheta are n x n matrices whose diagonal entries are the norm-squareds of the columns of X and Theta
    #the regularization terms are multiples of the traces of these
    return sum(matJ)+lam/float(2)*(trace(regX)+trace(regTheta))


def gradCostJ(Y,X,Theta,lam):
    '''
    the gradient of the cost function
    Y - an nm x nu predictions matrix
    X - an n x nm array whose columns are the feature vectors x^(i)
    Theta - an n x nu array whose columns are the weight (user preference) vectors theta^(j)
    lam - the regularization constant lambda
    '''
    nm=Y.shape[0]
    nu=Y.shape[1]
    n=X.shape[0]
    gradMatX=zeros(n*nm*nu)
    gradMatTheta=zeros(n*nm*nu)
    gradMatX=gradMatX.reshape(nm,n,nu)
    gradMatTheta=gradMatTheta.reshape(nu,n,nm)
    for i in range(nm):
        for k in range(n):
            for j in range(nu):
                if Y[i,j]!=-99.9: #Y an array of floats
                    gradMatX[i,k,j]=(dot(Theta[:,j],X[:,i])-Y[i,j])*Theta[k,j]
                    gradMatTheta[j,k,i]=(dot(Theta[:,j],X[:,i])-Y[i,j])*X[k,i]
    #summing over k gives the unregularized gradient matrices
    gradX=sum(gradMatX,axis=2)
    gradTheta=sum(gradMatTheta,axis=2)
    #we now add the regularization term
    for k in range(n):
        for i in range(nm):
            gradX[i,k]=gradX[i,k]+lam*X[k,i]
        for j in range(nu):
            gradTheta[j,k]=gradTheta[j,k]+lam*Theta[k,j]
    #it will be convenient to output both the gradient matrices and the gradient vector
    #we output the transposes of the gradient matrices so they are compatible with
    #the matrices X and Theta: this makes updating in gradient descent easy
    gradJx=transpose(gradX)
    gradJt=transpose(gradTheta)
    #we now reshape the gradient matrices into long vectors
    gradX=gradX.reshape(nm*n)
    gradTheta=gradTheta.reshape(nu*n)
    #and stick them together to make the gradient vector
    gradCost=concatenate((gradX,gradTheta))
    return gradJx,gradJt,gradCost


def descentTest(Y,n,al,lam,X=None,Theta=None):
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
    #gradJX and gradJT are the matrix components of the gradient matrix
    #grad J is the gradient vector
    gradJX,gradJT,gradJ=gradCostJ(Y,X,Theta,lam)
    numIter=0
    J=array([costJ(Y,X,Theta,lam)])
    count=zeros(1,dtype=int)
    while numIter<10:
        X=X-al*gradJX
        Theta=Theta-al*gradJT
        numIter+=1
        gradJX,gradJT,gradJ=gradCostJ(Y,X,Theta,lam)
        J=concatenate((J,array([costJ(Y,X,Theta,lam)])))
        count=concatenate((count,array([numIter])))
        print J[-1], count[-1]
    #print 'converges in %d iterations' %numIter
    return X,Theta,J,count

def gradDescent(Y,n,al,lam,error,X=None,Theta=None):
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
    #gradJX and gradJT are the matrix components of the gradient matrix
    #grad J is the gradient vector
    gradJX,gradJT,gradJ=gradCostJ(Y,X,Theta,lam)
    numIter=0
    J=array([costJ(Y,X,Theta,lam)])
    count=zeros(1,dtype=int)
    while dot(gradJ,gradJ)>error and numIter<400:
        X=X-al*gradJX
        Theta=Theta-al*gradJT
        numIter+=1
        gradJX,gradJT,gradJ=gradCostJ(Y,X,Theta,lam)
        J=concatenate((J,array([costJ(Y,X,Theta,lam)])))
        count=concatenate((count,array([numIter])))
        print J[-1], count[-1] #toggle here
    print 'GradJ is of size ',dot(gradJ,gradJ)
    print 'converges in %d iterations' %numIter
    return X,Theta,J,count


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
                            


###Try combining Cost and Grad functions.
###Question: should we recenter the scores so they run 0,...,4 instead of 1,...,5?
###Parameters to better understand: n, lambda
###parameters to fiddle with more: alpha, error
###Making train/test sets and maybe cross validation? May see how better to adjust lambda
###for test set: add rows of unranked movies, compare to actual rankings
###Q: are there users who have rated many movies? tastemakers?
###Q: are there movies that are very telling? Connect unrelated movies?
    

##scatter plot notes 200x200:
##cluster=findNear(Q,movies,-0.372,-0.369,-0.077,-0.073)
##['Faster Pussycat! Kill! Kill! (1965)', 'Brother Minister: The Assassination of Malcolm X (1994)', 'Theodore Rex (1995)', 'Horseman on the Roof, The (Hussard sur le toit, Le) (1995)', 'Maya Lin: A Strong Clear Vision (1994)']
##out1=findNear(Q,movies,-1,1,3,4)
##["Breakfast at Tiffany's (1961)"]
##out2=findNear(Q,movies,-4,-3,2,3)
##['Sound of Music, The (1965)']
##out3=findNear(Q,movies,-3,3,-3,-1.5)
##['Hoop Dreams (1994)', 'Star Wars (1977)', 'Priest (1994)', 'Ace Ventura: Pet Detective (1994)', 'Hot Shots! Part Deux (1993)', 'Jurassic Park (1993)', 'So I Married an Axe Murderer (1993)', 'Kids in the Hall: Brain Candy (1996)', 'Ghost and the Darkness, The (1996)', 'Return of the Pink Panther, The (1974)', 'Wrong Trousers, The (1993)', 'Empire Strikes Back, The (1980)', 'Return of the Jedi (1983)', 'Terminator, The (1984)']
##outCluster=findNear(Q,movies,1.8,2.4,0.8,2.2)
##['Ed Wood (1994)', 'Pulp Fiction (1994)', 'Welcome to the Dollhouse (1995)', 'Lone Star (1996)', 'Glengarry Glen Ross (1992)']
##out4=findNear(Q,movies,2.65,2.95,-0.8,-0.5)
##['Swingers (1996)']




###TO DO: make the predictionMatrix output the mean centered predictions.
### mean center+feature scaling 
###Question: should we recenter the scores so they run 0,...,4 instead of 1,...,5?
###Parameters to better understand: n, lambda
###parameters to fiddle with more: alpha, error
###Making train/test sets and maybe cross validation? May see how better to adjust lambda
###for test set: add rows of unranked movies, compare to actual rankings
###Q: are there users who have rated many movies? tastemakers?
###Q: are there movies that are very telling? Connect unrelated movies?

    
