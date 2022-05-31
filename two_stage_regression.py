

import numpy as np
import numpy.linalg
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import chi
from collections import namedtuple


from sklearn.linear_model import Ridge

Params = namedtuple('Params', ['W_rff', 'b_rff', 'U', 'U_b', 'W_FE_F', 'b_FE_F', 'W_pred', 'b_pred', 'q_1'])

class RFF_Projection:
    
    def __init__(self, kernel_width, seed, nRFF, n_feat):
        rbf_sampler = RBFSampler(gamma=kernel_width, random_state=seed, n_components=nRFF)
        rbf_sampler.fit(np.zeros((1, n_feat)))
      
        self.W = rbf_sampler.random_weights_
        self.b = rbf_sampler.random_offset_
        self.nRFF = nRFF
        
    def project(self, x):
        return np.cos((x.T.dot(self.W) + self.b).T)*np.sqrt(2.)/np.sqrt(self.nRFF)

# calculates the kahtri-rao product of two matrices
def khatriRaoProduct(X,Y):
    XY = np.zeros((X.shape[0]*Y.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        XY[:,i] = np.kron(X[:,i], Y[:,i].T).reshape(X.shape[0]*Y.shape[0])
    return XY

# perform ridge regression to X from Y with ridge regression parameter lr
def ridgeRegressionBiased(X, Y, lr):
    
    ridge = Ridge(fit_intercept=True, alpha=lr, random_state=0, normalize=True, tol=1e-20)
    ridge.fit(Y.T,X.T)
    W = ridge.coef_
    b = ridge.intercept_.reshape((-1,1))
    
    return W, b

# split data into P, F, FS, Obs
def featurize(data, k):
    nFeat = data.shape[0]
    nData = data.shape[1]
    
    print(f"stacked = np.zeros(({nFeat}*(2*{k}+1),{nData}-2*{k}))")
    stacked = np.zeros((nFeat*(2*k+1),nData-2*k))
    for i in range(2*k+1):
        stacked[nFeat*i:nFeat*(i+1),:] = data[:, i:nData-(2*k-i)]
    
    P = stacked[:nFeat*k, :]
    F = stacked[nFeat*k:2*nFeat*k, :]
    Obs = stacked[k*nFeat:(k+1)*nFeat, :]
    FS = stacked[(k+1)*nFeat:, :]
    
    return (Obs, P, F, FS)
    
    
    
def svd_projection(x, y, nSvd, whiten=False):
    # Calculate Covariance Matrices
    mu_x = np.mean(x,axis=1).reshape((-1,1))
    mu_y = np.mean(y,axis=1).reshape((-1,1))
    if whiten:
        x = x - mu_x
        y = y - mu_y
    C = x.dot(y.T)/x.shape[1] # X KP Y
    # Calculate matrix of singular vectors
    U, S, V = sp.sparse.linalg.svds(C, nSvd)
    
    # whiten projection
    for i in range(S.size):
        if S[i] > 0:
            S[i] = 1/np.sqrt(S[i])
            
    W = U.dot(np.diag(S))
    b = -W.T.dot(x).mean()
    
    return W, b 

def two_stage_regression(raw_data, # initially is bag of words
                         data, #one-hot
                         kernel_width_Obs, kernel_width_P, kernel_width_F, 
                         seed, 
                         nRFF_Obs, nRFF_P, nRFF_F,
                         dim_Obs, dim_P, dim_F, 
                         reg_rate, 
                         obs_window):


    n_feat = data.shape[1] 
    n_data = data.shape[0]

    # generate features of history/future
    # This is not the features of observations
    # It is the features of the past, future, shifted future and current observations
    print("featurizing")
    Obs, P, F, FS = featurize(data.T, obs_window) # take the first tr

    print("Obs: ", Obs.shape) 
    print("F: ", F.shape) 
    print("FS: ", FS.shape) 
    print("P: ", P.shape) 

    
    # split raw data 
    #raw_F = raw_data.T[:,obs_window:n_data-obs_window]
    raw_FS = raw_data.T[:,obs_window+1:n_data-obs_window+1]
    #raw_P = raw_data.T[:,:n_data-obs_window-1]

    print("raw data: ", raw_data.shape)
    #print("raw_F: ", raw_F.shape) 
    print("raw_FS: ", raw_FS.shape)
    #print("raw_P: ", raw_P.shape) 

    # project into RBF space.
    # There is something called Hilbert space. Basically project the features into this space?
    # See last paragraph of Section 3.1 of PSRNN paper
    print("project into rff space")
    
    Obs_rff = Obs
    P_rff = P
    F_rff = F
    FS_rff = FS
    
    Obs_Proj = RFF_Projection(kernel_width_Obs, seed*1, nRFF_Obs, Obs_rff.shape[0]) # used for input embedding
    Obs_rff = Obs_Proj.project(Obs_rff) #Obs from featurize

    P_Proj = RFF_Projection(kernel_width_P, seed*2, nRFF_P, P_rff.shape[0])
    P_rff = P_Proj.project(P_rff) #P from featurize
    
    F_Proj = RFF_Projection(kernel_width_F, seed*3, nRFF_F, F_rff.shape[0])
    F_rff = F_Proj.project(F_rff) #F from featurize
    FS_rff = F_Proj.project(FS_rff) #FS from featurize


    # SVD is used to reduce the dimensionality of the RFF vectors
    # Question: To find why the order of X and Y are used
    # project the data onto top few singular vectors
    print('project onto svd')
    U_Obs, U_Obs_b = svd_projection(Obs_rff, P_rff, dim_Obs)
    Obs_U = U_Obs.T.dot(Obs_rff) # Project Obs_rff to a lower dimension
    print("Obs_U shape: ", Obs_U.shape) 

    U_P, U_P_b = svd_projection(P_rff, F_rff, dim_P, whiten=True)
    P_U = U_P.T.dot(P_rff) #todo
    print("P_U shape: ", P_U.shape) 
    
    U_F, U_F_b = svd_projection(F_rff, P_rff, dim_F, whiten=True)
    F_U = U_F.T.dot(F_rff) #todo
    FS_U = U_F.T.dot(FS_rff) # todo
    print("F_U shape: ", F_U.shape) 
    print("FS_U shape: ", FS_U.shape) 

    
    # calculate extended future from shifted future and observation
    # Question: Why can extended future be calculated from KR product of shifted future and obs
    print('extended future')
    FE_U = khatriRaoProduct(FS_U, Obs_U)
    print("FE_U: ", FE_U.shape) 

    # TWO STAGE REGRESSION
    # Two-stage least-squares regression uses instrumental variables that are uncorrelated 
    # with the error terms to compute estimated values of the problematic predictor(s)
    # (the first stage), and then uses those computed values to estimate a linear regression 
    # model of the dependent variable (the second stage).

    # Question: Why a regression can estimate these terms?

    # W = (Sum of phi_t+1 KP w_t KP n_t) (Sum of n_t KP phi_t)
    # phi_t+1 = RFF(F_S) = FS_U (after SVD)
    # w_t = RFF(o_t) = Obs_U
    # n_t = RFF(h_t) = P_U
    # Maybe khatri_rao product to get FE_U is to get phi_t+1 P w_t without the summation

    # Z = (Sum of w_t KP w_t KP n_t) (Sum of n_t KP phi_t)
    # w_t = Obs_U
    # n_t = P_U
    # Does this imply that F_U is w_t KP w_t ?? Doesn't make sense.
    
    # stage 1 regression
    # 
    print('stage 1')
    W_F_P_bias, b_F_P_bias = ridgeRegressionBiased(F_U, P_U, reg_rate) # To estimate second term of W?
    W_FE_P_bias, b_FE_P_bias = ridgeRegressionBiased(FE_U, P_U, reg_rate) # To estimate first term of W?
    
    # apply stage 1 regression to data to generate input for stage2 regression
    print('apply stage 1')
    E_F_bias = W_F_P_bias.dot(P_U) + b_F_P_bias # estimate F_U
    E_FE_F_bias = W_FE_P_bias.dot(P_U) + b_FE_P_bias # estimate FE_U
    
    # stage 2 regression
    # Learn how to go from F_U to FE_U
    # Can we assume this is W of equation 3?
    print('stage 2')
    W_FE_F, b_FE_F = ridgeRegressionBiased(E_FE_F_bias, E_F_bias, reg_rate)
    
    # calculate initial state
    # Equation 2
    # phi_t is elements of E_F_bias or F_U_hat
    print('apply stage 2')    
    q_1 = np.mean(E_F_bias,axis=1).reshape((-1,1))
    print("q_1 shape: ", q_1.shape)

    # perform filtering using learned model    
    # qt+1 = (W x3 qt) (Z x3 qt)^-1 x2 ot
    s = np.zeros((F_U.shape[0],F_U.shape[1]+1))
    s[:,0] = q_1.reshape((dim_P))
    for i in range(F_U.shape[1]):

        # Here, (W x3 qt)
        W = W_FE_F.dot(s[:,i]) + b_FE_F.reshape(-1)
        W = W.reshape((dim_F, dim_P))

        # Here (W x3 qt) x2 ot
        s[:,i+1] = W.dot(Obs_U[:,i])

        # Here multiply by (Z x3 qt)^-1 ???
        # Can we assume that Z was not estimated in 2S regression?
        # Instead, it is calculated from norm(s[:,i+1])
        # Paper said Z is a normalization tensor
        s[:,i+1] = s[:,i+1]/np.linalg.norm(s[:,i+1])
    s = s[:,1:]

    print("s: ", s.shape) 
    print("raw_FS: ", raw_FS.shape) 
    
    # regress from state to predictions

    # From last part of 3.2 (PSRNN paper), quote
    # however, in order to generalize to the continuous setting 
    # with RFF features we train a regression model to predict w_t from qt ??
    #
    # But, w_t in paper is RFF projected, here we are using raw_FS, so...?

    W_pred_list = []
    b_pred_list = []
    from sklearn.linear_model import LinearRegression
    for i in range(raw_FS.shape[0]):
        y = raw_FS[i].reshape(-1)
        linreg = LinearRegression()
        linreg.fit(s.T, y)
        W_pred_list.append(linreg.coef_.reshape(1,-1))
        b_pred_list.append(linreg.intercept_)
    W_pred = np.concatenate(W_pred_list, axis=0)
    b_pred = np.array(b_pred_list).reshape(-1,1)
   
    # F_raw_augmented = raw_F.flatten()
    # unq_labels = set(F_raw_augmented.tolist())

    # idx = -1
    # for i in range(n_feat):
    #     if i not in unq_labels:
    #         F_raw_augmented[idx] = i
    #         idx -= 1
            
    # y = raw_FS.reshape((raw_FS.size))

    # print("s.T: ", s.T.shape) #(99890, 20) 
    # print("y: ", y.shape) #(99890,)
    # logreg.fit(s.T,y)
    # W_pred = logreg.coef_
    # b_pred = logreg.intercept_.reshape((-1,1))

    print("-------")

    print("Obs_Proj.W: ", Obs_Proj.W.shape) 
    print("Obs_Proj.b: ", Obs_Proj.b.shape)
    print("U_Obs: ", U_Obs.shape) 
    print("W_FE_F: ", W_FE_F.shape)
    print("b_FE_F: ", b_FE_F.shape) 
    print("W_pred: ", W_pred.shape) 
    print("b_pred: ", b_pred.shape) 
    print("q_1: ", q_1.shape) 

  
    tsr_params = Params(Obs_Proj.W,
                       Obs_Proj.b,
                       U_Obs,
                       False,
                       W_FE_F,
                       b_FE_F,
                       W_pred,
                       b_pred,
                       q_1)

    return tsr_params