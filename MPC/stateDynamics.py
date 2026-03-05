
import numpy as np
import scipy.linalg as la

def Vec(X):
    """
    Vectorize a matrix X (column-major ordering).
    for example:
      input shape: (a, b)
      output shape: (a*b, 1)
    """
    return X.reshape(-1, 1, order='F') 

def invVec(X):
    """
    Inverse vectorization of a matrix X (column-major ordering).
    for example:
      input shape: (a*b, 1)
      output shape: (a, b)
    """
    n = int(np.sqrt(X.shape[0]))
    return X.reshape(n, n, order='F') # reshape to (n,n) matrix
  
  
class StateDynamics(object):
  
  def __init__(self, n1, n2, p, W, A_E, A_S, B_S):
    """
    Initializes the class
    """
    self.x_E = np.zeros((n1,1)) # earth vector
    self.x_S = np.zeros((n2,1)) # sensor vector
    
    n = n1 + n2 # state size
    self.n1 = n1 # earth state size
    self.n2 = n2 # sensor state size
    self.n = n # state size
    self.x = np.vstack((self.x_E, self.x_S)) # state vector
    self.u = np.zeros((p,1)) # control input vector
    
    assert W.shape[0] == n, "W must be a square matrix of size n x n"
    assert W.shape[1] == n, "W must be a square matrix of size n x n"
    self.W = W # covariance matrix for process noise w
    self.p = p # control input size
    
    self.A = np.zeros((n,n)) # state transition matrix
    self.A[:n1, :n1] = A_E
    self.A[n1:, n1:] = A_S
    
    
    self.B = np.zeros((n,p)) # control input matrix
    self.B[n1:, :p] = B_S # shape (n2,p)  
    self.t = 0 # time step
    self.trajectory = [] # trajectory of the system
    self.trajectory.append([self.x, self.u]) # append initial state and control input to trajectory
  
  def get_earth_state_size(self): 
    return self.n1
  
  def get_sensor_state_size(self):
    return self.n2
  
  def get_state_size(self):
    return self.n
 
  def get_input_size(self):
    return self.p
 
  def get_W(self):
    '''
    covariance matrix for process noise w
    '''
    return self.W
  
  def get_A(self):
    '''
    state transition matrix
    '''
    return self.A
  
  def get_B(self):
    '''
    control input matrix
    '''
    return self.B
  
  def get_x_E(self):
    '''
    current earth state vector
    '''
    return self.x_E
  
  def get_x_S(self):
    '''
    current sensor state vector
    '''
    return self.x_S
  
  def get_x(self):
    '''
    current state vector
    '''
    return self.x 
  
  def set_u(self, u):
    '''
    set control input vector
    '''
    self.u = u
  
  def get_u(self):
    '''
    current control input vector
    '''
    return self.u
  
  def get_traj_history(self):
    '''
    trajectory of the system
    '''
    return self.trajectory
  
  def get_w(self):
      '''
      process noise w (drawn fresh from W each time)
      '''
      omega = np.linalg.cholesky(self.W)
      rng_noise = np.random.default_rng()  # always fresh, unseeded
      noise = omega @ rng_noise.standard_normal((self.n, 1))  # shape (n,1)
      self.w = noise  # store process noise
      return noise
  
  def forward(self):
    '''
    Forward kinematics of the system.
    '''
    w = self.get_w() # process noise w
    x1 = self.A @ self.x + self.B @ self.u + w # shape (n,1)
    self.x = x1 # update state vector
    self.x_E = x1[:self.n1] # update earth state vector
    self.x_S = x1[self.n1:] # update sensor state vector
    self.t += 1
    self.trajectory.append([self.x, self.u]) # append current state and control input to trajectory
    return self.t
  
# augmented system
  def get_z(self):
    '''
    get current augmented state vector
    '''
    x = self.x # shape (n,1)
    z1 = x # shape (n,1)
    z2 = Vec(x @ x.T) # shape (n^2, 1)
    z = (np.concatenate([z1.T, z2.T], axis=1)).T # shape (n+n^2, 1)
    return z, z1, z2
  
  def get_mu_tilde(self): # mu_tilde
    mu = self.B @ self.u # "mu" term, shape (n,1)
    Sigma = self.W # shape (n,n)
    term1 = mu # shape (n,1)
    term2 = Vec(mu @ mu.T + Sigma) # shape (n^2, 1) 
    mu_tilde = np.vstack((term1, term2)) # shape (n+n^2, 1)
    return mu_tilde # shape (n+n^2, 1)
  
  def get_Sigma_tilde(self):
    n = self.n # state size
    Sigma = self.W # shape (n,n)
    I_n = np.eye(n) # shape (n,n)
    I_n2 = np.eye(n**2) # shape (n^2,n^2)
    Mu = self.B @ self.u # shape (n,1)
    Phi = self.A # shape (n,n)
    X = self.x # shape (n,1)
    # print(f'n: {n}, Mu: {Mu.shape}, Sigma: {Sigma.shape}')
    Gamma = np.kron(I_n, (Mu + Phi @ X)) + np.kron((Mu + Phi @ X), I_n) # shape (n^2,n)
    
    Lambda = np.zeros((n**2, n**2))
    for i in range(n):
        for j in range(n):
            e_i = np.zeros((n,1))
            e_i[i] = 1
            e_j = np.zeros((n,1))
            e_j[j] = 1
            Lambda += np.kron(e_i @ e_j.T, e_j @ e_i.T) # shape (n^2,n^2)   
            
    Sigma11 = Sigma # shape (n,n)
    # print(f'Sigma: {Sigma.shape}, Gamma: {Gamma.shape}, Lambda: {Lambda.shape}')
    Sigma12 = Sigma @ Gamma.T # shape (n,n^2)
    Sigma21 = Gamma @ Sigma # shape (n^2,n)
    Sigma22 = Gamma @ Sigma @ Gamma.T + (I_n2 + Lambda) @ np.kron(Sigma, Sigma) # shape (n^2,n^2)
    
    Sigma_tilde = np.zeros((n+n**2, n+n**2), dtype=np.float64) # shape (n+n^2,n+n^2
    Sigma_tilde[:n, :n] = Sigma11
    Sigma_tilde[:n, n:] = Sigma12
    Sigma_tilde[n:, :n] = Sigma21
    Sigma_tilde[n:, n:] = Sigma22
    # print(f'sum sigma11: {np.sum(Sigma11)}, sum sigma12: {np.sum(Sigma12)}, sum sigma21: {np.sum(Sigma21)}, sum sigma22: {np.sum(Sigma22)}')
    return Sigma_tilde # shape (n+n^2,n+n^2)
  
  def get_A_tilde(self):
    '''
    Augmented state transition matrix; "Phi" term in function X = mu + Phi X + Omega epsilon
    '''
    n = self.n # state size
    Phi = self.A # state transition matrix, shape (n,n)
    mu = self.B @ self.u # shape (n,1)
    A11 = Phi # shape (n,n)
    A12 = np.zeros((n, n**2)) # shape (n,n^2)
    A21 = np.kron(mu, Phi) + np.kron(Phi, mu) # shape (n^2,n)
    A22 = np.kron(Phi, Phi) # shape (n^2,n^2)
    A_tilde = np.zeros((n+n**2, n+n**2)) # shape (n+n^2,n+n^2)
    A_tilde[:n, :n] = A11
    A_tilde[:n, n:] = A12
    A_tilde[n:, :n] = A21
    A_tilde[n:, n:] = A22
    return A_tilde.astype(np.float64) # shape (n+n^2,n+n^2)

  def get_B_tilde(self):
      """
      Expected to be derived by d mu_tilde / d u
      """
      B  = self.B                               # (n, p)
      u = self.u                               # (p, 1)
      mu  = B @ self.u                           # (n, 1)
      n, p = self.n, self.p
      I_n = np.eye(n)                        # (n, n)
      I_p = np.eye(p)                        # (p, p)
      term1 = B                                    # (n, p)
      term2 = np.kron(B, B) @ (np.kron(I_p, u) + np.kron(u, I_p)) # (n², p)
      return np.vstack((term1, term2), dtype=np.float64)   # (n + n², p)
  
  def get_w_tilde(self):
    w_tilde = np.zeros((self.n + self.n**2, 1)) # shape (n+n^2,1)
    w_tilde[:self.n] = self.w # shape (n,1)
    return w_tilde # shape (n+n^2,1)
  
class sensor(object):
  def __init__(self, C, M, V):
    """
    Initializes the class
    """
    # M shape - (m, n, n) where n is state size
    assert C.shape[0] == M.shape[0], "C and M must have the same length m"
    self.m = M.shape[0] # number of measurements
    self.n = C.shape[1]
    
    self.M = M.astype(np.float64) # covariance matrix for process noise w
    self.V = V.astype(np.float64) # covariance matrix for measurement noise v
    self.C = C.astype(np.float64) # measurement matrix, shape (m, n)
    
  def get_V(self):
    '''
    covariance matrix for process noise v
    '''
    return self.V # shape (m,m)
  
  def get_measA(self):
    '''
    measurement matrix 1; "A" term in Y = A + Bx + Σ e X.T C X + Dv
    Actualy, this matrix does not exist in our case, but we need it for the QKF, so we define it as a zero matrix.
    '''
    return np.zeros((self.m, 1)) # shape (m,1)

  def get_aug_measB(self):
    '''
    Augmented matrix of measurement matrix 2; B_tilde from "B" term in Y = A + Bx + Σ e X.T C X + Dv
    '''
    if self.M.ndim != 3:
        raise ValueError("M_stack must be (m, n, n) or list of (n, n) matrices")

    m, n = self.C.shape[0], self.C.shape[1]  # m = number of measurements, n = state size
    if self.M.shape[1:] != (n, n) or self.M.shape[0] != m:
        raise ValueError("Inconsistent shapes between C and M_stack")

    # build the right-hand block: each row i is vec(M[i].T)
    right_term = np.zeros((m, n**2)) # shape (m, n^2)
    for i in range(m):
        right_term[i] = Vec(self.M[i].T).squeeze() # shape (n^2,1) -> shape (n^2)

    # horizontal concatenation
    B_tilde = np.hstack((self.C, right_term)) # shape (m, n+n^2)
    # print (f'B_tilde shape: {B_tilde.shape}')
    return B_tilde   
  
  def measure(self, x):
    '''
    Measurement function
    '''
    term1 = self.C @ x # shape (m,1)
    term2 = np.zeros((self.m, 1)) 
    for i in range(self.m):
        e = np.zeros((self.m, 1))
        e[i] = 1
        term2 += e @ x.T @ self.M[i] @ x

    D = np.linalg.cholesky(self.V)
    rng_noise = np.random.default_rng()  # fresh, unseeded generator
    term3 = D @ rng_noise.standard_normal((self.m, 1))
    # print(f'term1: {term1.shape}, term2: {term2.shape}, term3: {term3.shape}')
    return term1 + term2 + term3

  
  def measure_pred(self, x_pred):
    '''
    Measurement function for predicted state
    '''
    term1 = self.C @ x_pred
    term2 = np.zeros((self.m, 1))
    for i in range(self.m):
        e = np.zeros((self.m, 1))
        e[i] = 1
        term2 += e @ x_pred.T @ self.M[i] @ x_pred
    return term1 + term2
  
  def g(self, x):
    term1 = self.C # B term in Y = A + Bx + Σ e X.T C X + Dv
    term2 = np.zeros((self.m, self.n))
    for i in range(self.m):
        e = np.zeros((self.m, 1))
        e[i] = 1
        term2 += e @ x.T @ self.M[i]
    return term1 + 2 * term2 # shape (m,n)
  
  def aug_measure(self, z):
    '''
    Measurement function for augmented state
    '''
    term1 = self.get_measA() # shape (m,1), which is a zero vector
    term2 = self.get_aug_measB() @ z # shape (m,1)
    D = np.linalg.cholesky(self.V) 
    rng_noise = np.random.default_rng()  # fresh, unseeded generator
    term3 = D @ rng_noise.standard_normal((self.m, 1)) # shape (m,1)
    return term1 + term2 + term3 # shape (m,1)
