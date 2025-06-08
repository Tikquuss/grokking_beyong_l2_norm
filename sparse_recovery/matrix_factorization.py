import torch
import numpy as np
from numpy.linalg import svd
import cvxpy as cp
from tqdm import tqdm

import matplotlib.pyplot as plt

# # Set the working directory to the parent directory of your top-level package
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sparse_recovery.compressed_sensing import create_signal, create_orthonormal_basis, get_measures
from utils.svd import np_SVD
from utils.products import face_splitting_product_numpy
from utils.data_tensor_completion import get_matrices, data_tensor_completion_general

########################################################################################
########################################################################################

def np_vec(A): return A.reshape(A.shape[0]*A.shape[1], order='F')
def np_unvec(x, m, n):
    """Inverse of the vectorization operator : mn ---> (m, n)
    https://math.stackexchange.com/a/3122442/1020794"""
    Im, In = np.eye(m), np.eye(n)
    return np.kron(np_vec(In).T, Im) @ np.kron(In, np.expand_dims(x, 1))

########################################################################################
########################################################################################

def create_matrix(n_1, n_2, r, distribution="normal", U=None, V=None, scaler=None, seed=0):
    """
    Generate a matrix A_star of size n_1 x n_2 with rank r

    :param n_1: int, number of rows
    :param n_2: int, number of columns
    :param r: int, rank of the matrix
    :param distribution: str, distribution of the singular values
    :param U: np.ndarray, left singular vectors
    :param V: np.ndarray, right singular vectors
    :param scaler: float, scaling factor for the singular values
    :param seed: int, random seed
    
    :return: np.ndarray, matrix A_star
    """
    n = min(n_1, n_2)
    scaler = (1/np.sqrt(n)) if scaler is None else 1.0
    a_star, b_star = create_signal(n, s=r, distribution=distribution, Phi=None, scaler=scaler, seed=seed)
    b_star = np.abs(b_star)  # Sigular values need to be positives
    b_star = np.sort(b_star)[::-1]  # Sort in descending order
    Simga_star = np.zeros((n_1, n_2))
    Simga_star[:n, :n] = np.diag(b_star)  # Fill the diagonal with b*
    A_star = Simga_star
    if U is not None :
        A_star = U @ Simga_star
    if V is not None :
        A_star = A_star @ V.T

    #U, Sigma_star, V = np_SVD(A_star, type_svd='full') # (n_1, r), (r, r), (n_2, r)
    U, Sigma_star, VT = np.linalg.svd(A_star) # (n_1, n_1), (min(n_1, n_2),), (n_2, n_2)
    return A_star, U, Sigma_star, V

########################################################################################
########################################################################################

def get_matrices_UV(n_1, n_2, rank, symmetric=False, normalize=True, scale=None, seed=None, generator:torch.Generator=None):
    """
    A_star : (n_1, n_2)
    U_star : (n_1, n_1)
    Sigma_star : (min(n_1, n_2),)
    V_star : (n_2, n_2)
    """
    A_star, U_star, Sigma_star, V_star = get_matrices(n_1, n_2, rank, 1, symmetric, normalize, scale, seed, generator)
    return A_star.numpy(), U_star.numpy(), Sigma_star.numpy(), V_star.numpy()

def get_matrices_IndentityUV(n_1, n_2, r, distribution="normal", scaler=None, seed=None):
    """
    A_star : (n_1, n_2)
    U_star : (n_1, n_1)
    Sigma_star : (min(n_1, n_2),)
    V_star : (n_2, n_2)
    """
    # U = np.eye(n_1) # (n_1, n_1)
    # V = np.eye(n_2) # (n_2, n_2)
    U, V = None, None
    return create_matrix(n_1, n_2, r, U=U, V=V, scaler=scaler, seed=seed)

def get_matrices_QR_UV(n_1, n_2, r, distribution="normal", scaler=None, seed=None):
    """
    A_star : (n_1, n_2)
    U_star : (n_1, n_1)
    Sigma_star : (min(n_1, n_2),)
    V_star : (n_2, n_2)
    """
    U = create_orthonormal_basis(n_1, scaler=1.0) # (n_1, n_1), orthogonal
    V = create_orthonormal_basis(n_2, scaler=1.0) # (n_2, n_2), orthogonal
    # U = np.eye(n_1) # (n_1, n_1)
    # V = np.eye(n_2) # (n_2, n_2)
    return create_matrix(n_1, n_2, r, U=U, V=V, scaler=scaler, seed=seed)
    

########################################################################################
########################################################################################

def calculate_local_coherence(A, U_n1=None, V_n2=None):
    """
    Calculates the local coherence scores for a matrix A with respect to sets of vectors U_n1 and V_n2.

    Parameters:
    ----------
    A : np.ndarray of shape (n_1, n_2)
        The matrix for which to calculate local coherences.
    U_n1 : np.ndarray of shape (n_1, N_1), optional
        Set of N_1 vectors of dimension n_1 for left coherence calculation. If None, uses the canonical basis.
    V_n2 : np.ndarray of shape (n_2, N_2), optional
        Set of N_2 vectors of dimension n_2 for right coherence calculation. If None, uses the canonical basis.

    Returns:
    -------
    mu : np.ndarray of shape (N_1,)
        Local coherence values with respect to U_n1.
    nu : np.ndarray of shape (N_2,)
        Local coherence values with respect to V_n2.
    """
    n_1, n_2 = A.shape

    # Use canonical bases if U_n1 or V_n2 are not provided
    if U_n1 is None :
        U_n1 = np.eye(n_1) # (n_1, N_1) = (n_1, n_1)
    if V_n2 is None :
        V_n2 = np.eye(n_2) # (n_2, N_2) = (n_2, n_2)

    # SVD
    Sigma, U, V = np_SVD(A, type_svd='compact') # (r, r), (n_1, r), (n_2, r)
    r = Sigma.shape[0]

    mu = (n_1/r) * np.linalg.norm(U_n1.T @ U, axis=1)**2 # (N_1,)
    nu = (n_2/r) * np.linalg.norm(V_n2.T @ V, axis=1)**2 # (N_2,)

    P = mu[:,None] + nu[None,:] # P_{ij} = mu_i + nu_j
    #P = mu.view(-1, 1) + nu.view(1, -1) #  P_{ij} = mu_i + nu_j

    # Return local coherences value
    return mu, nu, P # (N_1,), (N_2,)

########################################################################################
########################################################################################

# TODO : control coherence

def generate_orthogonal_U_V(n_1, n_2, tau_1, tau_2):
    """
    Generate orthogonal matrices U (n_1 x n_1) and V (n_2 x n_2) with controlled coherence.

    Parameters:
        n1 (int): Number of rows/columns of U.
        n2 (int): Number of rows/columns of V.
        tau1 (float): Proportion of canonical basis-aligned singular vectors for U.
        tau2 (float): Proportion of canonical basis-aligned singular vectors for V.

    Returns:
        U (numpy array): Orthogonal matrix of shape (n1, n1).
        V (numpy array): Orthogonal matrix of shape (n2, n2).
    """
    # Number of canonical vectors
    N_1 = int(np.floor(tau_1 * n_1))
    N_2 = int(np.floor(tau_2 * n_2))

    variance_1 = 1/n_1
    variance_2 = 1/n_2

    # Construct U
    U = np.eye(n_1) * np.sqrt(variance_1)  # Start with the identity matrix
    if N_1 < n_1:  # If there are non-canonical components
        # Replace the remaining (n_1 - N_1) columns with random orthogonal vectors
        Q = np.linalg.qr(np.random.randn(n_1, n_1 - N_1) * np.sqrt(variance_1) )[0]
        U[:, N_1:] = Q[:, :n_1 - N_1]  # Replace columns after the first N_1

    # Ensure U is orthogonal
    U, _ = np.linalg.qr(U)

    # Construct V
    V = np.eye(n_2) * np.sqrt(variance_2)  # Start with the identity matrix
    if N_2 < n_2:  # If there are non-canonical components
        # Replace the remaining (n_2 - N_2) columns with random orthogonal vectors
        Q = np.linalg.qr(np.random.randn(n_2, n_2 - N_2) * np.sqrt(variance_2) )[0]
        V[:, N_2:] = Q[:, :n_2 - N_2]  # Replace columns after the first N_2

    # Ensure V is orthogonal
    V, _ = np.linalg.qr(V)

    return U, V

# n_1, n_2 = 100, 100
# #Modulate
# all_mu = []
# all_nu = []
# all_tau = np.arange(0, 10+1)/10
# for tau_1, tau_2 in zip(all_tau, all_tau) :
#     U, V = generate_orthogonal_U_V(n_1, n_2, tau_1, tau_2)
#     Sigma = np.eye(max(n_1, n_2))[:n_1, :n_2] # (n_1, n_2)
#     r=5#min(n_1, n_2)
#     A = U[:,:r] @ Sigma[:r, :r] @ V[:,:r].T # (n_1, n_2)

#     mu = calculate_coherence(A=U, B=np.eye(n_1))
#     nu = calculate_coherence(A=V, B=np.eye(n_2))
#     print(mu, nu)

#     mu, nu = calculate_local_coherence(A, U_n1=None, V_n2=None)
#     all_mu.append(mu)
#     all_nu.append(nu)
#     print(alpha, max(max(mu), max(nu)), max(mu), max(nu))
#     print("=================")



# from matplotlib.colors import LogNorm
# label_fontsize=20
# ticklabel_fontsize=15

# rows, cols = 1, 2
# figsize=(6, 4)
# #figsize=(8, 6)
# #figsize=(15, 10)
# figsize=(cols*figsize[0], rows*figsize[1])
# fig = plt.figure(figsize=figsize)

# for i, (label, data) in enumerate(zip(["mu", "nu"], [all_mu, all_nu])):

#     ax = fig.add_subplot(rows, cols, i+1)

#     img_data = np.array(data) # (alphas, dimensions)
#     img = custom_imshow(
#         img_data, ax=ax, fig=fig, add_text=False,
#         hide_ticks_and_labels=False, xticklabels=np.arange(1, img_data.shape[1]+1), yticklabels=all_tau,
#         filter_step_xticks=5, filter_step_yticks=1  if i==0 else 10**5, log_x=False, log_y=False, base=10,
#         rotation_x=90, rotation_y=0,
#         x_label="Dimensions (n)",  y_label="$\\alpha$" if i==0 else "",
#         # Use LogNorm to apply a logarithmic scale
#         colormesh_kwarg={"shading":'auto', "cmap":'viridis'}, # 'norm':LogNorm(vmin=img_data.min(), vmax=img_data.max())
#         imshow_kwarg={},
#         colorbar=True, colorbar_label=f'$\\{label}$',
#         label_fontsize=label_fontsize,
#         ticklabel_fontsize=ticklabel_fontsize,
#         show=False, fileName=None, dpf=None
#     )

# ##
# #plt.savefig(f"{DIR_PATH_FIGURES}/TODO"  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

# plt.show()


########################################################################################
########################################################################################

# Face splitting product

def F_numpy(A, X1X2=None, X2_bullet_X1=None) :
    """
    A : (n_1, n_2)
    X1 : (N, n_1)
    X2 : (N, n_2)
    X2_bullet_X1 : (N, n_1 x n_2)
    """
    if X2_bullet_X1 is not None :
        y_star = X2_bullet_X1 @ np_vec(A) # (N, n_1 x n_2) x (n_1 x n_2,) = (N,)
    else :
        X1, X2 = X1X2
        #y_star =  np.array([ np.inner(X1[s], A @ X2[s]) for s in range(X1.shape[0])]) # (N,)
        y_star =  np.array([X1[s].T @ A @ X2[s] for s in range(X1.shape[0])]) # (N,)
        # X2_bullet_X1 = face_splitting_product_numpy(X2, X1) # (N, n_1 x n_2)
        # y_star = X2_bullet_X1 @ np_vec(A) # (N, n_1 x n_2) x (n_1 x n_2,) = (N,)
    return y_star # (N,)


def F_cvxpy(A, X1X2=None, X2_bullet_X1=None):
    """
    Compute the linear mapping applied to A:
    A : (n_1, n_2)
    X1 : (N, n_1)
    X2 : (N, n_2)
    X2_bullet_X1 : (N, n_1 x n_2)
    """
    if X2_bullet_X1 is not None:
        # Use CVXPY symbolic operations here
        return X2_bullet_X1 @ cp.vec(A)  # Ensure A is treated as a CVXPY variable
    else:
        # Symbolic computation for individual terms
        X1, X2 = X1X2
        return cp.vstack([cp.sum(cp.multiply(X1[s][:, None] @ X2[s][None, :], A)) for s in range(X1.shape[0])])

########################################################################################
########################################################################################

# Measures

def get_measures_matrix_completion(A_star, N, X1=None, X2=None, P=None, tau=0, one_hot=False, shuffle=True, seed=None):
    """
    Take a matrix A_star and generate two sets of measures (X1, X2, y_star) and (X1_bar, X2_bar, y_star_bar) 
    such that the first set contains N elements and the second set contains the remaining elements.

    y[i] = X1[i]^T A_star X2[i] if one_hot=True else A_star[X1[i], X2[i]]

    For X1, X2, y_star :
        * If P is given, the first N_tau = tau * N elements are selected in the order of decreasing values in P.
        * If tau < 1 (i.e., N_tau < N), the remaining N - N_tau elements are selected randomly.
    For X1_bar, X2_bar, y_star_bar :
        * They contain the remaining elements not selected in X1, X2, y_star.
    """
    assert 0 <= tau <= 1, "tau must be in [0, 1]"
    n_1, n_2 = A_star.shape
    n = n_1 * n_2
    assert 0 < N <= n, "N must be in (0, n_1 * n_2]"
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)

    # X1X2, y_star = data_matrix_completion(torch.from_numpy(A_star), one_hot=one_hot) # (n, n_1), (n, n_2), (n,)
    # X1, X2 = X1X2[:,0], X1X2[:,1] # (n, n_1), (n, n_2)
    (X1, X2), y_star = data_tensor_completion_general(torch.from_numpy(A_star), one_hot=one_hot) # (n, n_1), (n, n_2), (n_1,)
    X1, X2, y_star = X1.numpy(), X2.numpy(), y_star.numpy() # (n, n_1), (n, n_2), (n,)

    
    if P is not None:
        # Select the N_tau elements with the largest values in P (and the remaining elements randomly if tau < 1)

        # Compute the indices of the elements with the largest values in P
        flat_indices = np.argsort(P, axis=None)[::-1]  # Indices of the elements in decreasing order of P
        N_tau = int(tau * N)  # Number of elements to select
        top_indices = flat_indices[:N_tau]  # Indices of the top N_tau elements

        # Select the remaining elements randomly if tau < 1 (i.e., N_tau < N)
        remaining_indices = flat_indices[N_tau:]
        if len(remaining_indices) > 0 and (N - N_tau) > 0:
            selected_indices = np.concatenate([top_indices, np.random.choice(remaining_indices, N - N_tau, replace=False)])
        else :
            selected_indices = top_indices

    else:
        # If P is not given, select the first N elements
        if shuffle:
            selected_indices = np.random.permutation(y_star.shape[0])[:N]
        else:
            selected_indices = np.arange(N)

    # Select the remaining elements
    test_indices = np.setdiff1d(np.arange(y_star.shape[0]), selected_indices)
    X1_bar, X2_bar, y_star_bar = X1[test_indices], X2[test_indices], y_star[test_indices]  # (n_1*n_2-N, n_1), (n_1*n_2-N, n_2), (n_1*n_2-N,)

    # Select the N elements
    X1, X2, y_star = X1[selected_indices], X2[selected_indices], y_star[selected_indices]  # (N, n_1), (N, n_2), (N,)

    if seed is not None:
        np.random.set_state(old_state)

    return (X1, X2, y_star), (X1_bar, X2_bar, y_star_bar)

def get_measures_matrix_sensing(A_star, U_star, V_star, N, tau=0.0, variance=None, seed=None):
    M1, X1 = get_measures(N, Phi=U_star, tau=tau, variance=variance, seed=seed)
    M2, X2 = get_measures(N, Phi=V_star, tau=tau, variance=variance, seed=seed)
    X2_bullet_X1 = None
    X2_bullet_X1 = face_splitting_product_numpy(X2, X1)
    #y_star = F_cvxpy(A_star, X1X2=(X1, X2), X2_bullet_X1=X2_bullet_X1)
    y_star = F_numpy(A_star, X1X2=(X1, X2), X2_bullet_X1=X2_bullet_X1)
    return X1, X2, X2_bullet_X1, y_star

def get_data_matrix_factorization(A_star, U_star, V_star, N, problem, tau=0.0, variance=None, seed=None) :
    assert problem in ['matrix-completion', 'matrix-sensing']
    if problem == 'matrix-sensing':
        X1, X2, X2_bullet_X1, y_star = get_measures_matrix_sensing(A_star, U_star, V_star, N, tau, variance, seed=seed)
        X1_bar, X2_bar, X2_bullet_X1_bar, y_star_bar = None, None, None, None
    elif problem == 'matrix-completion':
        P = None
        if tau != 0 :
            mu, nu, P = calculate_local_coherence(A_star, U_n1=None, V_n2=None)
        (X1, X2, y_star), (X1_bar, X2_bar, y_star_bar) = get_measures_matrix_completion(A_star, N, P=P, tau=tau, one_hot=True, shuffle=True, seed=seed)

        X2_bullet_X1 = None
        X2_bullet_X1 = face_splitting_product_numpy(X2, X1)
        #y_star = F_cvxpy(A_star, X1X2=(X1, X2), X2_bullet_X1=X2_bullet_X1)
        y_star = F_numpy(A_star, X1X2=(X1, X2), X2_bullet_X1=X2_bullet_X1)

        if X1_bar.shape[0] > 0 :
            X2_bullet_X1_bar = face_splitting_product_numpy(X2_bar, X1_bar)
            #y_star_bar = F_cvxpy(A_star, X1X2=(X1_bar, X2_bar), X2_bullet_X1=X2_bullet_X1_bar)
            y_star_bar = F_numpy(A_star, X1X2=(X1_bar, X2_bar), X2_bullet_X1=X2_bullet_X1_bar)
        else :
            X2_bullet_X1_bar, y_star_bar = None, None

    return (X1, X2, X2_bullet_X1, y_star), (X1_bar, X2_bar, X2_bullet_X1_bar, y_star_bar)

########################################################################################
########################################################################################

# Convex optimization

def solve_matrix_factorization_nuclear_norm(n_1, n_2, y_star, X1X2=None, X2_bullet_X1=None, X1X2_bar=None, X2_bullet_X1_bar=None, reg=0, EPSILON=1e-6):
    """
    Solve the minimization problem (P5) to recover A*
    # Minimize ||A||_* subject to ||(X2 • X1)A - y*||_2 = sum_i (X1_i^T A X2_i - y*_i)^2 <= epsilon (epsilon = 0 in noiseless case
    
    For matrix completion, we want to solve
        Minimize ||A||_* subject to ||(X2 • X1)A - y*||_2 <= epsilon, 
        ie A_{ij} = A_{ij}^* for (i, j) in Omega
        or, if reg != 0, then
        Minimize ||A||_* subject to ||(X2 • X1)A - y*||_2 + reg * ||(X2_bar • X1_bar)A||_2 <= epsilon, ie A_{ij} = A_{ij}^* for (i, j) in Omega
        ie A_{ij} = A_{ij}^* for (i, j) in Omega and A_{ij} = 0 for (i, j) not in Omega
    """
    # Define the variable A (n_1 x n_2 matrix to be optimized)
    A = cp.Variable((n_1, n_2))

    # Define the objective function (nuclear norm of A)
    objective = cp.Minimize(cp.normNuc(A))

    # Define the constraints 
    if X1X2_bar is None and X2_bullet_X1_bar is None :
        reg = 0
    else :
        # If X1X2_bar and X2_bullet_X1_bar are not None, and reg is not 0, then the constraint is added
        # N_prime = X1X2_bar[0].shape[0] if X1X2_bar is not None else X2_bullet_X1_bar.shape[0]
        N_prime = 0
        N_prime = 0 if X1X2_bar[0] is None else X1X2_bar[0].shape[0]
        N_prime = 0 if X2_bullet_X1_bar is None else X2_bullet_X1_bar.shape[0]
        reg = reg if N_prime == 0 else 0
    #constraints = [F_cvxpy(A, X1X2=X1X2, X2_bullet_X1=X2_bullet_X1) = y_star]
    constraints = [
        cp.norm(F_cvxpy(A, X1X2=X1X2, X2_bullet_X1=X2_bullet_X1) - y_star, 'fro')
        + (reg * cp.norm(F_cvxpy(A, X1X2=X1X2_bar, X2_bullet_X1=X2_bullet_X1_bar), 'fro') if reg!=0 else 0) <= EPSILON]  # tolerance for numerical precision

    # Set up and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Solution
    A = A.value
    return A

########################################################################################
########################################################################################
### Gradient Descent : Subgradient & Projected & Proximal (ISTA) Gradient Descent 

########################################################################################
########################################################################################

########################################################################################
########################################################################################


def soft_threshold_matrix(A, threshold):
    """Applies the matrix soft-thresholding operator (SVD-based)."""
    U, Sigma, Vt = svd(A, full_matrices=False)
    return U @ np.diag(np.maximum(Sigma - threshold, 0)) @ Vt, Sigma

def subgradient_descent_matrix_factorization(
    X, 
    X_bar, 
    y_star, 
    A_star, 
    method="subgradient", 
    learning_rate=0.1, 
    beta_2=0.0, 
    beta_star=0.1, 
    beta_1=0.0, 
    init_scale=0.0,
    init_method = "identity", 
    max_iter=10**3, 
    eval_first=10**3, 
    eval_step=10**4, 
    tol=1e-6,
    verbose=False
):
    """
    Solves the matrix factorization problem:
    Minimizes (1/2) ||X vec(A) - y^*||_2^2 + (beta_2/2) * ||A||_F^2 + beta_star * ||A||_* + beta_1 * ||A||_1.
 
    Parameters:
    ----------
    X : np.ndarray, Measurement matrix of shape (N, n_1 * n_2).
    y_star : np.ndarray, Observation vector of shape (N,).
    A_star : np.ndarray, True matrix for comparison, of shape (n_1, n_2).
    method : str, Optimization method to use (subgradient, proj_subgradient, ISTA).
    learning_rate : float, Step size for gradient updates.
    beta_1, beta_*, beta_1 : float, Regularization parameter for l_2, l_* and l_1.
    init_scale: float, initialization scale
    init_method : str, Initialization method for A^{(1)}.
    max_iter : int, Maximum number of iterations.
    eval_first : int, First iteration to evaluate the error.
    eval_step : int, Evaluate the error every eval_step iterations.
    tol : float, Tolerance for convergence

    Returns:
    -------
    np.ndarray, Recovered sparse vector a of shape (n_1, n_2).
    list, List of recovery errors over iterations.
    list, List of training errors over iterations.
    list, List of all recovered sparse vectors over iterations.
    list, List of all singular values over iterations.
    list, List of all iterations.
    """

    assert method in ["subgradient", "proj_subgradient", "ISTA"]
    assert init_method in ["identity", "random", "least_square"]

    n_1, n_2 = A_star.shape
    n = n_1 * n_2
    N = y_star.shape[0]
    eps = 1e-6

    ###############################################################
    ###############################################################

    U, Sigma12, VT = np.linalg.svd(X, full_matrices=False) # (N, r), (r,), (n, r)
    Sigma = Sigma12  * Sigma12
    #a_LS = VT.T @ np.diag(1/(Sigma + beta_2)) @ VT @ X.T @ y_star
    print("0, learning_rate, 2 /  max(Sigma + beta_2)", 0, learning_rate, 2 / max(Sigma + beta_2))
    assert 0 < learning_rate < 2 / max(Sigma + beta_2)

    ###############################################################
    ###############################################################

    # Least square solution
    a_star = np_vec(A_star)
    A_LS = np_unvec(np.linalg.pinv(X.T @ X + beta_2) @ X.T @ y_star, n_1, n_2) # (n_1, n_2)

    ###############################################################
    ###############################################################

    # Initialize A^{(1)}
    if init_method == "identity":
        A = np.eye(max(n_1, n_2))[:n_1, :n_2]
    elif init_method == "random":
        A = np.random.randn(n_1, n_2) / np.sqrt(n)
    elif init_method == 'least_square':
        A = A_LS + 0
        
    A = A * init_scale

    ###############################################################
    ##### Sub gradient : no projection
    ###############################################################
    if method == "subgradient" :
        Pi = lambda A, alpha_t : A # (n_1, n_2)
    ###############################################################
    ##### Projected subgradient : Projection on the set of feasible solutions
    ###############################################################
    if method == "proj_subgradient":
        P = X.T @ np.linalg.inv(X @ X.T) # (n, N)
        Pi = lambda A, alpha_t : A - np_unvec( P @ (X @ np_vec(A) - y_star), n_1, n_2) # (n_1, n_2)
    ###############################################################
    ##### Priximal gradient (ISTA) : Soft thresholding
    ###############################################################
    elif method == "ISTA" :
        beta_star_ista = beta_star + 0
        Pi = lambda A, alpha_t: A
        #Pi = lambda A, alpha_t : soft_threshold_matrix(A, threshold=alpha_t * beta_star_ista) # (n_1, n_2)
        #beta_star = 0

    ###############################################################
    ###############################################################

    train_errors = []
    errors = []

    a = np_vec(A)
    err = np.linalg.norm(X @ a - y_star) / (np.linalg.norm(y_star)+eps)
    recovery_error = np.linalg.norm(A - A_star) / (np.linalg.norm(A_star)+eps)
    train_errors.append(err)
    errors.append(recovery_error)

    ###############################################################
    ###############################################################

    all_A = []
    all_A.append(A+0)
    
    all_Sigma = []
    all_Sigma.append(svd(A)[1])

    ###############################################################
    ###############################################################

    all_iterations = []
    all_iterations.append(0)

    ###############################################################
    ###############################################################

    for iteration in tqdm(list(range(1, max_iter+1))):
        ## Gradient the data-fitting term ||X vec(A)  - y*||_2^2
        grad = X.T @ (X @ a - y_star) # (n_1*n_2,)
        #grad += X_bar.T @ (X_bar @ a - 0) # TODO : necessary ?
        grad = np_unvec(grad, n_1, n_2) # (n_1, n_2)

        # Gradient of Frobenius norm
        grad += beta_2 * A

        # Gradient of the l1 term ||A||_1
        grad += beta_1 * np.sign(A)

        # # Subgradient of nuclear norm ||A||_* : see SVD is expensive, check if not ISTA before computing it
        # # U, Sigma, Vt = svd(A, full_matrices=False)
        # # h_A = U @ Vt
        # # grad += beta_star * h_A


        ## LR schedule
        alpha_t = learning_rate
        #alpha_t = learning_rate / iteration
        # if (iteration+1)%10**3 == 0 : learning_rate_t = learning_rate / 10**t
        #alpha_t = learning_rate / np.sqrt(iteration)
        #alpha_t = learning_rate / np.linalg.norm(sub_gradient)
        #alpha_t = err / np.linalg.norm(sub_gradient)**2

        if method == "ISTA":
            # Proximal gradient update for the trace norm
            A, Sigma = soft_threshold_matrix(A - alpha_t * grad, alpha_t * beta_star_ista)
        else:
            # Subgradient of nuclear norm ||A||_*
            if beta_star !=0 :
                U, Sigma, Vt = svd(A, full_matrices=False)
                h_A = U @ Vt
                grad += beta_star * h_A
            # Subgradient or projected subgradient
            A = A - alpha_t * grad
            A = Pi(A, alpha_t)

        a = np_vec(A)
        # Compute errors every eval_step iterations
        if iteration % eval_step == 0 or iteration < eval_first :
            err = np.linalg.norm(X @ a - y_star) / (np.linalg.norm(y_star)+eps)
            recovery_error = np.linalg.norm(A - A_star) / (np.linalg.norm(A_star)+eps)
            train_errors.append(err)
            errors.append(recovery_error)
            ## A
            all_A.append(A+0)
            all_Sigma.append(Sigma)

            all_iterations.append(iteration)

            if verbose:
                print(f"error : train={round(err,4)}, test={round(recovery_error,4)}")

        # # Convergence check
        # if np.linalg.norm(grad) < tol and recovery_error < tol:
        #       print(f"Converged at iteration {iteration}.")
        #       break

    return A, errors, train_errors, all_A, all_Sigma, all_iterations, A_LS

########################################################################################
########################################################################################

def prod_matrices(As:list):
    """
    Compute the product of matrices in As : A_1 @ A_2 @ ... @ A_k
    """
    A = As[0] + 0
    for k in range(1, len(As)): A = A @ As[k]
    return A

def deep_subgradient_descent_matrix_factorization(
    X, 
    X_bar, 
    y_star, 
    A_star, 
    L=2, 
    method="subgradient", 
    learning_rate=0.1, 
    beta_2=0.0, 
    beta_star=0.1, 
    beta_1=0.0,  
    init_scale=0.0,
    init_method = "identity", 
    max_iter=10**3, 
    eval_first=10**3, 
    eval_step=10**4, 
    tol=1e-6,
    verbose=False
):
    """
    Solves the matrix factorization problem:
    Minimizes (1/2) ||X vec(A) - y^*||_2^2 + (beta_2/2) * ||A||_F^2 + beta_star * ||A||_* + beta_1 * ||A||_1.

    Parameters:
    ----------
    X : np.ndarray, Measurement matrix of shape (N, n_1 * n_2).
    y_star : np.ndarray, Observation vector of shape (N,).
    A_star : np.ndarray, True matrix for comparison, of shape (n_1, n_2).
    method : str, Optimization method to use (subgradient, proj_subgradient, ISTA).
    learning_rate : float, Step size for gradient updates.
    beta_1, beta_*, beta_1 : float, Regularization parameter for l_2, l_* and l_1.
    init_scale: float, initialization scale
    init_method : str, Initialization method for A^{(1)}.
    max_iter : int, Maximum number of iterations.
    eval_first : int, First iteration to evaluate the error.
    eval_step : int, Evaluate the error every eval_step iterations.
    tol : float, Tolerance for convergence

    Returns:
    -------
    np.ndarray, Recovered sparse vector a of shape (n_1, n_2).
    list, List of recovery errors over iterations.
    list, List of training errors over iterations.
    list, List of all recovered sparse vectors over iterations.
    list, List of all singular values over iterations.
    list, List of all iterations.
    """
    assert method in ["subgradient", "proj_subgradient", "ISTA"]
    assert init_method in ["identity", "random", "least_square"]
    assert L >= 1

    n_1, n_2 = A_star.shape
    n = n_1 * n_2
    N = y_star.shape[0]

    ###############################################################
    ###############################################################

    U, Sigma12, VT = np.linalg.svd(X, full_matrices=False) # (N, r), (r,), (n, r)
    Sigma = Sigma12  * Sigma12
    #a_LS = VT.T @ np.diag(1/(Sigma + beta_2)) @ VT @ X.T @ y_star
    print("0, learning_rate, 2 /  max(Sigma + beta_2)", 0, learning_rate, 2 / max(Sigma + beta_2))
    assert 0 < learning_rate < 2 / max(Sigma + beta_2)

    ###############################################################
    ###############################################################

    # Least square solution
    a_star = np_vec(A_star)
    A_LS = np_unvec(np.linalg.pinv(X.T @ X + beta_2) @ X.T @ y_star, n_1, n_2) # (n_1, n_2)

    ###############################################################
    ###############################################################

    As = []
    dimensions = []

    d = max(n_1, n_2) if L > 1 else n_2
    A = np.ones((n_1, d))

    ## Initialize A_1
    if init_method == "identity": A_1 = np.eye(max(n_1, d))[:n_1, :d]
    elif init_method == "random": A_1 = np.random.randn(n_1, d) / np.sqrt(n_1*d)
    elif init_method == 'least_square': A_1 = A_LS + 0
    A_1 = A_1 * init_scale
    As.append(A_1 + 0) # [A1]
    A = A @ A_1 # A1 (n_1, d)
    dimensions.append( (n_1, d) )

    ## Initialize A_2 ... A_{L-1}
    if L > 2 :
        for k in range(L-2) :
            if init_method == "identity": A_k = np.eye(d)
            elif init_method == "random": A_k = np.random.randn(d, d) / d
            elif init_method == 'least_square': A_k = A_LS + 0
            A_k = A_k * init_scale
            As.append(A_k + 0) # [A1, A2, ..., Ak]
            A = A * A_k # A_1 @ A_2 @ ... @ A_k (n_1, d)
            dimensions.append( (d, d) )

    ## Initialize A_L
    if L > 1 :
        if init_method == "identity": A_L = np.eye(d)[:d, :n_2]
        elif init_method == "random": A_L = np.random.randn(d, n_2) / np.sqrt(d*n_2)
        elif init_method == 'least_square': A_L = A_LS + 0
        A_L = A_L * init_scale
        As.append(A_L + 0) # [A1, A2, ..., Ak, A_L]
        A = A @ A_L # A_1 @ A_2 @ ... @ A_k @ A_L (n_1, n_2)
        dimensions.append( (d, n_2) )

    ###############################################################
    ##### Sub gradient : no projection
    ###############################################################
    if method == "subgradient" :
        Pi = lambda A, alpha_t : A # (n_1, n_2)
    ###############################################################
    ##### Projected subgradient : Projection on the set of feasible solutions
    ###############################################################
    if method == "proj_subgradient":
        P = X.T @ np.linalg.inv(X @ X.T) # (n, N)
        Pi = lambda A, alpha_t : A - np_unvec( P @ (X @ np_vec(A) - y_star), n_1, n_2) # (n_1, n_2)
    ###############################################################
    ##### Priximal gradient (ISTA) : Soft thresholding
    ###############################################################
    elif method == "ISTA" :
        beta_star_ista = beta_star + 0
        Pi = lambda A, alpha_t: A
        #Pi = lambda A, alpha_t : soft_threshold_matrix(A, threshold=alpha_t * beta_star_ista) # (n_1, n_2)
        #beta_star = 0

    ###############################################################
    ###############################################################

    train_errors = []
    errors = []

    # Initial error
    a = np_vec(A)
    err = np.linalg.norm(X @ a - y_star) / np.linalg.norm(y_star)
    recovery_error = np.linalg.norm(A - A_star) / np.linalg.norm(A_star)
    train_errors.append(err)
    errors.append(recovery_error)

    ###############################################################
    ###############################################################

    all_A = []
    all_A.append(A+0)
    
    all_Sigma = []
    all_Sigma.append(svd(A)[1])

    ###############################################################
    ###############################################################

    all_iterations = []
    all_iterations.append(0)

    ###############################################################
    ###############################################################

    for iteration in tqdm(list(range(1, max_iter+1))):
        ## Gradient the data-fitting term ||X vec(A)  - y*||_2^2
        grad_A = X.T @ (X @ a - y_star) # (n_1*n_2,)
        #grad_A += X_bar.T @ (X_bar @ a - 0) # TODO : necessary ?
        grad_A = np_unvec(grad_A, n_1, n_2) # (n_1, n_2)

        """
        A = A_1 @ A_2 @ ... @ A_k @ A_L
        vec(A) = 
            * ( (A_2 @ ... @ A_L)^\top \kron I_n1 ) @ vec(A_1)
            * ( (A_{k+1} @ ... @ A_L)^\top  \kron (A_1 @ ... @ A_{k-1}) ) @ vec(A_k) for k=2,...,L-1
            * ( I_{n_2} \kron (A_1 @ ... @ A_{L-1}) ) @ vec(A_L)

        So, grad a = vec(A) wrt a_l=vec(A_l) = 
            * (A_2 @ ... @ A_L)^\top \kron I_n1
            * (A_{k+1} @ ... @ A_L)^\top  \kron (A_1 @ ... @ A_{k-1}) for k=2,...,L-1
            * I_{n_2} \kron (A_1 @ ... @ A_{L-1})
        
        And grad_A := grad Loss wrt vec(A) = X.T @ (X @ a - y_star) = X.T @ (X @ vec(A) - y_star)
        So, by chain rule,
            grad_A_k := grad Loss wrt vec(A_k)
                     = ( grad vec(A) wrt vec(A_k) ) @ ( grad Loss wrt vec(A) ) 

            and grad_A_k = grad_Loss wrt A_k = unvec(grad_A_k, dimensions[k])
        """

        ## LR schedule
        alpha_t = learning_rate
        #alpha_t = learning_rate / iteration
        # if (iteration+1)%10**3 == 0 : learning_rate_t = learning_rate / 10**t
        #alpha_t = learning_rate / np.sqrt(iteration)
        #alpha_t = learning_rate / np.linalg.norm(sub_gradient)
        #alpha_t = err / np.linalg.norm(sub_gradient)**2

        new_As = []

        # First matrix: G(A) (A2 ... AL)ᵀ + β2 * A1
        G_0 = grad_A.dot(np.prod(As[1:], axis=0).T) + beta_2 * A[0] + beta_1 * np.sign(As[0])
        if beta_star > 0:
            U, Sigma, Vt = svd(As[0], full_matrices=False)
            G_0 += beta_star * U @ Vt
        new_As.append(As[0] - alpha_t * G_0)

        #  Middle matrices: (A1 ... Ak-1)ᵀ G(A) (Ak+1 ... AL)ᵀ + β2 * Ak
        if L > 2:
            for k in range(1, L-1):
                G_k = np.prod(As[:k], axis=0).T.dot(grad_A).dot(np.prod(As[k+1:], axis=0).T) + beta_2 * As[k] + beta_1 * np.sign(As[k])
                if beta_star > 0:
                    U, Sigma, Vt = svd(As[k], full_matrices=False)
                    G_k += beta_star * U @ Vt
                new_As.append(As[k] - alpha_t * G_k)

        # Last matrix: (A1 ... AL-1)ᵀ G(A) + β2 * AL
        if L > 1:
            G_L = np.prod(As[:L-1], axis=0).T.dot(grad_A) + beta_2 * As[-1] + beta_1 * np.sign(As[-1])
            if beta_star > 0:
                U, Sigma, Vt = svd(As[L-1], full_matrices=False)
                G_L += beta_star * U @ Vt
            new_As.append(As[L-1] - alpha_t * G_L)


        # Update A
        As = new_As
        A = prod_matrices(As)

        a = np_vec(A)
        # Compute errors every eval_step iterations
        if iteration % eval_step == 0 or iteration < eval_first :
            err = np.linalg.norm(X @ a - y_star) / np.linalg.norm(y_star)
            recovery_error = np.linalg.norm(A - A_star) / np.linalg.norm(A_star)
            train_errors.append(err)
            errors.append(recovery_error)
            ## A
            all_A.append(A+0)
            all_Sigma.append(Sigma)

            all_iterations.append(iteration)

            if verbose:
                print(f"error : train={round(err,4)}, test={round(recovery_error,4)}")

        # # Convergence check
        # if np.linalg.norm(grad) < tol and recovery_error < tol:
        #       print(f"Converged at iteration {iteration}.")
        #       break

    return A, errors, train_errors, all_A, all_Sigma, all_iterations, A_LS

########################################################################################
########################################################################################

def get_gradient_matrix_factorization(X, y_star, all_A, beta_star, ord=2):
    norms_grad_1 = []
    norms_grad_2 = []
    norms_grad_ratio = []

    for A in tqdm(all_A):

        grad_1 = X.T @ (X @ np_vec(A) - y_star) # (n_1*n_2,)
        n_1 = np.linalg.norm(grad_1, ord=ord) #/ np.linalg.norm(b, ord=ord)

        U, Sigma, Vt = svd(A, full_matrices=False)
        grad_2 = beta_star * np_vec(U @ Vt)
        n_2 = np.linalg.norm(grad_2, ord=ord) #/ np.linalg.norm(b_ajusted, ord=ord)

        norms_grad_1.append(n_1)
        norms_grad_2.append(n_2)
        norms_grad_ratio.append(n_2 / n_1)

    return norms_grad_1, norms_grad_2, norms_grad_ratio

########################################################################################
########################################################################################

if __name__ == "__main__":
    
    ### Problem
    problem = 'matrix-completion'
    problem = 'matrix-sensing'

    n_1, n_2 = 10, 10  # Dimension of A*
    n = n_1 * n_2
    r=2
    symmetric=False
    seed = None
    N=70
    tau = .2

    A_star, U_star, Sigma_star, V_star = get_matrices_UV(n_1, n_2, rank=r, symmetric=symmetric, normalize=True, scale=None, seed=seed)

    # (X1, X2, X2_bullet_X1, y_star), (X1_bar, X2_bar, X2_bullet_X1_bar, y_star_bar) = get_data_matrix_factorization(
    #     A_star, U_star, V_star, N, problem, tau=tau, variance=None, seed=seed)

    # ###############################################################

    # A = solve_matrix_factorization_nuclear_norm(
    #     n_1, n_2, y_star,
    #     X1X2=(X1, X2), X2_bullet_X1=X2_bullet_X1,
    #     X1X2_bar=(X1_bar, X2_bar), X2_bullet_X1_bar=X2_bullet_X1_bar, EPSILON=1e-6)
    # recovery_error = np.linalg.norm(A - A_star, ord="fro") / np.linalg.norm(A_star, ord="fro")
    # print(recovery_error)
    
    # ###############################################################

#     alpha = 1e-2
#     beta_star = 0 * 1e-5
#     beta_2 = 1e-4
#     init_scale = 1e-0 * 5
#     init_method = "identity"
#     init_method = "random"

#     method = "subgradient"
#     # method = "proj_subgradient"
#     # method = "ISTA"

#     max_iter=10**6*1+1
#     #max_iter=10**5*2+1
#     #max_iter=10**4*5+1


#     # A, errors, train_errors, all_A, all_Sigma, iterations, A_LS = subgradient_descent_matrix_factorization(
#     #     X=X2_bullet_X1, X_bar=X2_bullet_X1_bar, y_star=y_star, A_star=A_star, 
#     #     method=method, learning_rate=alpha, beta_2=beta_2, beta_star=beta_star, 
#     #     init_scale=init_scale, init_method=init_method,
#     #     max_iter=max_iter, eval_first=10**3, eval_step=10**4, tol=1e-6
#     # )
#     # recovery_error = np.linalg.norm(A - A_star, ord="fro") / np.linalg.norm(A_star, ord="fro")
#     # print(recovery_error)

#     ###############################################################

#     A, errors, train_errors, all_A, all_Sigma, iterations, A_LS = deep_subgradient_descent_matrix_factorization(
#         X=X2_bullet_X1, X_bar=X2_bullet_X1_bar, y_star=y_star, A_star=A_star, L=2, 
#         method=method, learning_rate=alpha, beta_2=beta_2, beta_star=beta_star, 
#         init_scale=init_scale, init_method=init_method,
#         max_iter=max_iter, eval_first=10**3, eval_step=10**4, tol=1e-6
#     )
#     recovery_error = np.linalg.norm(A - A_star, ord="fro") / np.linalg.norm(A_star, ord="fro")
#     print(recovery_error)

#     ###############################################################

#     label_fontsize=20
#     ticklabel_fontsize=15

#     rows, cols = 1, 2
#     figsize=(6, 4)
#     figsize=(cols*figsize[0], rows*figsize[1])
#     fig = plt.figure(figsize=figsize)

#     ###################################################
#     ###################################################

#     ax = fig.add_subplot(rows, cols, 1)
#     #_, ax, _ = get_twin_axis(ax=ax, no_twin=True)
#     ax.plot(iterations, train_errors, label="Train error", color="blue", linestyle="-", linewidth=2)
#     ax.plot(iterations, errors, label="Recovery error", color="red", linestyle="-", linewidth=2)

#     ax.set_xscale("log")
#     ax.set_yscale("log")

#     plt.show()