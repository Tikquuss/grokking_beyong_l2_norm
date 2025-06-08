import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import scipy
import cvxpy as cp
from tqdm import tqdm

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# print(default_colors)

########################################################################################
########################################################################################

"""
Basis $\Phi$ for compressed sensing
We want a matrix $\Phi \in \mathbb{R}^{n \times m}$ with $\Phi \overset{iid}{\sim} \mathcal{N}(0, \sigma^2)$ such that $\Phi^\top \Phi \propto \mathbb{I}_n$.

If $\Phi \overset{iid}{\sim} \mathcal{N}(0, \sigma^2)$, then $\mathbb{E}[\Phi^\top \Phi] = n \sigma^2 \mathbb{I}_n$ since
$\mathbb{E}[\Phi^\top \Phi]_{i,j}
= \sum_{k=1}^n  \mathbb{E}[\Phi_{ki} \Phi_{kj}]
= \sum_{k=1}^n \delta_{ij} \mathbb{E}[\Phi_{ki}^2]
= \delta_{ij} n \sigma^2$.

So $$\mathbb{E}[\Phi^\top \Phi] = \mathbb{I}_n \text{ for } \sigma = 1/\sqrt{n}$$

* Fourier basis
* $Q$ of the QR decomposition on random normal : Haar mesures
* Sample $\Psi \overset{iid}{\sim} \mathcal{N}(0, \sigma^2)$ and take $\Phi = \Psi \left( \Psi^\top \Psi\right)^{-\frac{1}{2}}$ : Haar mesures
"""

def create_Fourier_basis(n):
    """
    Creates a complex-valued matrix Φ of size (n, n) where each element is defined by:
        Φ_{ji} = (1 / sqrt(n)) * exp(-2 * π * i * j * i / n)

    Parameters:
    ----------
    n : int
        The dimension of the square matrix Φ.

    Returns:
    -------
    np.ndarray
        A complex-valued (n, n) matrix Φ with each element calculated as
        (1 / sqrt(n)) * exp(-2 * π * i * j * i / n).
    """
    #i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    #Phi = np.exp(-2j * np.pi * i * j / n)
    Phi = np.fft.fft(np.eye(n))
    return Phi / np.sqrt(n)  # normalized

def create_orthonormal_basis(n, scaler=None, seed=None):
    """
    Creates an orthonormal basis from a QR decomposition of a random matrix

    Parameters:
    ----------
    n : int
        The dimension of the matrix.

    Returns:
    -------
    np.ndarray
        A real-valued (n, n) matrix Φ, orthonormal.
    """
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)
    Phi = np.random.randn(n, n)
    if seed is not None:
        np.random.set_state(old_state)
    scaler = (1/np.sqrt(n)) if scaler is None else 1.0
    Phi = scaler * Phi
    return np.linalg.qr(Phi)[0]  # orthonormalized
    
def create_normal_basis(n, scaler=None, seed=None, normalized=False):
    """
    Creates an orthonormal basis from the normalization of a random normal matrix

    Parameters:
    ----------
    n : int
        The dimension of the matrix.

    Returns:
    -------
    np.ndarray
        A real-valued (n, n) matrix Φ, orthonormal.
    """
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)

    scaler = (1/np.sqrt(n)) if scaler is None else 1.0
    Phi = scaler * np.random.randn(n, n)

    if seed is not None:
        np.random.set_state(old_state)

    if normalized :
        Phi = Phi @ scipy.linalg.fractional_matrix_power(Phi.T @ Phi, -0.5) # ortho-normalized

    return Phi


########################################################################################
########################################################################################
### Signal $a^*$ and $b^*=\Phi a^*$

def create_signal(n, s, distribution="normal", Phi=None, scaler=None, seed=0):
    """
    Generate a sparse representation a*, with ||a*||_0 <= s

    Parameters:
    ----------
    n : int
        The dimension of the signal.
    s : int
        The sparsity level of the signal.
    distribution : str
        The distribution of the non-zero entries in a*.
    Phi : np.ndarray
        The basis matrix of shape (n, n).
    scaler : float
        The scaling factor for the signal, ie the standard deviation of the non-zero entries (default: 1/sqrt(n)).
    seed : int
        The random seed for reproducibility.

    Returns:
    -------
    np.ndarray
        A sparse representation a* of shape (n,).
    np.ndarray
        The corresponding signal b* = Φ a* of shape (n,).
    """
    assert distribution in ["normal", "uniform"]
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)

    # a*
    a_star = np.zeros(n) # (n,)
    non_zero_indices = np.random.choice(n, s, replace=False) # (s,)
    a_star[non_zero_indices] = np.random.randn(s) if distribution=="normal" else np.random.choice(a=[-1, 0, 1], size=s, replace=True)
    # Make sure a*!=0
    while (a_star**2).sum() == 0 :
        a_star[non_zero_indices] = np.random.randn(s) if distribution=="normal" else np.random.choice(a=[-1, 0, 1], size=s, replace=True)

    if seed is not None:
        np.random.set_state(old_state)

    scaler = (1/np.sqrt(n)) if scaler is None else 1.0
    a_star = scaler * a_star
    b_star = a_star if Phi is None else Phi @ a_star
    return a_star, b_star


########################################################################################
########################################################################################
### Noise


def create_noise_from_scratch(N, SNR, n, s, distribution="normal", Phi=None, scaler=None, seed=0):
    """
    Create a noise vector ξ with the specified SNR = E||a*||_2^2 / E||ξ||_2^2

    Parameters:
    ----------
    N : int
        Dimension of the noise vector.
    n : int
        The dimension of the signal.
    s : int
        The sparsity level of the signal.
    SNR : float
        The signal-to-noise ratio. SNR = E||a*||_2^2 / E||ξ||_2^2
        SNR = np.inf : no noise
        SNR = 0 : only noise
    distribution : str
        The distribution of the non-zero entries in a*.
    Phi : np.ndarray
        The basis matrix of shape (n, n).
    scaler : float
        The scaling factor for the signal, ie the standard deviation of the non-zero entries (default: 1/sqrt(n)).
    seed : int
        The random seed for reproducibility.

    Returns:
    -------
    np.ndarray
        A noise vector ξ of shape (N,).
    """

    if SNR is None or SNR==np.inf :
        return np.zeros(N,)
    elif SNR==0 :
        return np.random.randn(N,) * np.inf

    maean_norm_a_star_square = np.mean([
        np.linalg.norm(create_signal(n, s, distribution, Phi=Phi, seed=None if seed is None else i*seed)[0])**2 
        for i in range(n * 10**2)
    ])
    sigma_xi = np.sqrt(maean_norm_a_star_square / (N * SNR))

    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)
    xi = sigma_xi * np.random.randn(N,) 
    if seed is not None:
        np.random.set_state(old_state)

    return xi

def create_noise_from_mean_norm_signal(N, SNR, maean_norm_a_star_square=None, seed=0):
    """
    Create a noise vector ξ with the specified SNR = E||a*||_2^2 / E||ξ||_2^2

    Parameters:
    ----------
    N : int
        Dimension of the noise vector.
    SNR : float
        The signal-to-noise ratio. SNR = E||a*||_2^2 / E||ξ||_2^2
        SNR = np.inf : no noise
        SNR = 0 : only noise
    seed : int
        The random seed for reproducibility.

    Returns:
    -------
    np.ndarray
        A noise vector ξ of shape (N,).
    """

    if SNR is None or SNR==np.inf :
        return np.zeros(N,)
    elif SNR==0 :
        return np.random.randn(N,) * np.inf

    sigma_xi = np.sqrt(maean_norm_a_star_square / (N * SNR))

    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)
    xi = sigma_xi * np.random.randn(N,) 
    if seed is not None:
        np.random.set_state(old_state)

    return xi

########################################################################################
########################################################################################
### Measures $X$

def get_measures(N, Phi, tau=0, variance=None, seed=None):
    """
    Generates a matrix M where r rows come from random columns of Phi,
    and the remaining N - r rows are random.
    Parameters:
    ----------
    Phi : np.ndarray
        Basis matrix of shape (n, m).
    N : int
        Number of rows for the generated matrix X.
    tau : float
        Proportion of rows in X that should be selected from columns of Phi.

    Returns:
    -------
    M : np.ndarray
        Generated matrix of shape (N, n).
    """
    assert 0 <= tau <= 1
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)

    n, m = Phi.shape
    if variance is None : variance = 1/n
    is_complex = np.iscomplexobj(Phi)

    if tau == 0 :
        M = np.random.randn(N, n) # (N, n)
        if is_complex : M = M + 1j * np.random.randn(N, n) # (N, n)
    else :
        if is_complex : M = np.zeros((N, n), dtype=complex)
        else : M = np.zeros((N, n))
        # Randomly choose tau*N columns from Phi to place in X
        N_1 = int(tau*N)
        #selected_columns = np.tile(np.arange(m), N_1 // m + 1)[:N_1] # repeat [0, ..., m-1] until the length is N_1
        if N_1<=m :
            #selected_columns = np.arange(N_1)
            selected_columns = np.random.choice(m, N_1, replace=False)
        else :
            selected_columns = np.zeros((N_1,), dtype=int)
            selected_columns[:m] = np.arange(m) # the first m columns are the columns of Phi
            selected_columns[m:] = np.random.choice(m, N_1-m, replace=True) # the last at sample randomnly from Phy
        # Fill the first r rows with random columns of Phi
        M[:N_1, :] = Phi[:, selected_columns].T * (1/np.sqrt(variance)) # Transpose to match dimensions (N_1, n)
        # Fill the remaining rows with random entries
        if is_complex :
            #for i in range(N_1, N): M[i, :] = np.random.randn(n) + 1j * np.random.randn(n)
            M[N_1:, :] = np.random.randn(N-N_1, n) + 1j * np.random.randn(N-N_1, n) # (N-N_1, n)
        else :
            #for i in range(N_1, N): M[i, :] = np.random.randn(n)
            M[N_1:, :] = np.random.randn(N-N_1, n) # (N-N_1, n)

    if seed is not None:
        np.random.set_state(old_state)

    M = np.sqrt(variance) * M
    X = M @ Phi # (N, m)
    return M, X # (N, n)


########################################################################################
########################################################################################
### Coherence

def calculate_coherence(A, B):
    """
    Calculates the coherence between the columns of two matrices A and B.

    Parameters:
    ----------
    A : np.ndarray
        Matrix of shape (q, m) where columns are the vectors to compare.
    B : np.ndarray
        Matrix of shape (q, m) where columns are the vectors to compare.

    Returns:
    -------
    float
        Maximum coherence between the columns of A and B.
    """
    # Normalize columns of A and B
    A_normalized = A / np.linalg.norm(A, axis=0, keepdims=True)
    B_normalized = B / np.linalg.norm(B, axis=0, keepdims=True)

    # Compute coherence matrix (absolute inner products between columns)
    coherence_matrix = np.abs(A_normalized.T @ B_normalized)

    # Set self-coherence to zero if A and B have the same columns
    if A.shape == B.shape and np.allclose(A, B):
        np.fill_diagonal(coherence_matrix, 0)

    # Return the maximum coherence value
    return np.max(coherence_matrix)

########################################################################################
########################################################################################
## Convex Programming

def solve_compressed_sensing_l1(X, y_star, EPSILON=1e-8):
    """
    Solve the l1-minimization problem to recover a :
    Minimize ||a||_1 subject to ||Xa - y*||_2 <= epsilon
    """
    a = cp.Variable(X.shape[1])
    objective = cp.Minimize(cp.norm(a, 1))
    #constraints = [X @ a = y]
    constraints = [cp.norm(X @ a - y_star, 2) <= EPSILON]  # tolerance for numerical precision
    problem = cp.Problem(objective, constraints)
    problem.solve()
    # Recovered sparse representation a
    return a.value

########################################################################################
########################################################################################
### Gradient Descent : Subgradient & Projected & Proximal (ISTA) Gradient Descent 

########################################################################################

def soft_threshold(a, threshold):
    """Applies the soft-thresholding operator element-wise to vector a."""
    return np.sign(a) * np.maximum(np.abs(a) - threshold, 0)

def soft_threshold_complex(a, threshold):
    """Applies the soft-thresholding operator to complex vector a."""
    magnitude = np.abs(a)
    phase = np.angle(a)
    # Shrink the magnitude while keeping the phase
    return np.maximum(magnitude - threshold, 0) * np.exp(1j * phase)

########################################################################################

def subgradient_descent(
    X, y_star, a_star, method="subgradient", learning_rate=0.1, beta_2=0., beta_1=0.1, init_scale=0.0,
    init_method = "identity", max_iter=10**3, eval_first=10**3, eval_step=10**3, 
    threshold_ajusted = -np.inf, # no ajustement,
    #threshold_ajusted = 1e-8, # ajust (keep only the values >= 1e-8 and set the rest to 0)
    tol=1e-6, 
    verbose=False
    ):
    """
    Solves the l1-minimization problem using Subgradient / Projected / Proximal (ISTA) Gradient Descent.
    Minimizes (1/2) ||X a - y_star||_2^2 + (beta_2/2) * ||a||_2^2 + beta_1 * ||a||_1.

    Parameters:
    ----------
    X : np.ndarray, Measurement matrix of shape (N, n).
    y_star : np.ndarray, Observation vector of shape (N,).
    a_star : np.ndarray, True signal for comparison, of shapes (n,)
    method : str, Optimization method to use (subgradient, proj_subgradient, ISTA).
    learning_rate : float, Step size for gradient updates.
    beta_1, beta_2 : float, Regularization parameter for l_1 and l_2.
    init_scale: float, initialization scale
    init_method : str, Initialization method for a.
    max_iter : int, Maximum number of iterations.
    eval_first : int, First iteration to evaluate the error.
    eval_step : int, Evaluate the error every eval_step iterations.
    adjust_threshold : float, threshold for ajustement (keep only the values >= threshold_ajusted and set the rest to 0)
    tol : float, Tolerance for convergence

    Returns:
    -------
    np.ndarray, Recovered sparse vector a of shape (n,)
    list, List of recovery errors over iterations.
    list, List of training errors over iterations.
    list, List of all recovered sparse vectors over iterations.
    list, List of all iterations.
    """

    assert method in ["subgradient", "proj_subgradient", "ISTA"]
    assert init_method in ["identity", "random", "least_square"]
    
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
    a_LS = np.linalg.pinv(X.conj().T @ X + beta_2) @ X.conj().T @ y_star # (n, )

    ###############################################################
    ###############################################################

    # Initialize A^{(1)}
    n = X.shape[1]
    is_complex = np.iscomplexobj(X)
    if is_complex :
        #a = np.zeros(n, dtype=complex)  # Initialize b as complex
        if init_method == "identity" :
            a = np.ones(n, dtype=complex)
        elif init_method == "random" :
            a = np.random.randn(n) / np.sqrt(n) + 1j * np.random.randn(n) / np.sqrt(n)
        elif init_method == "least_square" :
            a = a_LS + 0
    else :
        if init_method == "identity" :
            a = np.ones(n)  
        elif init_method == "random" :
            a = np.random.randn(n) / np.sqrt(n)
        elif init_method == "least_square" :
            a = a_LS + 0

    a = a * init_scale

    ###############################################################
    ##### Sub gradient : No projection
    ###############################################################
    if method == "subgradient" :
        Pi = lambda a, alpha_t : a # (n, )
    ###############################################################
    ##### Projected subgradient : Projection on the set of feasible solutions
    ###############################################################
    elif method == "proj_subgradient" :
        P = X.conj().T @ np.linalg.inv(X @ X.conj().T) # (n, N)
        Pi = lambda a, alpha_t : a - P @ (X @ a - y_star) # (n, )
    ###############################################################
    ##### Priximal gradient (ISTA) : Soft thresholding
    ###############################################################
    elif method == "ISTA" :
        beta_1_ista = beta_1 + 0
        beta_1 = 0 # no beta_1 for ISTA
        if is_complex :
            Pi = lambda a, alpha_t : soft_threshold_complex(a, alpha_t * beta_1_ista) # (n, )
        else :
            Pi = lambda a, alpha_t : soft_threshold(a, alpha_t * beta_1_ista) # (n, )

    ###############################################################
    ###############################################################

    train_errors = []
    errors = []

    a_ajusted = (np.abs(a)>=threshold_ajusted) * a # ajust (keep only the values > threshold_ajusted and set the rest to 0)
    y_ajusted = X @ a_ajusted
    err = np.linalg.norm(y_ajusted - y_star, 2) / np.linalg.norm(y_star, 2)
    recovery_error = np.linalg.norm(a_ajusted - a_star, 2) / np.linalg.norm(a_star, 2)

    train_errors.append(err)
    errors.append(recovery_error)

    ###############################################################
    ###############################################################

    all_a = []
    all_a.append(a+0)

    ###############################################################
    ###############################################################

    all_iterations = []
    all_iterations.append(0)

    ###############################################################
    ###############################################################

    for iteration in tqdm(list(range(1, max_iter+1))):
        ## Gradient the data-fitting term ||X a - y*||_2^2
        grad = X.conj().T @ (X @ a - y_star) # Use conjugate transpose for complex values
        if is_complex :
            # Gradient of the l2 term ||a||_2^2
            l2_norm = np.linalg.norm(a)
            grad = grad + beta_2 * (a/l2_norm if l2_norm != 0 else np.zeros_like(a))
            # Gradient of the l1 term ||a||_1
            h_a = np.sign(np.real(a)) + 1j * np.sign(np.imag(a))
            #h_a = (h_a!=0) * h_a + (h_a==0) * (np.random.choice([-1, 1], size=(n,)) + 1j * np.random.choice([-1, 1], size=(n,)))
            grad = grad + beta_1 * h_a
        else :
            # Gradient of the l2 term ||a||_2^2
            grad = grad + beta_2 * a
            # Gradient of the l1 term ||b||_1
            h_a = np.sign(a)
            #sub_gradient = (h_a!=0)*h_a + (h_a==0) * np.random.choice([-1, 1], size=(n,))
            grad = grad + beta_1 * h_a

        ## LR schedule
        alpha_t = learning_rate
        #alpha_t = learning_rate / iteration
        # if (iteration+1)%10**3 == 0 : learning_rate_t = learning_rate / 10**t
        #alpha_t = learning_rate / np.sqrt(iteration)
        #alpha_t = learning_rate / np.linalg.norm(sub_gradient)
        #alpha_t = err / np.linalg.norm(sub_gradient)**2

        ## Gradient step
        a = a - alpha_t * grad

        ## Projected subgradient & Proximal gradient
        a = Pi(a, alpha_t)

        if  iteration % eval_step == 0 or iteration < eval_first :
            ## Errors
            a_ajusted = (np.abs(a)>=threshold_ajusted) * a # ajust (keep only the values > threshold_ajusted and set the rest to 0)
            y_ajusted = X @ a_ajusted
            err = np.linalg.norm(y_ajusted - y_star, 2) / np.linalg.norm(y_star, 2)
            recovery_error = np.linalg.norm(a - a_star, 2) / np.linalg.norm(a_star, 2)

            train_errors.append(err)
            errors.append(recovery_error)
            all_a.append(a+0)
            all_iterations.append(iteration)

            if verbose :
                print(f"error : train={round(err,4)}, test={round(recovery_error,4)}")

        # ## Check for convergence
        # if  recovery_error < tol :
        #     print(f"Converged in {iteration} iterations.")
        #     break

    #print(a.round(3), a_star.round(3))

    return a, errors, train_errors, all_a, all_iterations, a_LS

########################################################################################


def deep_subgradient_descent(
    X, y_star, a_star, L=2, method="subgradient", learning_rate=0.1, beta_2=0., beta_1=0.1, init_scale=0.0,
    init_method = "identity", max_iter=10**3, eval_first=10**3, eval_step=10**3, 
    threshold_ajusted = -np.inf, # no ajustement,
    #threshold_ajusted = 1e-8, # ajust (keep only the values >= 1e-8 and set the rest to 0)
    tol=1e-6,
    verbose=False
    ):
    """
    Solves the l1-minimization problem using Subgradient / Projected / Proximal (ISTA) Gradient Descent.
    Minimizes (1/2) ||X a - y_star||_2^2 + (beta_2/2) * ||a||_2^2 + beta_1 * ||a||_1.

    Parameters:
    ----------
    X : np.ndarray, Measurement matrix of shape (N, n).
    y_star : np.ndarray, Observation vector of shape (N,).
    a_star : np.ndarray, True signal for comparison, of shapes (n,)
    method : str, Optimization method to use (subgradient, proj_subgradient, ISTA).
    learning_rate : float, Step size for gradient updates.
    beta_2, beta_1 : float, Regularization parameter for l_2 and l_1.
    init_scale: float, initialization scale
    init_method : str, Initialization method for a^{(1)}.
    max_iter : int, Maximum number of iterations.
    eval_first : int, First iteration to evaluate the error.
    eval_step : int, Evaluate the error every eval_step iterations.
    adjust_threshold : float, threshold for ajustement (keep only the values >= threshold_ajusted and set the rest to 0)
    tol : float, Tolerance for convergence

    Returns:
    -------
    np.ndarray, Recovered sparse vector a of shape (n,)
    list, List of recovery errors over iterations.
    list, List of training errors over iterations.
    list, List of all recovered sparse vectors over iterations.
    list, List of all iterations.
    """

    assert method in ["subgradient", "proj_subgradient", "ISTA"]
    assert init_method in ["identity", "random", "least_square"]
    assert L >= 1

    ###############################################################
    ###############################################################

    U, Sigma12, VT = np.linalg.svd(X, full_matrices=False) # (N, r), (r,), (n, r)
    Sigma = Sigma12  * Sigma12
    #a_LS = VT.T @ np.diag(1/(Sigma + beta_2)) @ VT @ X.T @ y_star
    print("0, learning_rate, 2 /  max(Sigma + beta_2)", 0, learning_rate, 2 / max(Sigma + beta_2))
    assert 0 < learning_rate < 2 / max(Sigma + beta_2)

    ###############################################################
    ###############################################################

    a_LS = np.linalg.pinv(X.conj().T @ X + beta_2) @ X.conj().T @ y_star # least square

    ###############################################################
    ###############################################################

    n = X.shape[1]
    as_list = [] # list of a_k
    a = np.ones(n)
    for k in range(L) :
        if init_method == "identity" : a_k = np.ones(n)  
        elif init_method == "random" : a_k = np.random.randn(n) / np.sqrt(n)
        elif init_method == "least_square" : a_k = a_LS + 0
        a_k = a_k * init_scale
        as_list.append(a_k + 0)
        a = a_k * a

    ###############################################################
    ##### Sub gradient : No projection
    ###############################################################
    if method == "subgradient" :
        Pi = lambda a, alpha_t : a # (n, )
    ###############################################################
    ##### Projected subgradient : Projection on the set of feasible solutions
    ###############################################################
    elif method == "proj_subgradient" :
        P = X.conj().T @ np.linalg.inv(X @ X.conj().T) # (n, N)
        Pi = lambda a, alpha_t : a - P @ (X @ a - y_star) # (n, )
    ###############################################################
    ##### Priximal gradient (ISTA) : Soft thresholding
    ###############################################################
    elif method == "ISTA" :
        beta_1_ista = beta_1 + 0
        beta_1 = 0 # no beta_1 for ISTA
        # if is_complex :
        #     Pi = lambda a, alpha_t : soft_threshold_complex(a, alpha_t * beta_1_ista) # (n, )
        # else :
        #     Pi = lambda a, alpha_t : soft_threshold(a, alpha_t * beta_1_ista) # (n, )
        
        Pi = lambda a, alpha_t : soft_threshold(a, alpha_t * beta_1_ista) # (n, )

    ###############################################################
    ###############################################################

    train_errors = []
    errors = []

    a_ajusted = (np.abs(a)>=threshold_ajusted) * a # ajust (keep only the values > threshold_ajusted and set the rest to 0)
    y_ajusted = X @ a_ajusted
    err = np.linalg.norm(y_ajusted - y_star, 2) / np.linalg.norm(y_star, 2)
    recovery_error = np.linalg.norm(a_ajusted - a_star, 2) / np.linalg.norm(a_star, 2)

    train_errors.append(err)
    errors.append(recovery_error)

    ###############################################################
    ###############################################################

    all_a = []
    all_a.append(a+0)

    ###############################################################
    ###############################################################

    all_iterations = []
    all_iterations.append(0)

    ###############################################################
    ###############################################################

    for iteration in tqdm(list(range(1, max_iter+1))):
        ## Gradient the data-fitting term ||X a - y*||_2^2
        grad_a = X.conj().T @ (X @ a - y_star) # Use conjugate transpose for complex values

        new_as_list = []
        for i in range(L) :
            # grad wrt a_i = prod_{k \ne i} a_k
            a_i = as_list[i]
            grad = np.ones(n)
            for k in range(i) : grad = grad * as_list[k]
            for k in range(i+1, L) : grad = grad * as_list[k]
            grad = grad_a * grad

            # Gradient of the l2 term ||a||_2^2
            grad = grad + beta_2 * a_i
            # Gradient of the l1 term ||a||_1
            h_a_i = np.sign(a_i)
            #sub_gradient = (h_a_i!=0)*h_a_i +(h_a_i==0) * np.random.choice([-1, 1], size=(n,))
            grad = grad + beta_1 * h_a_i

            ## LR schedule
            alpha_t = learning_rate
            #alpha_t = learning_rate / iteration
            # if (iteration+1)%10**3 == 0 : learning_rate_t = learning_rate / 10**t
            #alpha_t = learning_rate / np.sqrt(iteration)
            #alpha_t = learning_rate / np.linalg.norm(grad_a)
            #alpha_t = err / np.linalg.norm(grad_a)**2

            ## Gradient step
            a_i = a_i - alpha_t * grad

            ## Projected subgradient & Proximal gradient
            #a_i = Pi(a_i, alpha_t)

            new_as_list.append(a_i+0)

        # update a
        as_list = new_as_list
        a = np.ones(n)
        for k in range(L) :
            a = as_list[k] * a

        if  iteration % eval_step == 0 or iteration < eval_first :
            ## Errors
            a_ajusted = (np.abs(a)>=threshold_ajusted) * a # ajust (keep only the values > threshold_ajusted and set the rest to 0)
            y_ajusted = X @ a_ajusted
            err = np.linalg.norm(y_ajusted - y_star, 2) / np.linalg.norm(y_star, 2)
            recovery_error = np.linalg.norm(a_ajusted - a_star, 2) / np.linalg.norm(a_star, 2)

            train_errors.append(err)
            errors.append(recovery_error)
            all_a.append(a+0)
            all_iterations.append(iteration)

            #print(as_)

        # ## Check for convergence
        # if recovery_error < tol :
        #     print(f"Converged in {iteration} iterations.")
        #     break

    #print(a.round(3), a_star.round(3))

    return a, errors, train_errors, all_a, all_iterations, a_LS

########################################################################################
########################################################################################

def get_gradient(X, y_star, all_a, beta_1, is_complex=False, ord=2):
    """
    Calculate the gradient norms for all sparse vectors in all_a.
    """  
    norms_grad_1 = []
    norms_grad_2 = []
    norms_grad_ratio = []

    for a in tqdm(all_a):

        grad_1 = X.conj().T @ (X @ a - y_star) # Use conjugate transpose for complex values
        ng_1 = np.linalg.norm(grad_1, ord=ord) #/ np.linalg.norm(a, ord=ord)

        threshold_ajusted = 1e-3
        a_ajusted = (np.abs(a)>=threshold_ajusted) * a # ajust (keep only the values > threshold_ajusted and set the rest to 0)
        if is_complex :
            grad_2 = np.sign(np.real(a_ajusted)) + 1j * np.sign(np.imag(a_ajusted))
        else :
            grad_2 = beta_1 * np.sign(a_ajusted)
        ng_2 = np.linalg.norm(grad_2, ord=ord) #/ np.linalg.norm(a_ajusted, ord=ord)

        norms_grad_1.append(ng_1)
        norms_grad_2.append(ng_2)
        norms_grad_ratio.append(ng_2 / ng_1)

    return norms_grad_1, norms_grad_2, norms_grad_ratio


########################################################################################
########################################################################################