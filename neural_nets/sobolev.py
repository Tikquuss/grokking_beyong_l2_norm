import torch
import numpy as np
import math

############################################################################
################################# Jacobian #################################
############################################################################

def jacobian_batched(varphi:callable, X, create_graph=False):
    """
    varphi : R^q -----> R^d
    X : R^{N x q} 
    Compute the jacobian of the function varphi : R^q -----> R^d wrt to X in R^{N x q}
    D = [∂varphi(X_i) / ∂X_i] in R^{N x d x q}
    """
    # Thanks https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/5
    def _varphi_sum(X):
        return varphi(X).sum(dim=0) # (*q,)
    D = torch.autograd.functional.jacobian(_varphi_sum, X, create_graph=create_graph, vectorize=True) # (d, N, *q)
    #D = D.permute(1, 0, 2) # (N, d, q*)
    D = D.transpose(1, 0) # (N, d, q*)
    return D

############################################################################
################################# Hessian ##################################
############################################################################

def hessian_batched(varphi:callable, X, create_graph=False):
    """
    varphi : R^q -----> R^d
    X : R^{N x q} 
    Compute the hessian of the function varphi : R^q -----> R^d wrt to X in R^{N x q}
    H = [∂²varphi(X)_i / ∂X_j∂X_k] in R^{N x q x d x d}
    """
    def wrapped_jacobian(X):
        J = jacobian_batched(varphi, X, create_graph=True) # (N, d, q)
        if J.shape[1]==1 : J = J[:,0] # (N, q)
        return J# (N, d, q) or (N, q)
    H = jacobian_batched(wrapped_jacobian, X, create_graph=create_graph) # (d, q, N, d) or (N, q, q) 
    if H.shape[0]!=1 : # d!=1
        H = H.permute(2, 1, 0, 3).contiguous() # (N, q, d, d)
    return H


############################################################################
############################### 2 layers NNs ###############################
############################################################################

def derivatives_2layers_nn(
    X: torch.Tensor,        # shape (N, d)
    W: torch.Tensor,        # shape (h, d)
    V: torch.Tensor,        # shape (C, h); if C=1 then shape (1, h)
    phi,                    # function: R -> R (elementwise), e.g. torch.sin
    phi_prime,              # derivative of phi
    phi_doubleprime=None    # second derivative of phi (only used if C=1)
):
    """
    Compute:
      1) Y_i = V phi(W X_i), ie Y = phi(X W^T) V^T  (shape (N, C))
      2) J(X), the Jacobian dY/dX, J_i(X) = dY_i(X)/dX_i = V diag(phi'(W X_i )) W = W^T diag(phi'(W X_i )) v for C = 1
         - shape (N, C, d) if C>1
         - shape (N, d)   if C=1
      3) H(X): The Hessian dJ(X)/dX (shape (N, C, d, d))

    Arguments:
      X: (N, d) input
      W: (h, d) weight
      V: (C, h) output weight
      phi: elementwise activation function
      phi_prime: elementwise derivative of phi
      phi_doubleprime: elementwise 2nd derivative of phi (only needed if C=1)

    Returns:
      Y, J, H
        - Y: (N, C)
        - J: (N, C, d) if C>1, else (N, d)
        - H: (N, d, d) if C>1, else (N, C, d, d)
    """

    # --- 1) Forward pass: Y ---
    z = X @ W.transpose(0, 1)  # X W^T  -> shape (N, h)
    a = phi(z) # a = phi(z)   -> shape (N, h)
    Y = a @ V.transpose(0, 1) #  # Y = a V^T  -> shape (N, C)

    N, d = X.shape
    C, h = V.shape

    phiprime_z = phi_prime(z) # phi'(z): shape (N, h)

    # --- 2) Construct J(X) of shape (N, C, d) ---
    # We'll store J in (N, C, d).
    J = torch.zeros(N, C, d, dtype=X.dtype, device=X.device)
    for i in range(N):
        # diag(phi'(z[i])) W => can be done by elementwise multiply
        # shape (h, d). We'll denote DW = ...
        DW = phiprime_z[i].unsqueeze(1) * W  # (h, d)
        J[i] = V @ DW # (C, h) x (h, d) => (C, d)

    # By default, no Hessian
    H = None

    # --- 3) Construct H(X), if phi_doubleprime is provided ---
    if phi_doubleprime is not None:
        # We'll allocate different shapes depending on C
        if C == 1:
            # H is shape (N, d, d)
            H = torch.zeros(N, d, d, dtype=X.dtype, device=X.device)
        else:
            # H is shape (N, C, d, d)
            H = torch.zeros(N, C, d, d, dtype=X.dtype, device=X.device)

        phidoubleprime_z = phi_doubleprime(z)  # shape (N, h)

        # We'll do a nested loop: over samples, and (if C>1) over each output c
        for i in range(N):
            # row_factors for phi''(z[i]) => shape (h,)
            phi2_i = phidoubleprime_z[i]  # shape (h,)

            if C == 1:
                # We have a single row of V => shape (1, h).
                # Flatten to v => (h,)
                v = V.view(-1)
                # row_factors = v * phi''(z[i]) => shape (h,)
                row_factors = v * phi2_i
                # M = diag(row_factors) W => do elementwise multiply
                M = row_factors.unsqueeze(1) * W   # shape (h, d)
                # W^T @ M => (d, d)
                H[i] = W.t() @ M
            else:
                # multi-output => we do each output dimension c
                for c_idx in range(C):
                    # v_c => shape (h,)
                    v_c = V[c_idx]  # the c-th row
                    row_factors = v_c * phi2_i  # shape (h,)
                    M = row_factors.unsqueeze(1) * W  # (h, d)
                    # H[i, c_idx] => (d, d)
                    H[i, c_idx] = W.t() @ M
        if C == 1:
            # H is shape (N, d, d)
            H = H.unsqueeze(1) # (N, C, d, d)

    return Y, J, H # (N, C), (N, C, d), (N, C, d, d)

############################################################################



def pack_YJH(Y, J, H):
    """
    Y: (N, C)
    J: (N, C, d)
    H: (N, C, d, d)

    Return Z: (N, C + C*d + C*d^2)
    """
    N, C = Y.shape
    # J has shape (N, C, d) => flatten to (N, C*d)
    N2, C2, d = J.shape
    assert N2 == N and C2 == C

    # H has shape (N, C, d, d) => flatten to (N, C*d^2)
    N3, C3, d2, d3 = H.shape
    assert N3 == N and C3 == C and d2 == d and d3 == d

    J_flat = J.view(N, C*d)       # (N, C*d)
    H_flat = H.view(N, C*d*d)     # (N, C*d^2)

    # Concatenate along dim=1 => shape (N, C + C*d + C*d^2)
    Z = torch.cat([Y, J_flat, H_flat], dim=1)
    return Z

def unpack_YJH(Z, N, C=None, d=None):
    """
    Z: (N, C + C*d + C*d^2)
    Return original (Y, J, H):
      - Y: (N, C)
      - J: (N, C, d)
      - H: (N, C, d, d)
    """
    assert C is not None or d is not None
    if C is None :
        C = Z.shape[1] // (1 + d + d**2)
    if d is None :
        d = infer_d_from_CZshape1(C, Z.shape[1])

    # slice sizes
    size_Y = C
    size_J = C*d
    size_H = C*d*d

    Y_rec   = Z[:, :size_Y]                    # (N, C)
    J_slice = Z[:, size_Y : size_Y + size_J]   # (N, C*d)
    H_slice = Z[:, size_Y + size_J : ]         # (N, C*d^2)
    # Reshape
    J_rec = J_slice.view(N, C, d)          # (N, C, d)
    H_rec = H_slice.view(N, C, d, d)       # (N, C, d, d)
    return Y_rec, J_rec, H_rec

def infer_d_from_CZshape1(C, Z_shape1):
    """
    Solve for d in the equation:  C(1 + d + d^2) = Z_shape1.
    Returns d as an integer if it exists, or raises ValueError otherwise.
    """
    # M = Z_shape1 / C
    if Z_shape1 % C != 0:
        raise ValueError(
            f"Z.shape[1] = {Z_shape1} is not divisible by C={C}, "
            f"so we can't write it as C(1 + d + d^2)."
        )
    M = Z_shape1 // C  # integer division

    # Solve d^2 + d + (1 - M) = 0 => d = (-1 +/- sqrt(4M - 3)) / 2
    disc = 4*M - 3
    if disc < 0:
        raise ValueError(
            f"No real solution for d, because 4*M - 3 = {disc} < 0. "
            f"Check if (N, C, d) setup is valid."
        )

    sqrt_disc = int(math.isqrt(disc))  # integer sqrt
    if sqrt_disc * sqrt_disc != disc:
        # It's possible that disc is not a perfect square,
        # so the solution might not be an integer
        raise ValueError(
            f"sqrt(4*M - 3) = sqrt({disc}) is not an integer, "
            f"so d would not be integer."
        )

    # Try the positive root: (-1 + sqrt_disc) / 2
    numerator = -1 + sqrt_disc
    if numerator % 2 != 0:
        raise ValueError(
            f"(-1 + sqrt_disc) is not divisible by 2 => no integer d. "
            f"Got numerator={numerator}."
        )
    d_candidate = numerator // 2

    # Finally check if indeed 1 + d_candidate + d_candidate^2 == M
    if 1 + d_candidate + d_candidate**2 != M:
        raise ValueError(
            f"Candidate d={d_candidate} does not satisfy 1 + d + d^2 = M."
        )

    return d_candidate

############################################################################
############################################################################

if __name__ == "__main__":
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using device: {device}')

    seed=45
    torch.manual_seed(seed)

    q, d, N = (2, 1), 3, 4
    #q, d, N = (28, 28), 2 * 28 * 28, 100
    
    ##### Model
    input_dim = np.prod(q)
    #model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_dim, d))
    #W = model[-1].weight.data # (d, *q)

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_dim, d), torch.nn.Linear(d, d))
    W = model[-1].weight.data @ model[-2].weight.data # (d, d) x (d, *q)
    # #########################

    X = torch.randn(N, *q, requires_grad=True)  # (N, q)
    D = jacobian_batched(model, X) # (N, d, *q)
    D = D.reshape(N, d, input_dim)
    print(all([torch.allclose(D[i], W, atol=1e-6) for i in range(N)]))



    N, C, d = 2, 3, 4  # small shapes for illustration
    Y = torch.randn(N, C)
    J = torch.randn(N, C, d)
    H = torch.randn(N, C, d, d)
    Z = pack_YJH(Y, J, H)
    print("Z shape =", Z.shape)  # should be (N, C + C*d + C*d^2)
    # i.e. (2, 3 + 12 + 48) = (2, 63)

    Y_rec, J_rec, H_rec = unpack_YJH(Z, N, C=None, d=d)

    print("Recovered shapes:")
    print(" Y_rec =", Y_rec.shape)  # (2, 3)
    print(" J_rec =", J_rec.shape)  # (2, 3, 4)
    print(" H_rec =", H_rec.shape)  # (2, 3, 4, 4)
