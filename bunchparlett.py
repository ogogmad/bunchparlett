"""The main function here is "LDL" for computing an LDL decomposition using
the Bunch-Parlett pivoting strategy.

An LDL decomposition of a Hermitian matrix M (possibly indefinite) is a
factorization of the form

  P * M * P' == L * D * L' **-1
  
where P is a permutation matrix, L is a lower triangular matrix with '1's
along the diagonal, D is a block-diagonal matrix with 1x1 and 2x2 blocks,
and A' for any given matrix A denotes the conjugate-transpose of A.

These types of decompositions are useful for solving systems of equations
where the system is given by a (possibly indefinite) Hermitian matrix."""

from sympy import *

def bppivot(M):
    """Compute pivot P and value of s using Bunch-Parlett. See Section 4.4.3
    of Matrix Computations, 4th Edition for what these symbols mean."""
    alpha = (1 + sqrt(17))/8
    mu0 = 0
    for i in range(M.rows):
        for j in range(M.rows):
            if abs(M[i,j]) > mu0:
                mu0 = abs(M[i,j])
                exi, exj = i, j
    mu1 = 0
    for i in range(M.rows):
        if abs(M[i,i]) > mu1:
            mu1 = abs(M[i,i])
            exii = i
    if mu1 >= alpha * mu0:
        s = 1
        P = eye(M.rows).tolist()
        P[0], P[exii] = P[exii], P[0]
        return Matrix(P), s
    else:
        s = 2
        P = eye(M.rows).tolist()
        P[0], P[exi] = P[exi], P[0]
        P[1], P[exj] = P[exj], P[1]
        return Matrix(P), s

def getecb(M,s):
    """Get the E, C and B components of a Hermitian matrix M. See
    Section 4.4.3 of Matrix Computations, 4th Edition for what these mean."""
    E = M[:s,:s]
    C = M[s:,:s]
    B = M[s:,s:]
    return E, C, B

def bunchparlett(M):
    """Compute P and L factors of M using Bunch-Parlett. The algorithm is
    recursive."""
    if M.rows <= 2:
        P = eye(M.rows)
        L = eye(M.rows)
        return P, L
    n = M.rows
    P1, s = bppivot(M)
    Mprime = P1 * M * P1.T
    E, C, B = getecb(Mprime,s)
    L1 = Matrix([[eye(s),zeros(s,n-s)],[C * E ** -1, eye(n-s)]])
    Prest,  Lrest = bunchparlett(B - C * E**-1 * C.H)
    L = Matrix([[eye(s), zeros(s,n-s)],[Prest * C * E**-1, Lrest]])
    P = Matrix([[eye(s),zeros(s,n-s)],[zeros(n-s,s),Prest]]) * P1
    return P, L

def LDL(M):
    """Compute LDL factorization using Bunch-Parlett.
    The matrix D is block-diagonal with blocks of size either 1x1 or 2x2.
    
    Input: A Hermitian matrix M.
    Output: L, D and P such that M == P.T * L * D * L.H**-1 * P"""
    P, L = bunchparlett(M)
    L = simplify_split(L)
    D = L ** -1 * P * M * P.H * L.H ** -1
    return L, D, P

