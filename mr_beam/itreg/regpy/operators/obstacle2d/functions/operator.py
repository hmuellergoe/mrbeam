# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:38:26 2019

@author: Björn Müller
"""

import numpy as np
import scipy.linalg as scla
import scipy.sparse as scsp

def op_S(bd,dat):
    r""" set up matrix representing the single layer potential operator
    % (S\phi)(z(t)):=2|z'(t)|\int_0^{2\pi} \Phi(z(t),z(s)) |z'(s)| \phi(s) ds
    % Our implementation is based section 3.5 of the monograph "Inverse Acoustic
    % and Electomagnetic Scattering Theory" by R. Kress and D. Colton.
    % As opposed to this reference we multiplied by |z'(t)| to obtain a
    % complex symmetric matrix """

    dim = np.size(bd.z,1)
    M1 = -1/(2*np.pi)*dat.bessH0.real
    M = complex(0,1)/2*dat.bessH0
    M1_logsin = M1* dat.logsin
    M2 = M - M1_logsin
    for j  in range(0, dim):
        M2[j, j] = (complex(0, 1)/2 - dat.Euler_gamma/np.pi - 1/np.pi*np.log(dat.kappa/2*bd.zpabs[j]))

    S =  2*np.pi* (M1 * dat.logsin_weights + M2/dim) * (bd.zpabs.T.dot(bd.zpabs))
    return S



def op_K(bd,dat):
    r""" set up matrix representing the double layer potential operator
    % (K\phi)(z(t)):=2|z'(t)|\int_0^{2\pi}{\frac{\partial\Phi(z(t),z(s))}{\partial\nu(z(s))}
    % |z'(s)| \phi(z(s)) ds
    % Our implementation is based section 3.5 of the monograph "Inverse Acoustic
    % and Electomagnetic Scattering Theory" by R. Kress and D. Colton.
    % As opposed to this reference we multiplied by |z'(t)|."""

    dim = np.size(bd.z,1)
    kappa = dat.kappa

    aux = np.dot(bd.z.T, bd.normal) - np.dot(np.ones((dim, 2)),(bd.normal*bd.z))
    H = 0.5*complex(0, 1)*kappa**2* aux * dat.bessH1quot
    H1 = -kappa**2/(2*np.pi)*aux * dat.bessH1quot.real
    H2 = H - H1*dat.logsin
    for j in range(0, dim):
        H1[j, j] = 0
        H2[j, j] = 1/(2*np.pi) * (np.dot(bd.normal[:,j].T, bd.zpp[:,j])) / bd.zpabs[j]**2

    K =  (2*np.pi) * scsp.spdiags(bd.zpabs.T,0,dim,dim) * ( H1 * dat.logsin_weights + H2/dim )
    return K




def op_T(bd,dat):
    r"""set up matrix representing the normal derivative of the double-layer potential
    %(T\phi)(z(t)):=2|z'(t)|\frac{\partial}{\partial\nu(z(t))}
    %\int_0^{2\pi}{\frac{\partial\Phi(z(t),z(s))}{\partial\nu(z(s))}|z'(s)|%\phi(z(s))ds
    %Our implementation is based on the paper "On the numerical solution of a
    %hypersingular integral equation in scattering theory" by Rainer Kress,
    %J. Comp. Appl. Math. 61:345-360, 1995, and uses the formula
    %T = \frac{d}{ds}S\frac{d\phi}{ds} + \kappa^2\nu\cdot S(\nu\phi)"""

    dim=np.size(bd.z,1)
    z = bd.z
    zp = bd.zp
    zpp = bd.zpp
    zppp = bd.zppp
    zpabs = bd.zpabs
    kappa = dat.kappa

    N_tilde = kappa*(z.T.dot(zp) -  np.ones((dim,1)).dot(np.sum(z*zp, 0).reshape((1, dim))) / (dat.kdist+1e-5))
    N_tilde = -N_tilde.T.dot(N_tilde)
    Nker = complex(0,1)/2*N_tilde*( kappa**2*dat.bessH0 - 2*kappa**2*dat.bessH1quot) \
        +complex(0,1)*kappa**2/2*(zp.T.dot(zp)) * dat.bessH1quot  \
        + scla.toeplitz(np.append(np.asarray([np.pi/2]), 1/(4*np.pi)*np.sin(np.pi*np.arange(1, dim)/dim)**(-2)))
    N1  = -1/(2*np.pi)*N_tilde * (kappa**2*dat.bessH0.real-2*kappa**2*dat.bessH1quot.real)  \
        - kappa**2/(2*np.pi)* (zp.T.dot(zp)) * dat.bessH1quot.real
    N2 = Nker - N1*dat.logsin
    for j in range(0, dim):
        N1[j, j] = -kappa**2*zpabs[j]**2/(4*np.pi)
        N2[j, j] = kappa**2*zpabs[j]**2/(4*np.pi)  \
            * ( np.pi*complex(0,1) -1 -2*dat.Euler_gamma - 2*np.log(kappa*zpabs[j]/2) ) \
            + 1/12/np.pi +  1/(2*np.pi) * np.sum(zp[:,j]*zpp[:,j])**2 / zpabs[j]**4 \
             - 1/(4*np.pi) * np.sum(zpp[:,j]**2)                    / zpabs[j]**2 \
             - 1/(6*np.pi) * np.sum(zp[:,j]*zppp[:,j])             / zpabs[j]**2


    T_weights = np.zeros(dim)
    T_weights[np.arange(1, dim, 2)]=(1/dim) * np.sin(np.pi*np.arange(1, dim, 2)/dim)**(-2)
    T_weights[0] = -dim/4

    T =  scla.toeplitz(T_weights) \
        - 2*np.pi*( N1*dat.logsin_weights + N2/dim )  \
        + kappa**2*op_S(bd,dat)* (zp.T.dot(zp)) / (zpabs.T.dot(zpabs))
    return T
