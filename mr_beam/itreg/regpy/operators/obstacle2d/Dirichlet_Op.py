# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 20:12:14 2019

@author: Björn Müller
"""
# from regpy.operators.obstacle2d.Curves.StarTrig import StarTrig
from .functions.operator import op_S
from .functions.operator import op_T
from .functions.operator import op_K
from .functions.farfieldmatrix import farfield_matrix
from .functions.farfieldmatrix import farfield_matrix_trans
from .set_up_iopdata import setup_iop_data

from .. import Operator

import numpy as np
import scipy.linalg as scla


class DirichletOp(Operator):

    def __init__(self, domain, codomain=None, **kwargs):
        codomain = codomain or domain
#        codomain.discr.iscomplex()
        self.kappa = 3            # wave number
        self.N_ieq = 32           # 2*F_ieq is the number of discretization points
        self.N_ieq_synth = 32     # 2*N_ieq is the number of discretization points for
        """the boundary integral equation when computing synthetic data (choose different
         to N_ieq to avoid invere crime)"""
        self.meas_directions = 64 # measurement directions
#        self.inc_directions = np.asarray([1,0]).reshape((2,1))
        self.bd = StarTrig(64)


        self.op_name = 'DirichletOp'
        self.syntheticdata_flag = True
        self.kappa = 3    # wave number
        # directions of incident waves
        self.N_inc = 1
        t=2*np.pi*np.arange(0, self.N_inc)/self.N_inc
        #t = 0.5;
        self.inc_directions = np.append(np.cos(t), np.sin(t)).reshape((2, self.N_inc))


        self.N_meas = 64
        t= 2*np.pi*np.arange(0, self.N_meas)/self.N_meas
        self.N_ieq = 64
        self.meas_directions = np.append(np.cos(t), np.sin(t)).reshape((2, self.N_meas))


        self.true_curve = 'nonsym_shape'
        self.noiselevel = 0.000
        self.N_FK = 64
        """t= 2*pi*[1:p.N_FK]'/(2*p.N_FK);
        %p.init_guess = [cos(t);sin(t)];
        %'peanut','round_rect', 'apple',
        %'three_lobes','pinched_ellipse','smoothed_rectangle','nonsym_shape'"""







        """ 2 dimensional obstacle scattering problem with Dirichlet boundary condition
         see sec. 4 in T. Hohage "Logarithmic convergence rates of the iteratively
         regularized Gauss-Newton method for an inverse potential
         and an inverse scattering problem" Inverse Problems 13 (1997) 1279�1299"""


        self.dudn=None  # normal derivative of total field at boundary
        """ weights of single and double layer potentials"""
        self.wSL=-1*complex(0,1)*self.kappa
        self.wDL=1
        """ LU factors + permuation for integral equation matrix"""
        self.L=None
        self.U=None
        self.perm=None
        self.FF_combined=None
        self.op_name='DirichletOp'
        """ use a mixed single and double layer potential ansatz with
             weights wSL and wDL"""
        self.Ydim = 2* np.size(self.meas_directions) * np.size(self.inc_directions, 1)
        super().__init__(domain=domain, codomain=codomain)


    def _eval(self, coeff, **kwargs):


        """ solve the forward Dirichlet problem for the obstacle parameterized by
         coeff. Quantities needed again for the computation of derivatives and
         adjoints are stored as members of F."""

        """compute the grid points of the boundary parameterized by coeff and derivatives
        of the parametrization and save these quantities as members of F.bd"""
        self.bd.bd_eval(coeff, 2*self.N_ieq,2)
        Iop_data = setup_iop_data(self.bd,self.kappa)
        if self.wSL!=0:
            Iop = self.wSL*op_S(self.bd,Iop_data)
        else:
            Iop = np.zeros(np.size(self.bd.z,1),np.size(self.bd.z,1))
        if self.wDL!=0:
            Iop = Iop + self.wDL*(np.diag(self.bd.zpabs)+ op_K(self.bd,Iop_data))
        self.dudn = np.zeros((2*self.N_ieq,np.size(self.inc_directions,1)))
        FF_SL = farfield_matrix(self.bd,self.meas_directions,self.kappa,-1.,0.)
        self.perm_mat, self.L, self.U =scla.lu(Iop)
        self.perm=self.perm_mat.dot(np.arange(0, np.size(self.bd.z,1)))
        self.FF_combined = farfield_matrix(self.bd,self.meas_directions,self.kappa, \
                                           self.wSL,self.wDL)
        farfield = []

        for l in range(0, np.size(self.inc_directions, 1)):
            rhs = 2*np.exp(complex(0,1)*self.kappa*self.inc_directions[:,l].T.dot(self.bd.z))*  \
                (self.wDL*complex(0,1)*self.kappa*self.inc_directions[:,l].T.dot(self.bd.normal) \
                                         +self.wSL*self.bd.zpabs)

            self.dudn[:, l]=np.linalg.solve(self.L.T, \
                     np.linalg.solve(self.U.T, rhs[self.perm.astype(int)]))
            complex_farfield = np.dot(FF_SL, self.dudn[:,l])
            farfield=np.append(farfield, complex_farfield)
        return farfield

    def _derivative(self, h):
            der = []
            for l in range(0, np.size(self.inc_directions,1 )):
                rhs = - 2*self.dudn[:,l].dot(self.bd.der_normal(h)) * self.bd.zpabs.T
                phi=np.linalg.solve(self.U, np.linalg.solve(self.L, rhs[self.perm.astype(int)]))
                complex_farfield = self.FF_combined.dot(phi)
                der=np.append(der, complex_farfield)
            return der

    def _adjoint(self, g):
            res = np.zeros(2*self.N_ieq)
            rhs = np.zeros(2*self.N_ieq)
            N_FF = np.size(self.meas_directions,1)
            for  l in range(0, np.size(self.inc_directions,1)):
                g_complex=g[2*(l)*N_FF+np.arange(0, N_FF)]
                phi = self.FF_combined.T.dot(g_complex)

                rhs[self.perm.astype(int)]=np.linalg.solve(self.L, \
                    np.linalg.solve(self.U, phi.T)).T
                res = res -2*rhs*np.conjugate(self.dudn[:,l]).real
            adj = self.bd.adjoint_der_normal(res * self.bd.zpabs.T)
            return adj

    def accept_proposed(self, positions):
        return True
