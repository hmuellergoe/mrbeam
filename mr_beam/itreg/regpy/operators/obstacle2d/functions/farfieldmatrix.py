# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:19:44 2019

@author: Björn Müller
"""

import numpy as np

def farfield_matrix(bd,dire,kappa,weightSL,weightDL):
    """set up matrix corresponding to the far field evaluation of a combined
    %single and double layer potential with weigths weightSL and weightDL, rsp."""
#    print(dire)
    FFmat =  np.pi / (np.size(bd.z,1)*np.sqrt(8*np.pi*kappa)) * np.exp(-complex(0,1)*np.pi/4) \
         * (weightDL*kappa*dire.T.dot(bd.normal) +complex(0,1)*weightSL*np.matlib.repmat(bd.zpabs,np.size(dire,1),1)) \
         * np.exp(-complex(0,1)*kappa* (dire.T.dot(bd.z)))
    return FFmat


def farfield_matrix_trans(bd,dire,kappa,weightSL,weightDL):
    """set up matrix corresponding to the far field evaluation of a combined
    %single and double layer potential with weigths weightSL and weightDL, rsp.
    %according to eq. 4.12 by exploiting the linearity of the integral."""
    FFmat_a = 2*np.pi / (np.size(bd.z,1)*np.sqrt(8*np.pi*kappa)) * np.exp(complex(0,1)*np.pi/4) \
            * (-complex(0,1)*weightDL*kappa*dire.T*bd.normal) \
            * np.exp(-complex(0,1)*kappa* (dire.T * bd.z))

    FFmat_b = 2*np.pi / (np.size(bd.z,1)*np.sqrt(8*np.pi*kappa)) * np.exp(complex(0,1)*np.pi/4) \
            * (weightSL*np.matlib.repmat(bd.zpabs,np.size(dire,1),1)) \
            * np.exp(-complex(0,1)*kappa* (dire.T * bd.z))

# Is there really a komma?
    FFmat = [FFmat_a, FFmat_b]
    return FFmat
