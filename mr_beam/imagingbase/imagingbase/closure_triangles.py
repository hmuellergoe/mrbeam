import os
import numpy as NP
import numpy.linalg as LA
from ehtim.imaging.pol_imager_utils import qimage, uimage

def find_list_in_list(reference_array, inp):

    """
    ---------------------------------------------------------------------------
    Find occurrences of input list in a reference list and return indices 
    into the reference list

    Inputs:

    reference_array [list or numpy array] One-dimensional reference list or
                    numpy array in which occurrences of elements in the input 
                    list or array will be found

    inp             [list or numpy array] One-dimensional input list whose 
                    elements will be searched in the reference array and 
                    the indices into the reference array will be returned

    Output:

    ind             [numpy masked array] Indices of occurrences of elements 
                    of input array in the reference array. It will be of same 
                    size as input array. For example, 
                    inp = reference_array[ind]. Indices for elements which are 
                    not found in the reference array will be masked.
    ---------------------------------------------------------------------------
    """

    try:
        reference_array, inp
    except NameError:
        raise NameError('Inputs reference_array, inp must be specified')

    if not isinstance(reference_array, (list, NP.ndarray)):
        raise TypeError('Input reference_array must be a list or numpy array')
    reference_array = NP.asarray(reference_array).ravel()

    if not isinstance(inp, (list, NP.ndarray)):
        raise TypeError('Input inp must be a list or numpy array')
    inp = NP.asarray(inp).ravel()

    if (inp.size == 0) or (reference_array.size == 0):
        raise ValueError('One or both inputs contain no elements')

    sortind_ref = NP.argsort(reference_array)
    sorted_ref = reference_array[sortind_ref]
    ind_in_sorted_ref = NP.searchsorted(sorted_ref, inp)
    ii = NP.take(sortind_ref, ind_in_sorted_ref, mode='clip')
    mask = reference_array[ii] != inp
    ind = NP.ma.array(ii, mask=mask)

    return ind

def remove_scaling_in_minkoski_dots(mdp, denom_ind=0): 
    # mdp = Minkowski dot products
    # mdp has shape (nruns,n_dotproducts)
    # denom_ind could be anything from 0 to mdp.shape[-1]-1 
    # output invariants have shape (nruns,n_dotproducts-1)
    ci = mdp / mdp[...,[denom_ind]]
    rest_ind = NP.where(NP.arange(mdp.shape[-1]) != denom_ind)[0]
    return ci[...,rest_ind]

def unpack_realimag_to_real(inp, arrtype='matrix'):
    # arrtype='matrix' means 2x2 polarimetric case, otherwise scalars
    inshape = inp.shape
    outarr = NP.concatenate((inp.real[...,NP.newaxis], inp.imag[...,NP.newaxis]), axis=-1) # Convert from (...,N) complex to (...,N,2) real for scalars or from (...,N,2,2) complex to (...,N,2,2,->2) real for 2x2 matrices
    if arrtype == 'matrix':
        outarr = outarr.reshape(inshape[:-3]+(-1,)) # Convert from (...,N,2,2,->2) real to (...,8N) real
    elif arrtype == 'scalar':
        outarr = outarr.reshape(inshape[:-1]+(-1,)) #  Convert from (...,N,2) real to (...,2N) real
    return outarr

def repack_real_to_realimag(inp, arrtype='matrix'):
    # arrtype='matrix' means 2x2 polarimetric case, otherwise scalars
    inshape = inp.shape
    outarr = inp.reshape(inshape[:-1]+(-1,2)) # Convert the array from (...,2N) to (...,N,2) real float values
    outarr = outarr[...,0] + 1j * outarr[...,1] # (...,N) complex values
    if arrtype == 'matrix':
        outarr = outarr.reshape(inshape[:-1]+(-1,2,2)) # Convert from (...,4M) complex to (...,M,2,2) complex
    return outarr

def hermitian(inparr, axes=(-2,-1)):
    # axes denotes which two axes are to be Hermitian-transposed
    if not isinstance(inparr, NP.ndarray):
        raise TypeError('Input array inparr must be a numpy array')
    if inparr.ndim == 1:
        inparr = inparr.reshape(1,-1)
    
    if axes is None:
        axes = NP.asarray([-2,-1])
    if not isinstance(axes, (list,tuple,NP.ndarray)):
        raise TypeError('Input axes must be a list, tuple, or numpy array')
    axes = NP.asarray(axes).ravel()
    if axes.size != 2:
        raise ValueError('Input axes must be a two-element list,tuple, or numpy array') 
    negind = NP.where(axes < 0)[0]    
    if negind.size > 0:
        axes[negind] += inparr.ndim # Convert negative axis numbers to positive
    if axes[0] == axes[1]:
        raise ValueError('The two entries in axes cannot be the same')

    return NP.swapaxes(inparr, axes[0], axes[1]).conj()

def hat(inparr, axes=None):
    # axes denotes which two axes are to be Hermitian-transposed. If None, defaults to (-2,-1)
    if not isinstance(inparr, NP.ndarray):
        raise TypeError('Input array inparr must be a numpy array')
    if inparr.ndim == 1:
        inparr = inparr.reshape(1,-1)
    
    if axes is None:
        axes = NP.asarray([-2,-1])
    if not isinstance(axes, (list,tuple,NP.ndarray)):
        raise TypeError('Input axes must be a list, tuple, or numpy array')
    axes = NP.asarray(axes).ravel()
    if axes.size != 2:
        raise ValueError('Input axes must be a two-element list,tuple, or numpy array') 
    negind = NP.where(axes < 0)[0]    
    if negind.size > 0:
        axes[negind] += inparr.ndim # Convert negative axis numbers to positive
    if axes[0] == axes[1]:
        raise ValueError('The two entries in axes cannot be the same')

    inparr_H = hermitian(inparr, axes=axes)
    invaxes = inparr.ndim-2 + NP.arange(2) # For inverse, they must be the last two axes because of numpy.linalg.inv() requirements
    if not NP.array_equal(NP.sort(axes), invaxes): # the axes are not at the end, so move them for taking inverse        
        inparr_H = NP.moveaxis(inparr_H, NP.sort(axes), invaxes)
    inparr_IH = LA.inv(inparr_H)
    if not NP.array_equal(NP.sort(axes), invaxes): # the axes were moved to the end, so move them back        
        inparr_IH = NP.moveaxis(inparr_H, invaxes, NP.sort(axes))
    
    return inparr_IH

def gen_xcorr_matrix(m11_real, m11_imag, m12_real, m12_imag, m21_real, m21_imag, m22_real, m22_imag):
    # 4 complex numbers for a 2x2 matrix provided as real and imaginary parts, therefore 8 real inputs
    if ((m11_real.shape != m11_imag.shape) or (m11_real.shape != m12_real.shape) or (m11_real.shape != m12_imag.shape) or (m11_real.shape != m21_real.shape) or (m11_real.shape != m21_imag.shape) or (m11_real.shape != m22_real.shape) or (m11_real.shape != m22_imag.shape)):
        raise ValueError('Shapes of input arrays do not match')
    if NP.any(NP.iscomplex(m11_real)):
        raise ValueError('Input m11 real part has imaginary values')
    if NP.any(NP.iscomplex(m11_imag)):
        raise ValueError('Input m11 imaginary part has imaginary values')
    if NP.any(NP.iscomplex(m12_real)):
        raise ValueError('Input m12 real part has imaginary values')
    if NP.any(NP.iscomplex(m12_imag)):
        raise ValueError('Input m12 imaginary part has imaginary values')
    if NP.any(NP.iscomplex(m21_real)):
        raise ValueError('Input m21 real part has imaginary values')
    if NP.any(NP.iscomplex(m21_imag)):
        raise ValueError('Input m21 imaginary part has imaginary values')
    if NP.any(NP.iscomplex(m22_real)):
        raise ValueError('Input m22 real part has imaginary values')
    if NP.any(NP.iscomplex(m22_imag)):
        raise ValueError('Input m22 imaginary part has imaginary values')
    m11_real = m11_real.real
    m12_real = m12_real.real
    m21_real = m21_real.real
    m22_real = m22_real.real
    m11_imag = m11_imag.real
    m12_imag = m12_imag.real
    m21_imag = m21_imag.real
    m22_imag = m22_imag.real    
    
    m11 = m11_real + 1j * m11_imag
    m12 = m12_real + 1j * m12_imag
    m21 = m21_real + 1j * m21_imag
    m22 = m22_real + 1j * m22_imag

    inshape = m11.shape
    outshape = inshape + (2,2)
    outarr = NP.concatenate([m11[...,NP.newaxis], m12[...,NP.newaxis], m21[...,NP.newaxis], m22[...,NP.newaxis]], axis=-1).reshape(outshape) # shape is (...,2,2)

    return outarr

def gen_hermitian_matrix(m11, m12_real, m12_imag, m22, positive=False):
    # 2 real numbers for diagonals of 2x2 matrix, and one complex number for off-diagonal provided as real and imaginary parts, therefore 4 real inputs
    if positive:
        if NP.any(m11.real < 0.0):
            raise ValueError('Negative elements found in m11 for a positive definite matrix')
        if NP.any(m22.real < 0.0):
            raise ValueError('Negative elements found in m22 for a positive definite matrix')

    outarr = gen_xcorr_matrix(m11.real, NP.zeros_like(m11.real), m12_real, m12_imag, m12_real, -m12_imag, m22.real, NP.zeros_like(m22.real))
    return outarr

def perturb_xcvis_array(xcvis, eps=1e-7):
    # eps is the perturbtation
    xcvis_ri = unpack_realimag_to_real(xcvis, arrtype='matrix')
    xcvis_ri_eps = xcvis_ri[...,NP.newaxis,:] + eps*NP.eye(xcvis_ri.shape[-1]).reshape(xcvis_ri.shape[:-3]+(xcvis_ri.shape[-1],xcvis_ri.shape[-1]))
    outarr = repack_real_to_realimag(xcvis_ri_eps, arrtype='matrix')
    return outarr

def perturb_acvis_array(acvis, eps=1e-7):
    acvis_ri = unpack_realimag_to_real(acvis, arrtype='matrix')
    acvis_ri_eps = acvis_ri[...,NP.newaxis,:] + eps*NP.eye(acvis_ri.shape[-1]).reshape(acvis_ri.shape[:-3]+(acvis_ri.shape[-1],acvis_ri.shape[-1]))
    outarr = repack_real_to_realimag(acvis_ri_eps, arrtype='matrix')
    return outarr

def corrupt_visibilities(vis, g_a, g_b, pol_axes=None):
    if not isinstance(vis, NP.ndarray):
        raise TypeError('Input vis must be a numpy array')
    if not isinstance(g_a, NP.ndarray):
        raise TypeError('Input g_a must be a numpy array')
    if not isinstance(g_b, NP.ndarray):
        raise TypeError('Input g_b must be a numpy array')
    if vis.ndim != g_a.ndim:
        raise ValueError('Inputs vis and g_a must have same number of dimensions')
    if vis.ndim != g_b.ndim:
        raise ValueError('Inputs vis and g_b must have same number of dimensions')
    if g_a.ndim != g_b.ndim:
        raise ValueError('Inputs g_a and g_b must have same number of dimensions')
    if vis.shape[-2:] != (2,2):
        raise ValueError('The last two axes of vis must have shape (2,2)')
    if g_a.shape[-2:] != (2,2):
        raise ValueError('The last two axes of g_a must have shape (2,2)')
    if g_b.shape[-2:] != (2,2):
        raise ValueError('The last two axes of g_b must have shape (2,2)')
    return g_a @ vis @ hermitian(g_b, axes=pol_axes)

def generate_triangles(ids, baseid):
    # Generate triads of ids based on a baseid
    if not isinstance(ids, (list,NP.ndarray)):
        raise TypeError('Input ids must be a list or numpy array')
    ids = NP.asarray(ids).reshape(-1)
    if ids.size < 3:
        raise ValueError('Input ids must have at least  elements')
    if not isinstance(baseid, (int,str)):
        raise TypeError('Input baseid must be a scalar integer or string')
    if baseid not in ids:
        raise ValueError('Input baseid not found in inputs ids')
    ids = NP.unique(ids) # select all unique ids
    otherids = [id for id in ids if id!=baseid] # Find other ids except the baseid
    triads = [(baseid,oid1,oid2) for oid1_ind,oid1 in enumerate(otherids) for oid2_ind,oid2 in enumerate(otherids) if oid2_ind > oid1_ind]
    return triads

def corrs_list_on_loops(corrs, ant_pairs, loops, times, bl_axis=-3, pol_axes=[-2,-1]):
    if not isinstance(ant_pairs, (list,NP.ndarray)):
        raise TypeError('Input ant_pairs must be a list or numpy array')
    ant_pairs = NP.asarray(ant_pairs)
    if ant_pairs.ndim == 1:
        if ant_pairs.size != 2:
            raise ValueError('Input ant_pairs contains invalid shape')
    elif ant_pairs.ndim == 2:
        if ant_pairs.shape[-1] != 2:
            raise ValueError('Input ant_pairs contains invalid shape')
    else:
        raise ValueError('Input ant_pairs contains invalid shape')
    ant_pairs = NP.asarray(ant_pairs).reshape(-1,2)
    
    if not isinstance(bl_axis, int):
        raise TypeError('Input bl_axis must be an integer')
        
    if not isinstance(corrs, NP.ndarray):
        raise TypeError('Input corrs must be a numpy array')
    if corrs.shape[bl_axis] != ant_pairs.shape[0]:
        raise ValueError('Input corrs and ant_pairs do not have same number of baselines')
    
    if not isinstance(loops, (list,NP.ndarray)):
        raise TypeError('Input loops must be a list or numpy array')
    loops = NP.asarray(loops)
    if loops.ndim == 1:
        loops = loops.reshape(1,-1)
    elif loops.ndim != 2:
        raise ValueError('Input loops contains invalid shape')

    if not isinstance(pol_axes, (list,tuple,NP.ndarray)):
        raise TypeError('Input pol_axes must be a list, tuple, or numpy array')
    if len(pol_axes) != 2:
        raise ValueError('Input pol_axes must be a two-element sequence')
    else:
        for pax in pol_axes:
            if not isinstance(pax, int):
                raise TypeError('Input pol_axes must be a two-element sequence of integers')
    pol_axes = NP.array(pol_axes).ravel()
    tmpind = NP.where(pol_axes < 0)[0]
    if tmpind.size > 0:
        pol_axes[tmpind] += corrs.ndim # Convert to a positive value for the polarization axes
        
    corrs_lol = []
    for loopi, loop in enumerate(loops):
        for t in NP.unique(times):
            try:
                corrs_loop = []
                for i in range(len(loop)):
                    bl_ind = NP.where((ant_pairs[:,0] == loop[i]) & (ant_pairs[:,1]==loop[(i+1)%loop.size]) & (times == t))[0]
                    if bl_ind.size == 1:
                        corr = NP.copy(NP.take(corrs, bl_ind, axis=bl_axis))
                    elif bl_ind.size == 0: # Check for reversed pair
                        bl_ind = NP.where((ant_pairs[:,0] == loop[(i+1)%loop.size]) & (ant_pairs[:,1]==loop[i]) & (times == t))[0]
                        if bl_ind.size == 0:
                            raise IndexError('Specified antenna pair ({0:0d},{1:0d}) not found in input ant_pairs'.format(loop[i], loop[(i+1)%loop.size]))
                        elif bl_ind.size == 1: # Take Hermitian
                            corr = hermitian(NP.take(corrs, bl_ind, axis=bl_axis), axes=pol_axes)
                        elif bl_ind.size > 1:
                            raise IndexError('{0:0d} indices found for antenna pair ({1:0d},{2:0d}) in input ant_pairs'.format(bl_ind, loop[i], loop[(i+1)%loop.size]))
                    elif bl_ind.size > 1:
                        raise IndexError('{0:0d} indices found for antenna pair ({1:0d},{2:0d}) in input ant_pairs'.format(bl_ind, loop[i], loop[(i+1)%loop.size]))
                
                    corr = NP.take(corr, 0, axis=bl_axis)
                    corrs_loop += [corr]
                corrs_lol += [corrs_loop]
            except:
                pass
        
    return corrs_lol

def advariant(corrs_list, pol_axes=[-2,-1]):
    if not isinstance(corrs_list, list):
        raise TypeError('Input corrs_list must be a list')
    nedges = len(corrs_list)
    if nedges%2 == 0:
        raise ValueError('Input corrs_list must be a list made of odd number of elements for an advariant to be constructed')

    if not isinstance(pol_axes, (list,tuple,NP.ndarray)):
        raise TypeError('Input pol_axes must be a list, tuple, or numpy array')
    if len(pol_axes) != 2:
        raise ValueError('Input pol_axes must be a two-element sequence')
    else:
        for pax in pol_axes:
            if not isinstance(pax, int):
                raise TypeError('Input pol_axes must be a two-element sequence of integers')
    pol_axes = NP.array(pol_axes).ravel()
    
    advar = None
    matgrp = None
    for edgei,corr in enumerate(corrs_list):        
        if not isinstance(corr, NP.ndarray):
            raise TypeError('Element {0:0d} of corrs_list must be a numpy array'.format(edgei))

        if edgei == 0:
            tmpind = NP.where(pol_axes < 0)[0]
            if tmpind.size > 0:
                pol_axes[tmpind] += corr.ndim+1 # Convert to a positive value for the polarization axes

            expected_pol_axes = corr.ndim-2 + NP.arange(2)+1 # For inverse, they have to be the last two axes
            if not NP.array_equal(pol_axes, expected_pol_axes):
                raise ValueError('For advariant calculation, pol_axes must be the last two axes because of inherent assumptions about the axes over which matrix multiplication is performed')

        if corr.ndim == 1:
            corr = corr[...,NP.newaxis,NP.newaxis] # shape=(...,n=1,n=1)
        elif corr.ndim >= 2:            
            shape2 = NP.asarray(corr.shape[-2:])
            if (shape2[0] != shape2[1]):
                raise ValueError('The last two dimensions of each numpy array that forms the items in the input corrs_list must be equal')
            elif (shape2[0] != 1) and (shape2[0] != 2):
                raise ValueError('The last two dimensions of each numpy array that forms the items in the input corrs_list must be (1,1) or (2,2) for GL(1,C) and GL(2,C) matrices, respctively')
            if corr.ndim == 2:
                corr = corr[NP.newaxis,:,:] # shape=(ncorr=1,n=(1/2),n=(1/2))
        if edgei == 0:
            matgrp = NP.copy(shape2[0])
            advar = NP.copy(corr)
        else:
            if not NP.array_equal(shape2[0], matgrp):
                raise ValueError('Shape of list element {0:0d} not indentical to that of list element 0'.format(edgei))
            if edgei%2 == 0:
                advar = advar@corr
            else:
                advar = advar@hat(corr, axes=pol_axes)
                
    return advar

def advariants_multiple_loops(corrs_lol, pol_axes=[-2,-1]):
    if not isinstance(corrs_lol, list):
        raise TypeError('Input corrs_lol must be a list of lists')
    advars_list = []
    for ind, corrs_list in enumerate(corrs_lol):
        advars_list += [advariant(corrs_list, pol_axes=pol_axes)]
    return NP.moveaxis(NP.array(advars_list), 0, -3)

def vector_from_advariant(advars):
    if not isinstance(advars, NP.ndarray):
        raise TypeError('Input advars must be a numpy array')
    shape2 = advars.shape[-2:]
    matgrp = NP.copy(shape2[0])
    
    if (advars.shape[-1] != 2) and (advars.shape[-1] != 1):
        raise ValueError('Input advariant shape incompatible with GL(1,C) or GL(2,C) matrices')
    elif advars.shape[-1] == 2:
        if advars.shape[-1] != advars.shape[-2]:
            raise ValueError('Advariants must be 2x2 matrices in the last two dimensions')
        # Determine Pauli matrices
        pauli_sig_0 = NP.identity(matgrp, dtype=complex).reshape(tuple(NP.ones(advars.ndim-2, dtype=int))+tuple(shape2)) # Psigma_0: shape=(...,n=(1/2),n=(1/2))
        pauli_sig_1 = NP.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex).reshape(tuple(NP.ones(advars.ndim-2, dtype=int))+tuple(shape2)) # Psigma_1: shape=(...,n=(1/2),n=(1/2))
        pauli_sig_2 = NP.asarray([[0.0, -1j], [1j, 0.0]], dtype=complex).reshape(tuple(NP.ones(advars.ndim-2, dtype=int))+tuple(shape2)) # Psigma_2: shape=(...,n=(1/2),n=(1/2))
        pauli_sig_3 = NP.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex).reshape(tuple(NP.ones(advars.ndim-2, dtype=int))+tuple(shape2)) # Psigma_: shape=(...,n=(1/2),n=(1/2))

        # Determine components of 4-vectors
        z0 = 0.5 * NP.trace(advars@pauli_sig_0, axis1=-2, axis2=-1)
        z1 = 0.5 * NP.trace(advars@pauli_sig_1, axis1=-2, axis2=-1)
        z2 = 0.5 * NP.trace(advars@pauli_sig_2, axis1=-2, axis2=-1)
        z3 = 0.5 * NP.trace(advars@pauli_sig_3, axis1=-2, axis2=-1)

        z_4vect = NP.concatenate([z0[...,NP.newaxis], z1[...,NP.newaxis], z2[...,NP.newaxis], z3[...,NP.newaxis]], axis=-1) # shape=(...,4)
        return z_4vect
    else:
        if advars.shape[-1] != advars.shape[-2]:
            raise ValueError('Advariants must be 1x1 matrices in the last two dimensions')
        return advars # GL(1,C) already

def minkowski_dot(z1_4v, z2_4v=None):
    if not isinstance(z1_4v, NP.ndarray):
        raise TypeError('Input z1_4v must be a numpy array')
    shape1 = NP.array(z1_4v.shape)
    if shape1[-1] != 4:
        raise ValueError('The dimension of last axis in input z1_4v must equal 4 to be a valid 4-vector')
    
    metric = NP.array([[1,0,0,0], 
                       [0,-1,0,0], 
                       [0,0,-1,0], 
                       [0,0,0,-1]], dtype=float).reshape(-1,4) # Minkowski metric
    
    x1_4v = z1_4v.real # shape=(...,M,4)
    y1_4v = z1_4v.imag # shape=(...,M,4)
    stack1_4v = NP.concatenate([x1_4v, y1_4v], axis=-2) # shape=(...,2M,4)
    if z2_4v is None:
        stack2_4v = NP.copy(stack1_4v) # shape=(...,2M,4)
    else:
        if not isinstance(z2_4v, NP.ndarray):
            raise TypeError('Input z2_4v must be a numpy array')
        shape2 = NP.array(z2_4v.shape)
        if not NP.array_equal(shape1[-1], shape2[-1]):
            raise ValueError('The dimension of last axis in inputs z1_4v and z2_4v must match')
        x2_4v = z2_4v.real # shape=(...,N,4)
        y2_4v = z2_4v.imag # shape=(...,N,4)
        stack2_4v = NP.concatenate([x2_4v, y2_4v], axis=-2) # shape=(...,2N,4)
    
    mdp = NP.einsum('...ij,jk,...lk->...il', stack1_4v, metric, stack2_4v) # shape=(...2M,2N)
    if z2_4v is None: # Return only the upper diagonal
        upperind = NP.triu_indices(mdp.shape[-1])
        upperind_raveled = NP.ravel_multi_index(upperind, mdp.shape[-2:])
        mdp = mdp.reshape(mdp.shape[:-2]+(-1,))
        mdp = mdp[...,upperind_raveled]
    else:
        mdp = mdp.reshape(mdp.shape[:-2]+(-1,))       
    
    return mdp

def independent_minkowski_dots(z4v):
    if not isinstance(z4v, NP.ndarray):
        raise TypeError('Input z1_4v must be a numpy array')
    if z4v.ndim == 1:
        z4v = z4v.reshape(1,-1)
    inshape = NP.array(z4v.shape)
    z4v_basis = z4v[...,:2,:] # Choose first two complex 4-vectors for the basis
    z4v_rest = z4v[...,2:,:] # Choose first two complex 4-vectors for the rest
    
    mdp_basis_basis = minkowski_dot(z4v_basis)
    mdp_basis_rest = minkowski_dot(z4v_basis, z4v_rest)
    
    mdp = NP.concatenate([mdp_basis_basis, mdp_basis_rest], axis=-1)
    
    return mdp

def compute_cloinv(obs, baseid='AA', elements_subset=['AA', 'LM', 'AZ', 'AP', 'SM', 'SP', 'PV']):
    ehtarray_tarr = obs.tarr
    data = obs.data
    times = data["time"]#np.unique(data['time'])
    
    ant_pairs = list(zip(data['t1'], data['t2']))
    #print(ant_pairs)
    
    corr_2x2_vis_sampled = NP.zeros((len(data), 2, 2), dtype='complex')
    
    corr_2x2_vis_sampled[:,0,0] = obs.unpack(['rrvis'])['rrvis']
    corr_2x2_vis_sampled[:,0,1] = obs.unpack(['rlvis'])['rlvis']
    corr_2x2_vis_sampled[:,1,0] = obs.unpack(['lrvis'])['lrvis']
    corr_2x2_vis_sampled[:,1,1] = obs.unpack(['llvis'])['llvis']
    
    triads_indep = generate_triangles(elements_subset, baseid=baseid)
    #print(triads_indep)
    
    corrs_list_triangles = corrs_list_on_loops(corr_2x2_vis_sampled, ant_pairs=ant_pairs, loops=triads_indep, times=times, bl_axis=-3, pol_axes=(-2,-1))
    
    advars = advariants_multiple_loops(corrs_list_triangles, pol_axes=[-2,-1])
    
    z4v = vector_from_advariant(advars)
    # z4v_real = unpack_realimag_to_real(z4v, arrtype='scalar')
    z4v_real = NP.concatenate([z4v.real[...,NP.newaxis,:], z4v.real[...,NP.newaxis,:]], axis=-2)
    #print(z4v_real.shape)
    z4v_real = z4v_real.reshape(z4v_real.shape[:-3]+(-1,4))
    #print(z4v_real.shape)
    #print(z4v)
    
    mdp_indep = independent_minkowski_dots(z4v)
    cloinv = remove_scaling_in_minkoski_dots(mdp_indep)
    return cloinv

class cloinv_norm():
    def __init__(self, obs, img, baseid='AA', elements_subset=['AA', 'LM', 'AZ', 'AP', 'SM', 'SP', 'PV'], rescaling=1.):
        self.obs = obs
        self.img = img
        self.size = len(self.img.imvec)
        self.baseid = baseid
        self.elements_subset = elements_subset
        self.rescaling = rescaling
        self.cloinv = compute_cloinv(self.obs, baseid=self.baseid, elements_subset=self.elements_subset)
        
    def __call__(self, arr):
        self.img.qvec = qimage(self.img.imvec, self.rescaling*arr[0:self.size], 2*NP.pi*arr[self.size:2*self.size])
        self.img.uvec = uimage(self.img.imvec, self.rescaling*arr[0:self.size], 2*NP.pi*arr[self.size:2*self.size])
        obs_test = self.img.observe_same(self.obs, add_th_noise=False, phasecal=True, ampcal=True, ttype='direct')
        cloinv = compute_cloinv(obs_test, baseid=self.baseid, elements_subset=self.elements_subset)
        return NP.linalg.norm(self.cloinv-cloinv)
















