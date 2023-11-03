import numpy as np
import scipy.stats as ss

from regpy.operators import Operator, Multiplication, DirectSum
from regpy.discrs import Discretization
from regpy.functionals import HilbertNorm
from regpy.hilbert import L2

#from ehtim.imaging.imager_utils import chisqdata_camp
import ehtim.observing.obs_helpers as obsh
import ehtim.const_def as ehc


def construct_cl_trace(rrvis1, rlvis1, lrvis1, llvis1, rrvis2, rlvis2, lrvis2, llvis2, rrvis3, rlvis3, lrvis3, llvis3, rrvis4, rlvis4, lrvis4, llvis4):
    #non conjugate version       
    vismatrix1 = np.asarray([[rrvis1, rlvis1],[lrvis1, llvis1]])
    vismatrix2 = np.asarray([[rrvis4, rlvis4],[lrvis4, llvis4]]).conjugate()
    vismatrix3 = np.asarray([[rrvis2, rlvis2],[lrvis2, llvis2]])
    vismatrix4 = np.asarray([[rrvis3, rlvis3],[lrvis3, llvis3]])
    
    final_matrix1 = np.transpose(vismatrix1, axes=[2,0,1])
    final_matrix2 = np.linalg.inv(np.transpose(vismatrix2, axes=[2,0,1]))
    final_matrix3 = np.transpose(vismatrix3, axes=[2,0,1])
    final_matrix4 = np.linalg.inv(np.transpose(vismatrix4, axes=[2,0,1]))
    
    final_matrix = final_matrix1 @ final_matrix2 @ final_matrix3 @ final_matrix4
            
    cltrace1 = 0.5*np.trace(final_matrix, axis1=1, axis2=2)
    ###########################################################################
    #non conjugate version       
    vismatrix1 = np.asarray([[rrvis1, rlvis1],[lrvis1, llvis1]])
    vismatrix2 = np.asarray([[rrvis4, rlvis4],[lrvis4, llvis4]]).conjugate()
    vismatrix3 = np.asarray([[rrvis3, rlvis3],[lrvis3, llvis3]])
    vismatrix4 = np.asarray([[rrvis2, rlvis2],[lrvis2, llvis2]])
    
    final_matrix1 = np.transpose(vismatrix1, axes=[2,0,1])
    final_matrix2 = np.linalg.inv(np.transpose(vismatrix2, axes=[2,0,1]))
    final_matrix3 = np.transpose(vismatrix3, axes=[2,0,1])
    final_matrix4 = np.linalg.inv(np.transpose(vismatrix4, axes=[2,0,1]))
    
    final_matrix = final_matrix1 @ final_matrix2 @ final_matrix3 @ final_matrix4
            
    cltrace2 = 0.5*np.trace(final_matrix, axis1=1, axis2=2)
    ###########################################################################
    #non conjugate version       
    vismatrix1 = np.asarray([[rrvis1, rlvis1],[lrvis1, llvis1]])
    vismatrix2 = np.asarray([[rrvis2, rlvis2],[lrvis2, llvis2]])
    vismatrix3 = np.asarray([[rrvis4, rlvis4],[lrvis4, llvis4]]).conjugate()
    vismatrix4 = np.asarray([[rrvis3, rlvis3],[lrvis3, llvis3]])
    
    final_matrix1 = np.transpose(vismatrix1, axes=[2,0,1])
    final_matrix2 = np.linalg.inv(np.transpose(vismatrix2, axes=[2,0,1]))
    final_matrix3 = np.transpose(vismatrix3, axes=[2,0,1])
    final_matrix4 = np.linalg.inv(np.transpose(vismatrix4, axes=[2,0,1]))
    
    final_matrix = final_matrix1 @ final_matrix2 @ final_matrix3 @ final_matrix4
            
    cltrace3 = 0.5*np.trace(final_matrix, axis1=1, axis2=2)
    ###########################################################################
    #non conjugate version       
    vismatrix1 = np.asarray([[rrvis1, rlvis1],[lrvis1, llvis1]])
    vismatrix2 = np.asarray([[rrvis2, rlvis2],[lrvis2, llvis2]])
    vismatrix3 = np.asarray([[rrvis3, rlvis3],[lrvis3, llvis3]])
    vismatrix4 = np.asarray([[rrvis4, rlvis4],[lrvis4, llvis4]]).conjugate()
    
    final_matrix1 = np.transpose(vismatrix1, axes=[2,0,1])
    final_matrix2 = np.linalg.inv(np.transpose(vismatrix2, axes=[2,0,1]))
    final_matrix3 = np.transpose(vismatrix3, axes=[2,0,1])
    final_matrix4 = np.linalg.inv(np.transpose(vismatrix4, axes=[2,0,1]))
    
    final_matrix = final_matrix1 @ final_matrix2 @ final_matrix3 @ final_matrix4
            
    cltrace4 = 0.5*np.trace(final_matrix, axis1=1, axis2=2)
    ###########################################################################
    #non conjugate version       
    vismatrix1 = np.asarray([[rrvis1, rlvis1],[lrvis1, llvis1]])
    vismatrix2 = np.asarray([[rrvis3, rlvis3],[lrvis3, llvis3]])
    vismatrix3 = np.asarray([[rrvis4, rlvis4],[lrvis4, llvis4]]).conjugate()
    vismatrix4 = np.asarray([[rrvis2, rlvis2],[lrvis2, llvis2]])
    
    final_matrix1 = np.transpose(vismatrix1, axes=[2,0,1])
    final_matrix2 = np.linalg.inv(np.transpose(vismatrix2, axes=[2,0,1]))
    final_matrix3 = np.transpose(vismatrix3, axes=[2,0,1])
    final_matrix4 = np.linalg.inv(np.transpose(vismatrix4, axes=[2,0,1]))
    
    final_matrix = final_matrix1 @ final_matrix2 @ final_matrix3 @ final_matrix4
            
    cltrace5 = 0.5*np.trace(final_matrix, axis1=1, axis2=2)
    ###########################################################################
    #non conjugate version       
    vismatrix1 = np.asarray([[rrvis1, rlvis1],[lrvis1, llvis1]])
    vismatrix2 = np.asarray([[rrvis3, rlvis3],[lrvis3, llvis3]])
    vismatrix3 = np.asarray([[rrvis2, rlvis2],[lrvis2, llvis2]])
    vismatrix4 = np.asarray([[rrvis4, rlvis4],[lrvis4, llvis4]]).conjugate()
    
    final_matrix1 = np.transpose(vismatrix1, axes=[2,0,1])
    final_matrix2 = np.linalg.inv(np.transpose(vismatrix2, axes=[2,0,1]))
    final_matrix3 = np.transpose(vismatrix3, axes=[2,0,1])
    final_matrix4 = np.linalg.inv(np.transpose(vismatrix4, axes=[2,0,1]))
    
    final_matrix = final_matrix1 @ final_matrix2 @ final_matrix3 @ final_matrix4
            
    cltrace6 = 0.5*np.trace(final_matrix, axis1=1, axis2=2)
    
    return (cltrace1, cltrace2, cltrace3, cltrace4, cltrace5, cltrace6)

def make_closure_trace(blue1, blue2, red1, red2, polrep='stokes', debias=False):
    if polrep == 'stokes':
        rrsig1 = np.sqrt(blue1['sigma']**2 + blue1['vsigma']**2)
        rrsig2 = np.sqrt(blue2['sigma']**2 + blue2['vsigma']**2)
        rrsig3 = np.sqrt(red1['sigma']**2 + red1['vsigma']**2)
        rrsig4 = np.sqrt(red2['sigma']**2 + red2['vsigma']**2)

        rrp1 = blue1['vis'] + blue1['vvis']
        rrp2 = blue2['vis'] + blue2['vvis']
        rrp3 = red1['vis'] + red1['vvis']
        rrp4 = red2['vis'] + red2['vvis']

        llsig1 = np.sqrt(blue1['sigma']**2 + blue1['vsigma']**2)
        llsig2 = np.sqrt(blue2['sigma']**2 + blue2['vsigma']**2)
        llsig3 = np.sqrt(red1['sigma']**2 + red1['vsigma']**2)
        llsig4 = np.sqrt(red2['sigma']**2 + red2['vsigma']**2)

        llp1 = blue1['vis'] - blue1['vvis']
        llp2 = blue2['vis'] - blue2['vvis']
        llp3 = red1['vis'] - red1['vvis']
        llp4 = red2['vis'] - red2['vvis']

        rlsig1 = np.sqrt(blue1['qsigma']**2 + blue1['usigma']**2)
        rlsig2 = np.sqrt(blue2['qsigma']**2 + blue2['usigma']**2)
        rlsig3 = np.sqrt(red1['qsigma']**2 + red1['usigma']**2)
        rlsig4 = np.sqrt(red2['qsigma']**2 + red2['usigma']**2)

        rlp1 = blue1['qvis'] - 1j*blue1['uvis']
        rlp2 = blue2['qvis'] - 1j*blue2['uvis']
        rlp3 = red1['qvis'] - 1j*red1['uvis']
        rlp4 = red2['qvis'] - 1j*red2['uvis']

        lrsig1 = np.sqrt(blue1['qsigma']**2 + blue1['usigma']**2)
        lrsig2 = np.sqrt(blue2['qsigma']**2 + blue2['usigma']**2)
        lrsig3 = np.sqrt(red1['qsigma']**2 + red1['usigma']**2)
        lrsig4 = np.sqrt(red2['qsigma']**2 + red2['usigma']**2)

        lrp1 = blue1['qvis'] + 1j*blue1['uvis']
        lrp2 = blue2['qvis'] + 1j*blue2['uvis']
        lrp3 = red1['qvis'] + 1j*red1['uvis']
        lrp4 = red2['qvis'] + 1j*red2['uvis']

    elif polrep == 'circ':
        vtype = 'rrvis'
        sigmatype = 'rrsigma'
        rrsig1 = blue1[sigmatype]
        rrsig2 = blue2[sigmatype]
        rrsig3 = red1[sigmatype]
        rrsig4 = red2[sigmatype]

        rrp1 = blue1[vtype]
        rrp2 = blue2[vtype]
        rrp3 = red1[vtype]
        rrp4 = red2[vtype]
        
        vtype = 'llvis'
        sigmatype = 'llsigma'
        llsig1 = blue1[sigmatype]
        llsig2 = blue2[sigmatype]
        llsig3 = red1[sigmatype]
        llsig4 = red2[sigmatype]

        llp1 = blue1[vtype]
        llp2 = blue2[vtype]
        llp3 = red1[vtype]
        llp4 = red2[vtype]
        
        vtype = 'rlvis'
        sigmatype = 'rlsigma'
        rlsig1 = blue1[sigmatype]
        rlsig2 = blue2[sigmatype]
        rlsig3 = red1[sigmatype]
        rlsig4 = red2[sigmatype]

        rlp1 = blue1[vtype]
        rlp2 = blue2[vtype]
        rlp3 = red1[vtype]
        rlp4 = red2[vtype]
        
        vtype = 'lrvis'
        sigmatype = 'lrsigma'
        lrsig1 = blue1[sigmatype]
        lrsig2 = blue2[sigmatype]
        lrsig3 = red1[sigmatype]
        lrsig4 = red2[sigmatype]

        lrp1 = blue1[vtype]
        lrp2 = blue2[vtype]
        lrp3 = red1[vtype]
        lrp4 = red2[vtype]

    return [[rrp1, rlp1, lrp1, llp1],
            [rrp2, rlp2, lrp2, llp2],
            [rrp3, rlp3, lrp3, llp3],
            [rrp4, rlp4, lrp4, llp4]]

def find_quad_array(obs, timetype=False):
    #READ OUT ALL THE VISIBILITIES
    if timetype is False:
        timetype = obs.timetype

    # Get data sorted by time
    tlist = obs.tlist(conj=True)
    out = []
    cas = []
    tt = 1
    for tdata in tlist:

        # sys.stdout.write('\rGetting closure amps:: type %s %s , count %s, scan %i/%i' %
        #                 (vtype, ctype, count, tt, len(tlist)))
        # sys.stdout.flush()
        tt += 1

        time = tdata[0]['time']
        if timetype in ['GMST', 'gmst'] and obs.timetype == 'UTC':
            time = obsh.utc_to_gmst(time, obs.mjd)
        if timetype in ['UTC', 'utc'] and obs.timetype == 'GMST':
            time = obsh.gmst_to_utc(time, obs.mjd)

        sites = np.array(list(set(np.hstack((tdata['t1'], tdata['t2'])))))
        if len(sites) < 4:
            continue

        # Create a dictionary of baseline data at the current time including conjugates;
        l_dict = {}
        for dat in tdata:
            l_dict[(dat['t1'], dat['t2'])] = dat

        # Minimal set
        quadsets = obsh.quad_minimal_set(sites, obs.tarr, obs.tkey)


        # Loop over all closure amplitudes
        for quad in quadsets:
            # Blue is numerator, red is denominator
            if (quad[0], quad[1]) not in l_dict.keys():
                continue
            if (quad[2], quad[3]) not in l_dict.keys():
                continue
            if (quad[1], quad[2]) not in l_dict.keys():
                continue
            if (quad[0], quad[3]) not in l_dict.keys():
                continue

            try:
                blue1 = l_dict[quad[0], quad[1]]
                blue2 = l_dict[quad[2], quad[3]]
                red1 = l_dict[quad[0], quad[3]]
                red2 = l_dict[quad[1], quad[2]]
            except KeyError:
                continue

            # Compute the closure trace
            visibilities = np.asarray(make_closure_trace(blue1, blue2, red1, red2, polrep=obs.polrep)).flatten()

            # Add the closure amplitudes to the equal-time list
            # Our site convention is (12)(34)/(14)(23)
            dtype=np.dtype('float,str,str,str,str,float,float,float,float,float,float,float,float,complex,complex,complex,complex,complex,complex,complex,complex,complex,complex,complex,complex,complex,complex,complex,complex')
            cas.append(np.array((time,
                                 quad[0], quad[1], quad[2], quad[3],
                                 blue1['u'], blue1['v'], blue2['u'], blue2['v'],
                                 red1['u'], red1['v'], red2['u'], red2['v'],
                                 visibilities[0], visibilities[1], visibilities[2], visibilities[3],
                                 visibilities[4], visibilities[5], visibilities[6], visibilities[7],
                                 visibilities[8], visibilities[9], visibilities[10], visibilities[11],
                                 visibilities[12], visibilities[13], visibilities[14], visibilities[15]),dtype=dtype))
                                #dtype=ehc.DTCAMP))

    clamparr = np.array(cas)
    return clamparr

def build_cltrace_operator(wrapper, timetype=False, errors=True):
    obs = wrapper.Obsdata.copy()
    #HERE we define the closure traces with the recovered img as imvec (since im and img are not exactly centered the same)
    #now the imvec in true_trace and recovered trace match again
    #this is done on real data, i.e. replace the visibilities with the recovered visibilities
    #pass on recovered image by wrapper.Prior
    obs_reco = wrapper.Prior.observe_same(obs, add_th_noise=False, phasecal=True, ampcal=True, ttype=wrapper.ttype)
    obs.data['vis'] = obs_reco.data['vis']
    obs.data['vvis'] = obs_reco.data['vvis']
    
    clamparr = find_quad_array(obs, timetype=timetype)
 
    #NOW CONSTRUCT THE FORWARD MATRICES
    #clamparr = wrapper.Obsdata.c_amplitudes(mode='all', count='min', ctype='camp', debias=False)
    
    uv1 = np.hstack((clamparr['f5'].reshape(-1, 1), clamparr['f6'].reshape(-1, 1)))
    uv2 = np.hstack((clamparr['f7'].reshape(-1, 1), clamparr['f8'].reshape(-1, 1)))
    uv3 = np.hstack((clamparr['f9'].reshape(-1, 1), clamparr['f10'].reshape(-1, 1)))
    uv4 = np.hstack((clamparr['f11'].reshape(-1, 1), clamparr['f12'].reshape(-1, 1)))

    # make fourier matrices
    A4 = (obsh.ftmatrix(wrapper.Prior.psize, wrapper.Prior.xdim, wrapper.Prior.ydim, uv1, pulse=wrapper.Prior.pulse, mask=wrapper.embed_mask),
          obsh.ftmatrix(wrapper.Prior.psize, wrapper.Prior.xdim, wrapper.Prior.ydim, uv2, pulse=wrapper.Prior.pulse, mask=wrapper.embed_mask),
          obsh.ftmatrix(wrapper.Prior.psize, wrapper.Prior.xdim, wrapper.Prior.ydim, uv3, pulse=wrapper.Prior.pulse, mask=wrapper.embed_mask),
          obsh.ftmatrix(wrapper.Prior.psize, wrapper.Prior.xdim, wrapper.Prior.ydim, uv4, pulse=wrapper.Prior.pulse, mask=wrapper.embed_mask)
          )
    
    time = clamparr['f0']
    
#EXTRACT VISIBILITIES
    rrvis1 = clamparr['f13']
    rlvis1 = clamparr['f14']
    lrvis1 = clamparr['f15']
    llvis1 = clamparr['f16']
    
    rrvis2 = clamparr['f17']
    rlvis2 = clamparr['f18']
    lrvis2 = clamparr['f19']
    llvis2 = clamparr['f20']
    
    rrvis3 = clamparr['f21']
    rlvis3 = clamparr['f22']
    lrvis3 = clamparr['f23']
    llvis3 = clamparr['f24']
    
    rrvis4 = clamparr['f25']
    rlvis4 = clamparr['f26']
    lrvis4 = clamparr['f27']
    llvis4 = clamparr['f28']

    #CONSTRUCT CLOSURE TRACES
    cltraces = construct_cl_trace(rrvis1, rlvis1, lrvis1, llvis1, rrvis2, rlvis2, lrvis2, llvis2, rrvis3, rlvis3, lrvis3, llvis3, rrvis4, rlvis4, lrvis4, llvis4)
    
    #Construct forward operators
    orders = ['ABCD', 'ABDC', 'ACBD', 'ACDB', 'ADBC', 'ADCB']
    ops = []
    for i in range(len(orders)):
        ops.append(ClosureTracePol(A4, order=orders[i])-cltraces[i])
        
    final_op = DirectSum(*ops)
    
    cop = Makecopies(ops[0].domain, 6)
 
    if errors:
        cltraces = []
        seeds = np.arange(1, 11)
        l = len(wrapper.Obsdata.data)
        for i in range(10):
            obs_noise = wrapper.Obsdata.copy().switch_polrep(polrep_out='circ')
            np.random.seed = seeds[i]
            obs_noise.data['rrvis'] += obs_noise.data['rrsigma']*(np.random.normal(0.0, 1.0, l)+((1j))*np.random.normal(0.0, 1.0, l))
            obs_noise.data['rlvis'] += obs_noise.data['rlsigma']*(np.random.normal(0.0, 1.0, l)+((1j))*np.random.normal(0.0, 1.0, l))
            obs_noise.data['lrvis'] += obs_noise.data['lrsigma']*(np.random.normal(0.0, 1.0, l)+((1j))*np.random.normal(0.0, 1.0, l))
            obs_noise.data['llvis'] += obs_noise.data['llsigma']*(np.random.normal(0.0, 1.0, l)+((1j))*np.random.normal(0.0, 1.0, l))
            clamparr = find_quad_array(obs_noise, timetype=timetype)
            
            rrvis1n = clamparr['f13']
            rlvis1n = clamparr['f14']
            lrvis1n = clamparr['f15']
            llvis1n = clamparr['f16']
            
            rrvis2n = clamparr['f17']
            rlvis2n = clamparr['f18']
            lrvis2n = clamparr['f19']
            llvis2n = clamparr['f20']
            
            rrvis3n = clamparr['f21']
            rlvis3n = clamparr['f22']
            lrvis3n = clamparr['f23']
            llvis3n = clamparr['f24']
            
            rrvis4n = clamparr['f25']
            rlvis4n = clamparr['f26']
            lrvis4n = clamparr['f27']
            llvis4n = clamparr['f28']
            
            traces = construct_cl_trace(rrvis1n, rlvis1n, lrvis1n, llvis1n, rrvis2n, rlvis2n, lrvis2n, llvis2n, rrvis3n, rlvis3n, lrvis3n, llvis3n, rrvis4n, rlvis4n, lrvis4n, llvis4n)
            cltraces.append(traces)
        
        cltraces = np.asarray(cltraces)
        #cctp = ss.circstd(np.angle(cltraces), high=np.pi, low=-np.pi, axis=0)
        
        cctp = np.std(cltraces, axis=0)
        
        weightings = []
        for i in range(len(orders)):
            weightings.append(Multiplication(ops[i].codomain, 1/cctp[i]))
        
        mult = DirectSum(*weightings)
        func = HilbertNorm(L2(final_op.codomain)) * mult * final_op * cop
        return func
    
    else:
        func = HilbertNorm(L2(final_op.codomain)) * final_op * cop
        return func
    
class CompMultiplication(Operator):
    def __init__(self, op1, op2):
        assert op1.domain == op2.domain
        assert op1.codomain == op2.codomain
        self.op1 = op1
        self.op2 = op2
        super().__init__(op1.domain, op1.codomain)
        
    def _eval(self, x, differentiate=False):
        self.y1 = self.op1(x)
        self.y2 = self.op2(x)
        return self.op1(x)*self.op2(x)
    
    def _derivative(self, h):
        return self.y1*self.op2._derivative(h)+self.op1._derivative(h) * self.y2
    
    def _adjoint(self, v):
        return self.op2._adjoint(self.y1.conjugate() * v) + self.op2._adjoint(self.y1.conjugate() * v)
    
class Makecopies(Operator):
    def __init__(self, domain, size):
        self.size = size
        codomain = domain**size
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        return np.concatenate([x.flatten()]*self.size)
    
    def _adjoint(self, y):
        toret = []
        for i in range(self.size):
            toret.append(self.codomain.split(y)[i])
        return np.mean(np.asarray(toret), axis=0)

class ClosureTracePol(Operator):
    def __init__(self, A4, order='ABCD'):
        self.order = order
        #self.conj = conj
        #self.wrapper = wrapper
        #domain = Discretization(self.wrapper.xtuple.shape)
        domain = Discretization((3, A4[0].shape[1]))
        
        #Note from ehtim.obsdata line 3307, that the convention here is: 12, 34, 14, 23
        #We need: 12, 32, 34, 14 here
        #Or 
        #_, _, A4 = chisqdata_camp(self.wrapper.Obsdata, self.wrapper.Prior, self.wrapper.embed_mask)
        self.A4 = []
        if self.order == 'ABCD':
            self.A4.append(A4[0])
            self.A4.append(A4[3])
            self.A4.append(A4[1])
            self.A4.append(A4[2])
        if self.order == 'ABDC':
            self.A4.append(A4[0])
            self.A4.append(A4[3])
            self.A4.append(A4[2])
            self.A4.append(A4[1])
        if self.order == 'ACBD':
            self.A4.append(A4[0])
            self.A4.append(A4[1])
            self.A4.append(A4[3])
            self.A4.append(A4[2])
        if self.order == 'ACDB':
            self.A4.append(A4[0])
            self.A4.append(A4[1])
            self.A4.append(A4[2])
            self.A4.append(A4[3])
        if self.order == 'ADBC':
            self.A4.append(A4[0])
            self.A4.append(A4[2])
            self.A4.append(A4[3])
            self.A4.append(A4[1])            
        if self.order == 'ADCB':
            self.A4.append(A4[0])
            self.A4.append(A4[2])
            self.A4.append(A4[1])
            self.A4.append(A4[3])
        codomain = Discretization(self.A4[0].shape[0], dtype=complex)

        super().__init__(domain, codomain)

    def _eval(self, x, differentiate=False):

        iimage = x[0]
        qimage = x[1]
        uimage = x[2]
        
        brightness_matrix = np.asarray([[iimage, qimage-1j*uimage],[qimage+1j*uimage, iimage]])
        
        vismatrix1 = np.asarray([[ self.A4[0] @ brightness_matrix[0,0], self.A4[0] @ brightness_matrix[0,1]], [self.A4[0] @ brightness_matrix[1,0], self.A4[0] @ brightness_matrix[1,1]]])
        vismatrix2 = np.asarray([[ self.A4[1] @ brightness_matrix[0,0], self.A4[1] @ brightness_matrix[0,1]], [self.A4[1] @ brightness_matrix[1,0], self.A4[1] @ brightness_matrix[1,1]]])
        vismatrix3 = np.asarray([[ self.A4[2] @ brightness_matrix[0,0], self.A4[2] @ brightness_matrix[0,1]], [self.A4[2] @ brightness_matrix[1,0], self.A4[2] @ brightness_matrix[1,1]]])
        vismatrix4 = np.asarray([[ self.A4[3] @ brightness_matrix[0,0], self.A4[3] @ brightness_matrix[0,1]], [self.A4[3] @ brightness_matrix[1,0], self.A4[3] @ brightness_matrix[1,1]]])
        
        if self.order == 'ABCD':
            vismatrix2 = vismatrix2.conjugate()
        if self.order == 'ABDC':
            vismatrix2 = vismatrix2.conjugate()
        if self.order == 'ACBD':
            vismatrix3 = vismatrix3.conjugate()
        if self.order == 'ACDB':
            vismatrix4 = vismatrix4.conjugate()
        if self.order == 'ADBC':
            vismatrix3 = vismatrix3.conjugate()
        if self.order == 'ADCB':
            vismatrix4 = vismatrix4.conjugate()
        
        self.final_matrix1 = np.transpose(vismatrix1, axes=[2,0,1])
        self.final_matrix2 = np.linalg.inv(np.transpose(vismatrix2, axes=[2,0,1]))
        self.final_matrix3 = np.transpose(vismatrix3, axes=[2,0,1])
        self.final_matrix4 = np.linalg.inv(np.transpose(vismatrix4, axes=[2,0,1]))
        
        final_matrix = self.final_matrix1 @ self.final_matrix2 @ self.final_matrix3 @ self.final_matrix4
                
        return 0.5*np.trace(final_matrix, axis1=1, axis2=2)

    def _derivative(self, h):
        iimage = h[0]
        qimage = h[1]
        uimage = h[2]
        
        brightness_matrix = np.asarray([[iimage, qimage-1j*uimage],[qimage+1j*uimage, iimage]])
        
        vismatrix1 = np.asarray([[ self.A4[0] @ brightness_matrix[0,0], self.A4[0] @ brightness_matrix[0,1]], [self.A4[0] @ brightness_matrix[1,0], self.A4[0] @ brightness_matrix[1,1]]])
        vismatrix2 = np.asarray([[ self.A4[1] @ brightness_matrix[0,0], self.A4[1] @ brightness_matrix[0,1]], [self.A4[1] @ brightness_matrix[1,0], self.A4[1] @ brightness_matrix[1,1]]])
        vismatrix3 = np.asarray([[ self.A4[2] @ brightness_matrix[0,0], self.A4[2] @ brightness_matrix[0,1]], [self.A4[2] @ brightness_matrix[1,0], self.A4[2] @ brightness_matrix[1,1]]])
        vismatrix4 = np.asarray([[ self.A4[3] @ brightness_matrix[0,0], self.A4[3] @ brightness_matrix[0,1]], [self.A4[3] @ brightness_matrix[1,0], self.A4[3] @ brightness_matrix[1,1]]])
    
        if self.order == 'ABCD':
            vismatrix2 = vismatrix2.conjugate()
        if self.order == 'ABDC':
            vismatrix2 = vismatrix2.conjugate()
        if self.order == 'ACBD':
            vismatrix3 = vismatrix3.conjugate()
        if self.order == 'ACDB':
            vismatrix4 = vismatrix4.conjugate()
        if self.order == 'ADBC':
            vismatrix3 = vismatrix3.conjugate()
        if self.order == 'ADCB':
            vismatrix4 = vismatrix4.conjugate()
    
        final_matrix1h = np.transpose(vismatrix1, axes=[2,0,1])
        final_matrix2h = np.transpose(vismatrix2, axes=[2,0,1])
        final_matrix3h = np.transpose(vismatrix3, axes=[2,0,1])
        final_matrix4h = np.transpose(vismatrix4, axes=[2,0,1])
    
        #derivative of non-linear part    
        final_matrix = -self.final_matrix1 @ self.final_matrix2 @ self.final_matrix3 @ self.final_matrix4 @ final_matrix4h @ self.final_matrix4 \
            + self.final_matrix1 @ self.final_matrix2 @ final_matrix3h @ self.final_matrix4 \
            - self.final_matrix1 @ self.final_matrix2 @ final_matrix2h @ self.final_matrix2 @ self.final_matrix3 @ self.final_matrix4 \
            + final_matrix1h @ self.final_matrix2 @ self.final_matrix3 @ self.final_matrix4
              
        return 0.5*np.trace(final_matrix, axis1=1, axis2=2)
    
    def _adjoint(self, h):
        final_matrix = np.zeros((len(h),2,2), dtype=complex)
        final_matrix[:,0,0] = h
        final_matrix[:,1,1] = h
        
        final_matrix1 = np.conjugate(self.final_matrix1)
        final_matrix2 = np.conjugate(self.final_matrix2)
        final_matrix3 = np.conjugate(self.final_matrix3)
        final_matrix4 = np.conjugate(self.final_matrix4)
        
        vismatrix1 = final_matrix @ np.transpose(final_matrix2 @ final_matrix3 @ final_matrix4, axes=[0,2,1])
        vismatrix2 = -np.transpose(final_matrix1 @ final_matrix2, axes=[0,2,1]) @ final_matrix @ np.transpose(final_matrix2 @ final_matrix3 @ final_matrix4, axes=[0,2,1])
        vismatrix3 = np.transpose(final_matrix1 @ final_matrix2, axes=[0,2,1]) @ final_matrix @ np.transpose(final_matrix4, axes=[0,2,1])
        vismatrix4 = -np.transpose(final_matrix1 @ final_matrix2 @ final_matrix3 @ final_matrix4, axes=[0,2,1]) @ final_matrix @ np.transpose(final_matrix4, axes=[0,2,1])
           
        A4 = np.conjugate(np.asarray(self.A4))
        
        if self.order == 'ABCD':
            vismatrix2 = vismatrix2.conjugate()
        if self.order == 'ABDC':
            vismatrix2 = vismatrix2.conjugate()
        if self.order == 'ACBD':
            vismatrix3 = vismatrix3.conjugate()
        if self.order == 'ACDB':
            vismatrix4 = vismatrix4.conjugate()
        if self.order == 'ADBC':
            vismatrix3 = vismatrix3.conjugate()
        if self.order == 'ADCB':
            vismatrix4 = vismatrix4.conjugate()
        
        brightness_matrix = np.zeros((2, 2, self.domain.shape[1]), dtype=complex)
        brightness_matrix[0,0] = A4[0].transpose() @ vismatrix1[:,0,0] \
                    + A4[1].transpose() @ vismatrix2[:,0,0] \
                    + A4[2].transpose() @ vismatrix3[:,0,0] \
                    + A4[3].transpose() @ vismatrix4[:,0,0]
        brightness_matrix[0,1] = A4[0].transpose() @ vismatrix1[:,0,1] \
                    + A4[1].transpose() @ vismatrix2[:,0,1] \
                    + A4[2].transpose() @ vismatrix3[:,0,1] \
                    + A4[3].transpose() @ vismatrix4[:,0,1]
        brightness_matrix[1,0] = A4[0].transpose() @ vismatrix1[:,1,0] \
                    + A4[1].transpose() @ vismatrix2[:,1,0] \
                    + A4[2].transpose() @ vismatrix3[:,1,0] \
                    + A4[3].transpose() @ vismatrix4[:,1,0]
        brightness_matrix[1,1] = A4[0].transpose() @ vismatrix1[:,1,1] \
                    + A4[1].transpose() @ vismatrix2[:,1,1] \
                    + A4[2].transpose() @ vismatrix3[:,1,1] \
                    + A4[3].transpose() @ vismatrix4[:,1,1]
                    
        toret = np.zeros(self.domain.shape, dtype=complex)
        
        toret[0] = brightness_matrix[0,0] + brightness_matrix[1,1]
        toret[1] = brightness_matrix[0,1] + brightness_matrix[1,0]
        toret[2] = 1j*brightness_matrix[0,1] - 1j*brightness_matrix[1,0]
        
        return 0.5*np.real(toret)
    
    
    
    
    
    