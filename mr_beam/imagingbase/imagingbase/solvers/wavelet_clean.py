from imagingbase.regpy_utils import RegpySolver as Solver

import logging
from regpy.util import classlogger
import numpy as np

from copy import deepcopy

from MSI.MSDecomposition import WaveletTransform2D as msiWaveletTransform2D
from MSI.MSDecomposition import DoG2D as msiDoG2D
from MSI.MSMDDecomposition import DoG2D as msmdDoG2D
from MSI.MSDecomposition import Bessel2D as msiBessel2D
from MSI.MSMDDecomposition import Bessel2D as msmdBessel2D

from joblib import Parallel, delayed

from imagingbase.solvers.utils import shift2D


class DOGCLEAN(Solver):
    def __init__(self, dmap, dbeam, psf, window, widths, add_only_positive_coeff=False, gain=0.1, angle=0, ellipticities=1, num_cores=1, md=False, smoothing_scale=None, weights=None, threshold=0.1, parallel=True, **args):
        self.num_cores = num_cores 
        
        super().__init__()
        self.add_only_positive_coeff = add_only_positive_coeff
        self.gain = gain
        self.window = window.flatten()
        
        self.md = md
        self.ellipticities = ellipticities
        self.angle = angle

        self.dmap = dmap
        self.dbeam = dbeam
        self.psf = psf
        
        self.flux_beam = np.sum(self.dbeam)
        
        self.dmap /= self.flux_beam
        self.dbeam /= self.flux_beam

        self.shape = np.asarray( self.dbeam.shape )
        #Indices on which to find peaks in CLEAN iterations
        #self.indices = np.arange(0, np.prod(self.shape))[self.window]

        #Sanity Checks
        assert self.dmap.shape == self.dbeam.shape
        assert self.psf.shape == self.dbeam.shape
        assert len(self.dbeam.shape) == 2
        #assert that image dimension is even
        #assert ( (self.shape//2)*2 == np.asarray(self.shape) ).any()
        
        #Wavelet Transform Object
        self.log.info("Initialize MS-decompositions ...")
        if self.md:
            self.dog_trafo_clean = msmdDoG2D(widths, angle=angle, ellipticities=ellipticities, all_scales=True, smoothing_scale=smoothing_scale)    
        else:
            self.dog_trafo_clean = msiDoG2D(widths, angle=angle, ellipticities=ellipticities, all_scales=True, smoothing_scale=smoothing_scale)
        self.dog_trafo_dirty = deepcopy( self.dog_trafo_clean )
        self.dog_trafo_dirty.addbeam(self.dbeam)
      
        #Decompose Beam
        self.log.info("Decompose Beam ...")
        dog_beam = self.dog_trafo_clean.decompose( self.dbeam )
        self.length = len(dog_beam)
        self.beam_list = []
        self.normalization = []
        
        self.mask = np.column_stack([self.window]*self.length).flatten()
        self.indices = np.arange(0, self.length*np.prod(self.shape))[self.mask]

        self.log.info("Compute Cross-terms ...")
        self.beam_list = np.zeros((self.length, 2*(self.shape[0]//2)+self.shape[0], 2*(self.shape[1]//2)+self.shape[1], self.length), dtype=float) 
        
        results = []
        if parallel:
            results=Parallel(n_jobs=self.num_cores)(delayed(self._decompose_beam)(dog_beam[i][0]) for i in range(self.length))   
        else:
            for i in range(self.length):
                results.append(self._decompose_beam(dog_beam[i][0]))
                   
        for i in range(self.length):
            for j in range(self.length):
                self.beam_list[i, :, :, j] = np.pad(results[i][0][j], self.shape//2)
            self.normalization.append(results[i][1])

        self.log.info("Decompose psf ...")
        dog_psf = self.dog_trafo_clean.decompose( self.psf )
        self.psf_list = []
        self.power_scales = []
        for i in range(self.length):
            psf_scale = dog_psf[i][0]
            #self.power_scales.append( np.linalg.norm(psf_scale) )
            #self.power_scales.append( np.max(psf_scale) )
            self.psf_list.append( psf_scale/self.normalization[i] )

        #decompose dirty map in list of scales. CLEANing is done on these scales
        self.log.info("Decompose map ...") 
        dog_map = self.dog_trafo_dirty.decompose( self.dmap )
        self.map_list = np.zeros((self.shape[0], self.shape[1], self.length))
        for i in range(self.length):
            self.map_list[:,:,i] = dog_map[i][0]

        self.y = self.dmap.flatten()
        self.x = []

        #store the constant remainder
        #for i in range(self.length):
        #    self.dmap -= self.map_list[i]
        
        self.counter = 0

        self.log.info("Compute weights ...")
        if weights == None:
            weights = np.ones(self.length)
        self.weights = weights
        assert len(self.weights) == self.length
        #self.weights /= np.array(self.power_scales)

        dirac = np.zeros(self.shape)
        dirac[self.shape[0]//2, self.shape[1]//2] = 1 
        beams = self.dog_trafo_dirty.decompose(dirac)
        beams_clean = self.dog_trafo_clean.decompose(dirac)
        for i in range(len(self.weights)):
            self.power_scales.append( np.linalg.norm(beams[i][0]) * np.linalg.norm(beams_clean[i][0]) )
            #self.power_scales.append(1)
        self.weights /= np.array(self.power_scales)
        
        self.threshold = np.zeros(len(self.weights))
        if "udbeam" in args.keys():
            u_trafo_dirty = deepcopy( self.dog_trafo_clean )
            u_trafo_dirty.addbeam(args.get("udbeam"))
            beams = u_trafo_dirty.decompose(dirac)
            
        for i in range(len(self.weights)):
            self.threshold[i] = (np.linalg.norm(beams[i][0])/np.linalg.norm(beams_clean[i][0]) > threshold)
        self.weights *= self.threshold
        
    def _decompose_beam(self, image):
        dog_beam_scale = self.dog_trafo_dirty.decompose( image )
        #Normalize beam scales to 1
        #normalization = np.max( dog_beam_scale[i][0] )
        normalization = np.max( image )
        beam_scale = []
        for j in range(self.length):
            beam_scale.append( dog_beam_scale[j][0] / normalization )
        return [beam_scale, normalization]    
        
    def _minor_loop(self):
        slice_min = self.shape-self.index
        slice_max = slice_min+self.shape
        #minor loop at every subscale
        self.map_list -= self.gain * self.strength * self.beam_list[self.scale, slice_min[0]:slice_max[0], slice_min[1]:slice_max[1], :]
 
    def _minor_loop_util(self, to_shift, indices, strength):
        return strength * shift2D(to_shift, indices-self.shape//2)

    def _find_peak(self):   
        if self.add_only_positive_coeff:
            args =  np.asarray( np.unravel_index(
                self.indices[ np.argmax( (self.weights * self.map_list).flatten() [self.mask]) ],
                (self.shape[0],self.shape[1],self.length) ) )
        else:    
            args =  np.asarray( np.unravel_index(
                self.indices[ np.argmax( abs( (self.weights * self.map_list).flatten() [self.mask]) ) ],
                (self.shape[0],self.shape[1],self.length) ) )
        self.scale = args[2]
        self.index = np.asarray([args[0], args[1]])
        self.strength = self.map_list[self.index[0], self.index[1], self.scale]


    def _next(self):
        #major loop
        self._find_peak()
        self.x.append([self.scale, self.index, self.strength, self.gain])
        self._minor_loop()
        self.counter += 1
        #if (self.counter//50)*50 == self.counter:
        #    self.y = (self.dmap+np.sum(self.map_list, axis=0)).flatten()
        #    print('rms', np.std(self.y[self.window]), 'at iteration: ', self.counter)

    
    def finalize(self):
        self.dmap = np.sum(self.map_list, axis=2)
        cmap = self.dmap.copy()
        
        result = Parallel(n_jobs=self.num_cores)(delayed(self._minor_loop_util)(self.psf_list[item[0]], item[1], self.gain*item[2]) for item in self.x)
        cmap += np.sum( np.array(result), axis=0)
        return cmap
    
    def intermediate(self):
        cmap = np.zeros(self.shape)
        
        result = Parallel(n_jobs=self.num_cores)(delayed(self._minor_loop_util)(self.psf_list[item[0]], item[1], self.gain*item[2]) for item in self.x)
        cmap += np.sum( np.array(result), axis=0)        

        self.x = []
        return cmap

    def update_map_list(self, dmap):
        self.dmap = dmap / self.flux_beam

        dog_map = self.dog_trafo_dirty.decompose( self.dmap )
        self.map_list = np.zeros((self.shape[0], self.shape[1], self.length))
        for i in range(self.length):
            self.map_list[:,:,i] = dog_map[i][0]





##################################################################################################

class BesselCLEAN(Solver):
    def __init__(self, dmap, dbeam, psf, window, widths, gain=0.1, angle=0, ellipticities=1, num_cores=1, md=False, add_only_positive_coeff=False, support=None, smoothing_scale=None, weights=None, threshold=0.1, paralle=True, **args):
        self.num_cores = num_cores
        
        super().__init__()
        self.add_only_positive_coeff = add_only_positive_coeff
        self.gain = gain
        self.window = window.flatten()
        
        self.md = md
        self.ellipticities = ellipticities
        self.angle = angle

        self.dmap = dmap
        self.dbeam = dbeam
        self.psf = psf
        
        self.flux_beam = np.sum(self.dbeam)
        
        self.dmap /= self.flux_beam
        self.dbeam /= self.flux_beam

        self.shape = np.asarray( self.dbeam.shape )
        #Indices on which to find peaks in CLEAN iterations
        #self.indices = np.arange(0, np.prod(self.shape))[self.window]

        #Sanity Checks
        assert self.dmap.shape == self.dbeam.shape
        assert self.psf.shape == self.dbeam.shape
        assert len(self.dbeam.shape) == 2
        #assert that image dimension is even
        #assert ( (self.shape//2)*2 == np.asarray(self.shape) ).any()
        
        
        #Wavelet Transform Object
        self.log.info("Initialize MS-decompositions ...")
        if self.md:
            self.trafo_clean = msmdBessel2D(widths, support=support, angle=angle, ellipticities=ellipticities, all_scales=True, smoothing_scale=smoothing_scale, **args)    
        else:
            self.trafo_clean = msiBessel2D(widths, support=support, angle=angle, ellipticities=ellipticities, all_scales=True, smoothing_scale=smoothing_scale, **args)
        self.trafo_dirty = deepcopy( self.trafo_clean )
        self.trafo_dirty.addbeam(self.dbeam)
      
        #Decompose Beam
        self.log.info("Decompose Beam ...")
        trafo_beam = self.trafo_clean.decompose( self.dbeam )
        self.length = len(trafo_beam)
        self.beam_list = []
        self.normalization = []
        
        self.mask = np.column_stack([self.window]*self.length).flatten()
        self.indices = np.arange(0, self.length*np.prod(self.shape))[self.mask]

        self.log.info("Compute Cross-terms ...")
        self.beam_list = np.zeros((self.length, 2*(self.shape[0]//2)+self.shape[0], 2*(self.shape[1]//2)+self.shape[1], self.length), dtype=float) 
        results=[]
        if parallel:
            results=Parallel(n_jobs=self.num_cores)(delayed(self._decompose_beam)(trafo_beam[i][0]) for i in range(self.length))
        else:
            for i in range(self.length):
                results.append(self._decompose_beam(trafo_beam[i][0]))
          
        for i in range(self.length):
            for j in range(self.length):
                self.beam_list[i, :, :, j] = np.pad(results[i][0][j], self.shape//2)
            self.normalization.append(results[i][1])

        self.log.info("Decompose psf ...")
        trafo_psf = self.trafo_clean.decompose( self.psf )
        self.psf_list = []
        self.power_scales = []
        for i in range(self.length):
            psf_scale = trafo_psf[i][0]
            #self.power_scales.append( np.linalg.norm(psf_scale) )
            #self.power_scales.append( np.max(psf_scale) )
            self.psf_list.append( psf_scale/self.normalization[i] )

        #decompose dirty map in list of scales. CLEANing is done on these scales
        self.log.info("Decompose map ...") 
        trafo_map = self.trafo_dirty.decompose( self.dmap )
        self.map_list = np.zeros((self.shape[0], self.shape[1], self.length))
        for i in range(self.length):
            self.map_list[:,:,i] = trafo_map[i][0]

        self.y = self.dmap.flatten()
        self.x = []

        #store the constant remainder
        #for i in range(self.length):
        #    self.dmap -= self.map_list[i]
        
        self.counter = 0

        self.log.info("Compute weights ...")
        if weights == None:
            weights = np.ones(self.length)
        self.weights = weights
        assert len(self.weights) == self.length
        #self.weights /= np.array(self.power_scales)

        dirac = np.zeros(self.shape)
        dirac[self.shape[0]//2, self.shape[1]//2] = 1 
        beams = self.trafo_dirty.decompose(dirac)
        beams_clean = self.trafo_clean.decompose(dirac)
        for i in range(len(self.weights)):
            self.power_scales.append( np.linalg.norm(beams[i][0]) * np.linalg.norm(beams_clean[i][0]) )
            #self.power_scales.append(1)
        self.weights /= np.array(self.power_scales)
        
        self.threshold = np.zeros(len(self.weights))
        if "udbeam" in args.keys():
            u_trafo_dirty = deepcopy( self.trafo_clean )
            u_trafo_dirty.addbeam(args.get("udbeam"))
            beams = u_trafo_dirty.decompose(dirac)
            
        for i in range(len(self.weights)):
            self.threshold[i] = (np.linalg.norm(beams[i][0])/np.linalg.norm(beams_clean[i][0]) > threshold)
        self.weights *= self.threshold
        
        #if md:
        #    for i in range(len(widths)):
        #        index = (ellipticities+1)*(i+1)-1
        #        alpha = np.linalg.norm(beams_clean[index][0]) / np.linalg.norm(beams_clean[index-1][0])
        #        self.weights[index] /= alpha
        
    def _decompose_beam(self, image):
        trafo_beam_scale = self.trafo_dirty.decompose( image )
        #Normalize beam scales to 1
        #normalization = np.max( dog_beam_scale[i][0] )
        normalization = np.max( image )
        beam_scale = []
        for j in range(self.length):
            beam_scale.append( trafo_beam_scale[j][0] / normalization )
        return [beam_scale, normalization]    

    def _minor_loop(self):
        slice_min = self.shape-self.index
        slice_max = slice_min+self.shape
        #minor loop at every subscale
        self.map_list -= self.gain * self.strength * self.beam_list[self.scale, slice_min[0]:slice_max[0], slice_min[1]:slice_max[1], :]
 
    def _minor_loop_util(self, to_shift, indices, strength):
        return strength * shift2D(to_shift, indices-self.shape//2)

    def _find_peak(self):   
        if self.add_only_positive_coeff:
            args =  np.asarray( np.unravel_index(
                self.indices[ np.argmax( (self.weights * self.map_list).flatten() [self.mask]) ],
                (self.shape[0],self.shape[1],self.length) ) )
        else:    
            args =  np.asarray( np.unravel_index(
                self.indices[ np.argmax( abs( (self.weights * self.map_list).flatten() [self.mask]) ) ],
                (self.shape[0],self.shape[1],self.length) ) )
        self.scale = args[2]
        self.index = np.asarray([args[0], args[1]])
        self.strength = self.map_list[self.index[0], self.index[1], self.scale]


    def _next(self):
        #major loop
        self._find_peak()
        self.x.append([self.scale, self.index, self.strength, self.gain])
        self._minor_loop()
        self.counter += 1
        #if (self.counter//50)*50 == self.counter:
        #    self.y = (self.dmap+np.sum(self.map_list, axis=0)).flatten()
        #    print('rms', np.std(self.y[self.window]), 'at iteration: ', self.counter)

    
    def finalize(self):
        self.dmap = np.sum(self.map_list, axis=2)
        cmap = self.dmap.copy()
        
        result = Parallel(n_jobs=self.num_cores)(delayed(self._minor_loop_util)(self.psf_list[item[0]], item[1], self.gain*item[2]) for item in self.x)
        cmap += np.sum( np.array(result), axis=0)
        return cmap
    
    def intermediate(self):
        cmap = np.zeros(self.shape)
        
        result = Parallel(n_jobs=self.num_cores)(delayed(self._minor_loop_util)(self.psf_list[item[0]], item[1], self.gain*item[2]) for item in self.x)
        cmap += np.sum( np.array(result), axis=0)        

        self.x = []
        return cmap

    def update_map_list(self, dmap):
        self.dmap = dmap / self.flux_beam

        trafo_map = self.trafo_dirty.decompose( self.dmap )
        self.map_list = np.zeros((self.shape[0], self.shape[1], self.length))
        for i in range(self.length):
            self.map_list[:,:,i] = trafo_map[i][0]
            
##################################################################################################

class HybridCLEAN(Solver):
    
    log = classlogger
    
    def __init__(self, dmap, dbeam, psf, window, widths, widths_dogs, gain=0.1, angle=0, ellipticities=1, num_cores=1, md=False, add_only_positive_coeff=False, support=None, smoothing_scale=None, weights=None, threshold=0.1, parallel=True, **args):
        self.num_cores = num_cores
        
        super().__init__()
        self.add_only_positive_coeff = add_only_positive_coeff
        self.gain = gain
        self.window = window.flatten()
        
        self.md = md
        self.ellipticities = ellipticities
        self.angle = angle

        self.dmap = dmap
        self.dbeam = dbeam
        self.psf = psf
        
        self.flux_beam = np.sum(self.dbeam)
        
        self.dmap /= self.flux_beam
        self.dbeam /= self.flux_beam

        self.shape = np.asarray( self.dbeam.shape )
        #Indices on which to find peaks in CLEAN iterations
        #self.indices = np.arange(0, np.prod(self.shape))[self.window]

        #Sanity Checks
        assert self.dmap.shape == self.dbeam.shape
        assert self.psf.shape == self.dbeam.shape
        assert len(self.dbeam.shape) == 2
        #assert that image dimension is even
        #assert ( (self.shape//2)*2 == np.asarray(self.shape) ).any()
        
        
        #Wavelet Transform Object
        self.log.info("Initialize MS-decompositions ...")
        if self.md:
            self.bessel_trafo_clean = msmdBessel2D(widths, support=support, angle=angle, ellipticities=ellipticities, all_scales=True, smoothing_scale=smoothing_scale, **args)    
        else:
            self.bessel_trafo_clean = msiBessel2D(widths, support=support, angle=angle, ellipticities=ellipticities, all_scales=True, smoothing_scale=smoothing_scale, **args)
        self.bessel_trafo_dirty = deepcopy( self.bessel_trafo_clean )
        self.bessel_trafo_dirty.addbeam(self.dbeam)
        
        #Wavelet Transform Object
        if self.md:
            self.dog_trafo_clean = msmdDoG2D(widths_dogs, angle=angle, ellipticities=ellipticities, all_scales=True, **args)    
        else:
            self.dog_trafo_clean = msiDoG2D(widths_dogs, angle=angle, ellipticities=ellipticities, all_scales=True, **args)
        self.dog_trafo_dirty = deepcopy( self.dog_trafo_clean )
        self.dog_trafo_dirty.addbeam(self.dbeam)
      
        #Decompose Beam
        self.log.info("Decompose Beam ...")
        trafo_beam = self.dog_trafo_clean.decompose( self.dbeam )
        self.length = len(trafo_beam)
        self.beam_list = []
        self.normalization = []
        
        self.mask = np.column_stack([self.window]*self.length).flatten()
        self.indices = np.arange(0, self.length*np.prod(self.shape))[self.mask]

        self.log.info("Compute Cross-terms ...")
        self.beam_list = np.zeros((self.length, 2*(self.shape[0]//2)+self.shape[0], 2*(self.shape[1]//2)+self.shape[1], self.length), dtype=float) 
        results=[]
        if parallel:
            results=Parallel(n_jobs=self.num_cores)(delayed(self._decompose_beam)(trafo_beam[i][0]) for i in range(self.length))           
        else:
            for i in range(self.length):
                results.append(self._decompose_beam(trafo_beam[i][0]))
        for i in range(self.length):
            for j in range(self.length):
                self.beam_list[i, :, :, j] = np.pad(results[i][0][j], self.shape//2)
            self.normalization.append(results[i][1])

        self.log.info("Decompose psf ...")
        trafo_psf = self.dog_trafo_clean.decompose( self.psf )
        self.psf_list = []
        self.power_scales = []
        for i in range(self.length):
            psf_scale = trafo_psf[i][0]
            #self.power_scales.append( np.linalg.norm(psf_scale) )
            #self.power_scales.append( np.max(psf_scale) )
            self.psf_list.append( psf_scale/self.normalization[i] )

        #decompose dirty map in list of scales. CLEANing is done on these scales
        self.log.info("Decompose map ...")        
        trafo_map = self.bessel_trafo_dirty.decompose( self.dmap )
        self.map_list = np.zeros((self.shape[0], self.shape[1], self.length))
        for i in range(self.length):
            self.map_list[:,:,i] = trafo_map[i][0]

        self.y = self.dmap.flatten()
        self.x = []

        #store the constant remainder
        #for i in range(self.length):
        #    self.dmap -= self.map_list[i]
        
        self.counter = 0

        self.log.info("Compute weights ...")
        if weights == None:
            weights = np.ones(self.length)
        self.weights = weights
        assert len(self.weights) == self.length
        #self.weights /= np.array(self.power_scales)

        dirac = np.zeros(self.shape)
        dirac[self.shape[0]//2, self.shape[1]//2] = 1 
        beams = self.bessel_trafo_dirty.decompose(dirac)
        beams_clean = self.bessel_trafo_clean.decompose(dirac)
        for i in range(len(self.weights)):
            self.power_scales.append( np.linalg.norm(beams[i][0]) * np.linalg.norm(beams_clean[i][0]) )
            #self.power_scales.append(1)
        self.weights /= np.array(self.power_scales)
        
        self.threshold = np.zeros(len(self.weights))
        if "udbeam" in args.keys():
            u_bessel_trafo_dirty = deepcopy( self.bessel_trafo_clean )
            u_bessel_trafo_dirty.addbeam(args.get("udbeam"))
            beams = u_bessel_trafo_dirty.decompose(dirac)
            
        for i in range(len(self.weights)):
            self.threshold[i] = (np.linalg.norm(beams[i][0])/np.linalg.norm(beams_clean[i][0]) > threshold)
        self.weights *= self.threshold
        
    def _decompose_beam(self, image):
        trafo_beam_scale = self.bessel_trafo_dirty.decompose( image )
        #Normalize beam scales to 1
        #normalization = np.max( dog_beam_scale[i][0] )
        normalization = np.max( image )
        beam_scale = []
        for j in range(self.length):
            beam_scale.append( trafo_beam_scale[j][0] / normalization )
        return [beam_scale, normalization]    

    def _minor_loop(self):
        slice_min = self.shape-self.index
        slice_max = slice_min+self.shape
        #minor loop at every subscale
        self.map_list -= self.gain * self.strength * self.beam_list[self.scale, slice_min[0]:slice_max[0], slice_min[1]:slice_max[1], :]
 
    def _minor_loop_util(self, to_shift, indices, strength):
        return strength * shift2D(to_shift, indices-self.shape//2)

    def _find_peak(self):   
        if self.add_only_positive_coeff:
            args =  np.asarray( np.unravel_index(
                self.indices[ np.argmax( (self.weights * self.map_list).flatten() [self.mask]) ],
                (self.shape[0],self.shape[1],self.length) ) )
        else:    
            args =  np.asarray( np.unravel_index(
                self.indices[ np.argmax( abs( (self.weights * self.map_list).flatten() [self.mask]) ) ],
                (self.shape[0],self.shape[1],self.length) ) )
        self.scale = args[2]
        self.index = np.asarray([args[0], args[1]])
        self.strength = self.map_list[self.index[0], self.index[1], self.scale]


    def _next(self):
        #major loop
        self._find_peak()
        self.x.append([self.scale, self.index, self.strength, self.gain])
        self._minor_loop()
        self.counter += 1
        #if (self.counter//50)*50 == self.counter:
        #    self.y = (self.dmap+np.sum(self.map_list, axis=0)).flatten()
        #    print('rms', np.std(self.y[self.window]), 'at iteration: ', self.counter)

    
    def finalize(self):
        self.dmap = np.sum(self.map_list, axis=2)
        cmap = self.dmap.copy()
        
        result = Parallel(n_jobs=self.num_cores)(delayed(self._minor_loop_util)(self.psf_list[item[0]], item[1], self.gain*item[2]) for item in self.x)
        cmap += np.sum( np.array(result), axis=0)
        return cmap
    
    def intermediate(self):
        cmap = np.zeros(self.shape)
        
        result = Parallel(n_jobs=self.num_cores)(delayed(self._minor_loop_util)(self.psf_list[item[0]], item[1], self.gain*item[2]) for item in self.x)
        cmap += np.sum( np.array(result), axis=0)        

        self.x = []
        return cmap

    def update_map_list(self, dmap):
        self.dmap = dmap / self.flux_beam

        trafo_map = self.bessel_trafo_dirty.decompose( self.dmap )
        self.map_list = np.zeros((self.shape[0], self.shape[1], self.length))
        for i in range(self.length):
            self.map_list[:,:,i] = trafo_map[i][0]