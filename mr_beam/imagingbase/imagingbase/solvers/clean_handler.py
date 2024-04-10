from imagingbase.solvers.wavelet_clean import DOGCLEAN, BesselCLEAN, HybridCLEAN
from imagingbase.solvers.clean import CLEAN as HCLEAN
from imagingbase.solvers.auto_clean import CLEAN as AutoCLEAN
from regpy.operators import Convolution
from imagingbase.operators.msi import DOGDictionary, BesselDictionary
from regpy.discrs import Discretization
from imagingbase.ehtim_wrapper import EhtimObsdata
from imagingbase.solvers.utils import BuildMerger

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
from scipy.ndimage import shift
from IPython.display import display

import regpy.stoprules as rules

from ehtim.calibrating import self_cal as sc
from ehtim.image import make_square
from ehtim.plotting.comp_plots import plotall_obs_compare, plot_cphase_obs_compare, plot_bl_obs_compare, plotall_obs_im_compare
from ehtim import RADPERUAS
import ehtim.observing.obs_helpers as obsh

import logging
from regpy.util import classlogger

from copy import deepcopy 

TRANSFORMS = ['DoG', 'Wt', 'Bessel', 'Hogbom', 'Hybrid', 'Auto']
MSTRANSFORMS = ['DoG', 'Wt', 'Bessel', 'Hybrid']
PLOTOPTIONS = ['dmap', 'cmap', 'dbeam', 'psf', 'window', 'fit', 'test', 'scales', 'cphase', 'phase', 'amp', 'cmps', 'reco_scale']
CLIST = ['b', 'g', 'r', 'c', 'm', 'y']

#Handler to handle Multiscale Clean iterations interactively
#Call methods from ipython console
#Arguments:
#->wrapper: regpy/ehtim_wrapper/EhtimWrapper object containing the observation file, prior, ...
#->transform: which transform do we use for multiscale approach, select Hogbom for standard CLEAN
#->widths: widths for multiscale decomposition
#->power: power of std-weighting of data points, optional (default: 0)
#->uweight: weighting of uv-points, 0 for natural weighting, 1 for uniform weighting
#->psf_from_prior: compute psf from prior (wrapper)
#->args: keyword arguments for the computation of multiscale decompositions

class CLEAN:
    
    log = classlogger
    
    def __init__(self, wrapper, transform, widths, power=None, uweight=None, psf_from_prior=False, udbeam_threshold=False, **args):
        self.log.info("Initialize solver ...")
        self.wrapper = wrapper
        assert transform in TRANSFORMS
        self.transform = transform
        self.widths = widths

        #Select correct power and uv-weight in observation file
        if power != None and uweight != None:
            self.wrapper.Obsdata.uvweight(power, weight=uweight)
        if power != None and uweight == None:
            self.wrapper.Obsdata.uvweight(power, weight=uweight)
        self.power = self.wrapper.Obsdata.power
        self.uweight = self.wrapper.Obsdata.weight

        self.args = args
        
        self.md = False
        if "md" in self.args.keys():
            self.md = self.args.get("md")
            
        self.padding_factor = 1
        if "padding_factor" in self.args.keys():
            self.padding_factor = self.args.get("padding_factor")
            
        self.natural_correction = False
        if "natural_correction" in self.args.keys():
            self.natural_correction = self.args.get("natural_correction")
            
        xdim = (self.wrapper.Prior.xdim // 2) * 2
        assert xdim+1 == self.wrapper.Prior.xdim

        self.log.info("Initialize dirty beam and dirty map ...")
        #Find dmap, dbeam and psf from ehtim wrapper
        if "sdmap" in self.args.keys():
            self.dmap = self.args.get("sdmap")
            if "sdbeam" in self.args.keys():
                self.dbeam = self.args.get("sdbeam")
            else:
                self.dbeam = self.wrapper.Obsdata.dirtybeam(self.padding_factor * self.wrapper.Prior.xdim // 2 *2, self.padding_factor * self.wrapper.Prior.fovx(), natural_correction=self.natural_correction).imarr()[(self.padding_factor-1)*xdim//2:(self.padding_factor+1)*xdim//2+1, (self.padding_factor-1)*xdim//2:(self.padding_factor+1)*xdim//2+1]
        else:
            dmap, dbeam = self.wrapper.Obsdata.dirty_image_beam(self.padding_factor * xdim + 1, self.padding_factor * self.wrapper.Prior.fovx(), natural_correction=self.natural_correction)
            self.dmap = dmap.imarr()[(self.padding_factor-1)*xdim//2:(self.padding_factor+1)*xdim//2+1, (self.padding_factor-1)*xdim//2:(self.padding_factor+1)*xdim//2+1]
            self.dbeam = dbeam.imarr()[(self.padding_factor-1)*xdim//2:(self.padding_factor+1)*xdim//2+1, (self.padding_factor-1)*xdim//2:(self.padding_factor+1)*xdim//2+1]
        #self.dmap = self.wrapper.Obsdata.dirtyimage(self.wrapper.Prior.xdim, self.wrapper.Prior.fovx()).imarr().copy()
        self.initial_dmap = self.dmap.copy()
        #self.dbeam = self.wrapper.Obsdata.dirtybeam(self.wrapper.Prior.xdim, self.wrapper.Prior.fovx()).imarr().copy()
        if psf_from_prior:
            self.psf = self.wrapper.Prior.imarr().copy()
        else:
            self.psf = self.wrapper.Obsdata.cleanbeam(self.wrapper.Prior.xdim, self.wrapper.Prior.fovx()).imarr().copy()
        self.psf /= np.sum(self.psf)
        
        self.udbeam = None
        if udbeam_threshold:
            self.wrapper.Obsdata.uvweight(0, weight=1)
            self.udbeam = self.wrapper.Obsdata.dirtybeam(self.padding_factor * xdim + 1, self.padding_factor * self.wrapper.Prior.fovx()).imarr()[(self.padding_factor-1)*xdim//2:(self.padding_factor+1)*xdim//2+1, (self.padding_factor-1)*xdim//2:(self.padding_factor+1)*xdim//2+1]
            self.wrapper.Obsdata.uvweight(self.power, weight=self.uweight)

        #Initilialize window
        window = np.zeros(self.dmap.shape, dtype=bool)
 
        #Initialize solver object
        if self.transform == 'DoG':
            self.solver = DOGCLEAN(self.dmap.copy(), self.dbeam.copy(), self.psf, window, self.widths, num_cores=wrapper.num_cores, **self.args)
            grid = Discretization(self.dmap.shape)
            domain = grid**(self.solver.length)
            self.merger = DOGDictionary(domain, grid, self.widths, grid.shape, num_cores=wrapper.num_cores, **self.args)
            if self.solver.dog_trafo_clean.update_smoothing_scale:
                self.merger.set_smoothing_scale(self.solver.dog_trafo_clean.smoothing_scale)
                
        if self.transform == 'Bessel':
            self.solver = BesselCLEAN(self.dmap.copy(), self.dbeam.copy(), self.psf, window, self.widths, num_cores=wrapper.num_cores, **self.args)
            grid = Discretization(self.dmap.shape)
            domain = grid**(self.solver.length)
            self.merger = BesselDictionary(domain, grid, self.widths, grid.shape, num_cores=wrapper.num_cores, **self.args)
            if self.solver.trafo_clean.update_smoothing_scale:
                self.merger.set_smoothing_scale(self.solver.trafo_clean.smoothing_scale)
                
        if self.transform == 'Hybrid':
            grid = Discretization(self.dmap.shape)
            domain = grid**(len(widths))
            self.log.info("Initialize merger ...")
            self.merger = BesselDictionary(domain, grid, self.widths, grid.shape, num_cores=wrapper.num_cores, **self.args)
            Builder = BuildMerger(self)
            self.log.info("Build merger ...")
            self.merger = Builder._build_merger()
            self.widths_dogs = self.merger.dogdict.widths.copy()
            self.log.info("Build solver ...")
            self.solver = HybridCLEAN(self.dmap.copy(), self.dbeam.copy(), self.psf, window, self.widths, self.widths_dogs, num_cores=wrapper.num_cores, udbeam=self.udbeam, **self.args)
                
        if self.transform == 'Hogbom':
            self.solver = HCLEAN(self.dmap.copy(), self.dbeam.copy(), self.psf.copy(), window)

        if self.transform == 'Auto':
            self.log.info("Initialize Autocorrelation preproducts")
            if "window" not in self.args.keys():
                self.args["window"] = np.ones(self.dmap.shape, dtype=bool)
            
            if "window_corr" not in self.args.keys():
                self.args["window_corr"] = self.args.get("window")

            self.solver = AutoCLEAN(self.dmap.copy(), self.dbeam.copy(), self.psf.copy(), **self.args)
            
        self.reco = np.zeros(self.dmap.shape)
        
        self.log.info("Finalize initialization ...")
        #Initialize convolution object (forward operator)
        self.conv = Convolution(Discretization(self.dbeam.shape), self.dbeam)
        self.conv_psf = Convolution(Discretization(self.psf.shape), self.psf)

        #Initialize window for plotting and computation of rms
        self.plotting_bounds = np.array([0, self.dmap.shape[0], 0, self.dmap.shape[1]])
        self.plotting_window = np.zeros(self.dmap.shape, dtype=int)
        self.plotting_window[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]] = True

        self.cmap_list = []
        
        #Allocate keywords for later control
        self.obs_computed = False
        self.update_uniform_dmap = False
        
        self.levels = 0.005 * 2**(np.arange(16)/2)
        
    #sets the window for plotting interactively from console input
    def set_bounds(self):
        self.log.info('Specify Plotting Bounds')
        limits = input()
        if limits == '':
            pass
        else:
            limits = np.asarray(eval(limits))
            assert limits.dtype == int
            assert len(limits) == 4
            self.plotting_bounds = limits
        self.log.info('{}'.format(self.plotting_bounds))
        self.plotting_window *= 0
        self.plotting_window[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]] = 1

    #same, but from script input
    def set_bounds_from_script(self, limits):
        self.log.info('Specify Plotting Bounds')
        self.log.info('{}'.format(limits))
        assert limits.dtype == int
        assert len(limits) == 4
        self.plotting_bounds = limits
        self.plotting_window *= 0
        self.plotting_window[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]] = 1

    #sets window, weights (scales) and gain interactively from console input
    def precompute(self):
        self.log.info("Specify Weighting")
        weights = input()
        if weights == '':
            pass
        else:
            if self.transform in MSTRANSFORMS:
                weights = np.asarray(eval(weights))
                assert len(weights) == self.solver.length
                assert weights.dtype == float
                self.solver.weights = weights
                self.solver.weights /= np.array(self.solver.power_scales)  
                self.solver.weights *= self.solver.threshold
                #self.solver.weights /= self.solver.maxs

        self.log.info("Specify Gain")
        gain = input()
        if gain == '':
            pass
        else:
            self.solver.gain = float(gain)
            
        self.log.info("Specify window")
        limits = input()
        if limits == '':
            pass
        else:
            limits = np.asarray(eval(limits))
            assert limits.dtype == int
            assert len(limits) == 4
            self.limits = limits
            self.solver.window = self.solver.window.reshape(self.dmap.shape)
            self.solver.window[limits[0]:limits[1], limits[2]:limits[3]] = True
            self.solver.window = self.solver.window.flatten()
            if self.transform in MSTRANSFORMS:
                self.solver.mask = np.column_stack([self.solver.window]*self.solver.length).flatten()
                self.solver.indices = np.arange(0, self.solver.length*np.prod(self.solver.shape))[self.solver.mask]
            else:
                self.solver.indices = np.arange(0, np.prod(self.solver.shape))[self.solver.window]
    
    #sets window from input
    def update_window(self, window):
        self.log.info("Update window")
        self.solver.window = window.copy()
        if self.transform in MSTRANSFORMS:
            self.solver.mask = np.column_stack([self.solver.window]*self.solver.length).flatten()
            self.solver.indices = np.arange(0, self.solver.length*np.prod(self.solver.shape))[self.solver.mask]
        else:
            self.solver.indices = np.arange(0, np.prod(self.solver.shape))[self.solver.window]
    
    #sets weights (scales) from input  
    def update_weights(self, weights):
        if self.transform in MSTRANSFORMS:
            self.log.info("Update weights")
            self.log.info("{}".format(weights))
            assert len(weights) == self.solver.length
            assert weights.dtype == float
            self.solver.weights = weights.copy()
            self.solver.weights /= np.array(self.solver.power_scales) 
            self.solver.weights *= self.solver.threshold
            #self.solver.weights /= self.solver.maxs
    
    #internal helper routine for the selection of special scales
    def _select_scales(self, scales_list):
        scales_list = np.asarray(scales_list, dtype=float)
        weights = np.zeros(self.solver.length, dtype=float)
        for i in range(self.solver.length):
            weights[i] = (float(i) in scales_list)
        self.update_weights(np.asarray(weights, dtype=float))
    
    #select scales specified in scales_list for cleaning only    
    def select_scales(self, scales_list):
        assert self.transform in MSTRANSFORMS
        if self.solver.md:
            scales_list = np.asarray(scales_list, dtype=float)
            scales = []
            for i in range(len(self.widths)):
                if i in scales_list:
                    for j in range(self.solver.ellipticities+1):
                        scales.append( (self.solver.ellipticities+1) * i + j )
            self._select_scales(scales)
        else:
            self._select_scales(scales_list)
    
    #select all scales    
    def select_all_scales(self):
        assert self.transform in MSTRANSFORMS
        self.update_weights(np.ones(self.solver.length, dtype=float))
    
    #update gain from input    
    def update_gain(self, gain):
        self.log.info("Update gain")
        self.log.info('{}'.format(gain))
        self.solver.gain = gain

    #The same as precompute but from script-dictionary           
    def precompute_from_dictionary(self, dict):
        if self.transform in MSTRANSFORMS:
            weights = dict["weights"].copy()       
            assert len(weights) == self.solver.length
            assert weights.dtype == float
            self.solver.weights = weights
            self.solver.weights /= np.array(self.solver.power_scales)
            self.solver.weights *= self.solver.threshold
            #self.solver.weights /= self.solver.maxs        

        gain = dict["gain"]
        self.solver.gain = float(gain)

        limits = dict["limits"].copy()
        assert limits.dtype == int
        assert len(limits) == 4
        self.limits = limits
        self.solver.window = self.solver.window.reshape(self.dmap.shape)
        self.solver.window[limits[0]:limits[1], limits[2]:limits[3]] = True
        self.solver.window = self.solver.window.flatten()
        if self.transform in MSTRANSFORMS:
            self.solver.mask = np.column_stack([self.solver.window]*self.solver.length).flatten()
            self.solver.indices = np.arange(0, self.solver.length*np.prod(self.solver.shape))[self.solver.mask]
        else:
            self.solver.indices = np.arange(0, np.prod(self.solver.shape))[self.solver.window]

    #Apply startmod from wrapper.InitIm
    def startmod(self):
        self.log.info("Startmod ...")
        assert self.solver.counter == 0
        self.reco = self.wrapper.InitIm.imarr().copy()

        self.dmap = self.initial_dmap - self.conv(self.reco)
        self.solver.update_map_list(self.dmap)

    #Merge a list of components and compute recovered image, mostly used internally
    def merge(self, cmap_list, scale="all"):        
        if self.transform in MSTRANSFORMS:
            self.log.info("Merge component list")
            
            indices = self._merge(cmap_list, scale=scale)
            return self.merger(indices.flatten())
        
        if self.transform == "Hogbom":
            self.log.info("Merge component list")
            indices = np.zeros((self.solver.shape[0], self.solver.shape[1]))
            for i in range(len(cmap_list)):
                cmap = cmap_list[i]
                for j in range(len(cmap)):
                    position = cmap[j][0]
                    strength = cmap[j][1]
                    gain = cmap[j][2]
                    indices[position[0], position[1]] += gain * strength / self.solver.max_beam
            return indices
        
        if self.transform == "Auto":
            self.log.info("Merge component list")
            indices = np.zeros((self.solver.shape[0], self.solver.shape[1]))
            for i in range(len(cmap_list)):
                cmap = cmap_list[i]
                for j in range(len(cmap)):
                    position = cmap[j][0]
                    strength = cmap[j][1]
                    gain = cmap[j][2]
                    basis_function = cmap[j][3]
                    indices += gain * strength * shift(basis_function, position-self.solver.shape//2) / self.solver.max_beam
            return indices

    def _merge(self, cmap_list, scale="all"):
        assert self.transform in MSTRANSFORMS
        self.log.info("Merge component list")
        
        if scale == "all":
            scales = np.arange(self.solver.length)
        else:
            scales = scale.copy()
        
        indices = np.zeros((self.solver.length, self.solver.shape[0], self.solver.shape[1]))
        for i in range(len(cmap_list)):
            cmap = cmap_list[i]
            for j in range(len(cmap)):
                scale = cmap[j][0]
                position = cmap[j][1]
                strength = cmap[j][2]
                gain = cmap[j][3]
                if scale in scales:
                    indices[scale, position[0], position[1]] += gain * strength / self.solver.normalization[scale]
        return indices
        
    #set recovered image from input and update map list
    def set_reco(self, reco):
        self.log.info("Set restored map")
        self.reco = reco.copy()
        
        self.dmap = self.initial_dmap - self.conv(self.reco)
        self.solver.update_map_list(self.dmap)
        
        self.cmap_list = []
        
        self.obs_computed = False

    #project recovered image to positive only image
    def project_to_positive(self, reco, fraction=0):
        peak = np.max(reco)
        limit = -fraction * peak
        self.set_reco(np.maximum(reco, limit))

    #Run the solver. The number of iterations is specified by maxit.
    def run(self, maxit = None, recompute_map_list=False):
        if maxit == None:
            maxit = self.wrapper.maxit
        self.log.info('Start running CLEAN with {} iterations'.format(maxit))
        self.counter_old = self.solver.counter
        self.dmap_old = self.dmap.copy()
        if self.transform in MSTRANSFORMS:
            self.map_list_old = self.solver.map_list.copy()

        stoprule = rules.CountIterations(max_iterations=maxit)
        self.cmap, _ = self.solver.run(stoprule)
        
        self.cmap_list.append(self.cmap)
        
        self.solver.x = []

        self.log.info('Stop running CLEAN')
        self.log.info('Convolve with psf')

        self.reco_last = self.merge([self.cmap])

        self.reco += self.reco_last
        #if self.transform == "Auto":
        #    self.dmap = self.solver.dmap
        #else:
        self.dmap = self.initial_dmap - self.conv(self.reco)
        self.log.info('Solver finished')
        self.log.info('std: {}'.format(np.std(self.dmap.flatten()[self.solver.window * self.plotting_window.flatten()])))
        self.log.info('at iteration: {}'.format(self.solver.counter))
        self.log.info('total flux {}'.format(self.finalize().total_flux()))
        
        self.obs_computed = False
        
        if recompute_map_list:
            self.solver.update_map_list(self.dmap)

        return self.cmap

    #Finalize solver and format output to ehtim.image object    
    def finalize(self, psf=False):
        if psf:
            return self.wrapper.formatoutput( self.conv_psf(self.reco+self.dmap) )
        else:
            return self.wrapper.formatoutput(self.reco+self.dmap)

    #Self-calibrate the image by ehtim. Update dirty image.
    def selfcal(self, method='phase', update_map=False, solution_interval=0, scan_solutions=False, **args):
        if 'cmpnts' in args:
            cmpnts = args['cmpnts']
        else:
            cmpnts = self.reco
        self.log.info("self-calibrate")
        self.log.info(method)
        out = self.wrapper.formatoutput(cmpnts)
        self.wrapper.Obsdata = EhtimObsdata( sc.self_cal(self.wrapper.Obsdata, out, method=method, solution_interval=solution_interval, scan_solutions=scan_solutions), num_cores=self.wrapper.num_cores )

        self.wrapper.Obsdata.uvweight(self.power, weight=self.uweight)

        if update_map:
            self.dmap = self.wrapper.Obsdata.dirtyimage(self.wrapper.Prior.xdim, self.wrapper.Prior.fovx(), natural_correction=self.natural_correction).imarr().copy()
            self.initial_dmap = self.dmap.copy()
            self.dmap -= self.conv(cmpnts)
            if self.transform in MSTRANSFORMS:
                self.solver.update_map_list(self.dmap)
                
            self.update_uniform_dmap = True
            
        self.obs_computed = False
     
        #initial starting calibration with Init image
    def startcal(self, method='phase', solution_interval=0, scan_solutions=False, **args):
        self.log.info("self-calibrate")
        self.log.info(method)        
        out = self.wrapper.InitIm
        
        self.wrapper.Obsdata = EhtimObsdata( sc.self_cal(self.wrapper.Obsdata, out, method=method, solution_interval=solution_interval, scan_solutions=scan_solutions), num_cores=self.wrapper.num_cores )

        self.wrapper.Obsdata.uvweight(self.power, weight=self.uweight)

        self.dmap = self.wrapper.Obsdata.dirtyimage(self.wrapper.Prior.xdim, self.wrapper.Prior.fovx(), natural_correction=self.natural_correction).imarr().copy()
        self.initial_dmap = self.dmap.copy()
        if self.transform in MSTRANSFORMS:
            self.solver.update_map_list(self.dmap)
            
        self.update_uniform_dmap = True
        
        self.obs_computed = False
    
    #Plotting routines
    def plot(self, toplot, interactive=True, **args):
        assert toplot in PLOTOPTIONS
        if interactive:
            #dirty map
            if toplot == 'dmap':
                fig = plt.figure()
                ax = fig.gca()
                im = ax.imshow(self.dmap[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]],
                    extent=(self.plotting_bounds[2], self.plotting_bounds[3], self.plotting_bounds[1], self.plotting_bounds[0]))
                fig.colorbar(im, ax=ax)
                display(fig)
                
            #clean map    
            if toplot == 'cmap':
                fig = plt.figure()
                ax = fig.gca()
                im = ax.imshow(self.reco[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]],
                    extent=(self.plotting_bounds[2], self.plotting_bounds[3], self.plotting_bounds[1], self.plotting_bounds[0]))
                fig.colorbar(im, ax=ax)
                display(fig)

            #dirty beam
            if toplot == 'dbeam':
                fig = plt.figure()
                ax = fig.gca()
                im = ax.imshow(self.dbeam[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]],
                    extent=(self.plotting_bounds[2], self.plotting_bounds[3], self.plotting_bounds[1], self.plotting_bounds[0]))
                fig.colorbar(im, ax=ax)
                display(fig)

            #psf
            if toplot == 'psf':
                fig = plt.figure()
                ax = fig.gca()
                im = ax.imshow(self.psf[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]],
                    extent=(self.plotting_bounds[2], self.plotting_bounds[3], self.plotting_bounds[1], self.plotting_bounds[0]))
                fig.colorbar(im, ax=ax)
                display(fig)
             
            #window    
            if toplot == 'window':
                fig = plt.figure()
                ax = fig.gca()
                wind = self.solver.window.reshape(self.dmap.shape)
                ax.imshow(wind[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]],
                    extent=(self.plotting_bounds[2], self.plotting_bounds[3], self.plotting_bounds[1], self.plotting_bounds[0]))
                display(fig)
             
            #Fit of recovered image to amplitudes    
            if toplot == "fit":
                fig = plt.figure()
                ax = fig.gca()
                if self.obs_computed == False:
                    reco = self.wrapper.formatoutput(self.reco)
                    self.test_obs = reco.observe_same(self.wrapper.Obsdata, ttype=self.wrapper.ttype, add_th_noise=False)
                plotall_obs_compare([self.wrapper.Obsdata, self.test_obs], 'uvdist', 'amp', clist=['b', 'm'], conj=True, ttype=self.wrapper.ttype, debias=False, ebar=False, axis=ax)
                display(fig)
                
                self.obs_computed = True
             
            #Fit of recovered image+dirty map to amplitudes    
            if toplot == "test":
                fig = plt.figure()
                ax = fig.gca()
                reco = self.finalize()
                plotall_obs_im_compare(self.wrapper.Obsdata, reco, 'uvdist', 'amp', clist=['b', 'm'], conj=True, ttype=self.wrapper.ttype, debias=False, ebar=False, axis=ax)
                display(fig)
             
            #Fourier transform of wavelet scales    
            if toplot == 'scales':
                assert self.transform in MSTRANSFORMS
                fig = plt.figure()
                ax = fig.gca()
                dirac = np.zeros(self.dmap.shape)
                dirac[self.dmap.shape[0]//2, self.dmap.shape[1]//2] = 1
                if self.transform == "DoG":
                    res = self.solver.dog_trafo_clean.decompose(dirac)
                if self.transform == "Bessel":
                    res = self.solver.trafo_clean.decompose(dirac) 
                if self.transform == "Hybrid":
                    res = self.solver.bessel_trafo_clean.decompose(dirac)
                length = self.solver.length // len(CLIST) + 1
                clist = length * CLIST
                
                obs_list = []
                maxs = []
                for i in range(self.solver.length):
                    color = clist[i]
                    scale = self.wrapper.formatoutput( res[i][0]  / self.solver.normalization[i] )
                    obs_list.append(scale.observe_same(self.wrapper.Obsdata, ttype=self.wrapper.ttype, add_th_noise=False))
                    maxs.append(np.max(obs_list[i].unpack('amp')['amp']))
                maximum = np.max(np.asarray(maxs))
                plotall_obs_compare(obs_list, 'uvdist', 'amp', clist=clist, rangey=[0,maximum], conj=True, ttype=self.wrapper.ttype, debias=False, ebar=False, axis=ax)
                display(fig)
             
            #closure phase, antennas given by fields    
            if toplot == 'cphase':
                fig = plt.figure()
                ax = fig.gca()
                if self.obs_computed == False:
                    out = self.wrapper.formatoutput(self.reco)
                    self.test_obs = out.observe_same(self.wrapper.Obsdata, ttype=self.wrapper.ttype, add_th_noise=False)
                fields = args.get('fields')
                field1 = fields[0]
                field2 = fields[1]
                field3 = fields[2]
                plot_cphase_obs_compare([self.wrapper.Obsdata, self.test_obs], field1, field2, field3, clist=['b', 'm'], axis=ax, ttype=self.wrapper.ttype)
                display(fig)
                
                self.obs_computed = True
            
            #phases, antennas specified by fields
            if toplot == 'phase':
                fig = plt.figure()
                ax = fig.gca()
                if self.obs_computed == False:
                    out = self.wrapper.formatoutput(self.reco)
                    self.test_obs = out.observe_same(self.wrapper.Obsdata, ttype=self.wrapper.ttype, add_th_noise=False)
                fields = args.get('fields')
                field1 = fields[0]
                field2 = fields[1]
                plot_bl_obs_compare([self.wrapper.Obsdata, self.test_obs], field1, field2, 'phase', clist=['b', 'm'], debias=False, axis=ax, ttype=self.wrapper.ttype, ebar=False)
                display(fig)
                
                self.obs_computed = True
             
            #amplitudes, antennas specified by fields    
            if toplot == 'amp':
                fig = plt.figure()
                ax = fig.gca()
                if self.obs_computed == False:
                    out = self.wrapper.formatoutput(self.reco)
                    self.test_obs = out.observe_same(self.wrapper.Obsdata, ttype=self.wrapper.ttype, add_th_noise=False)
                fields = args.get('fields')
                field1 = fields[0]
                field2 = fields[1]
                plot_bl_obs_compare([self.wrapper.Obsdata, self.test_obs], field1, field2, 'amp', clist=['b', 'm'], debias=False, axis=ax, ttype=self.wrapper.ttype)
                display(fig)      
                
                self.obs_computed = True
            
            #components, plots the location of found components    
            if toplot == 'cmps':
                fig = plt.figure()
                ax = fig.gca()
                for i in range(len(self.cmap_list)):
                    cmap = self.cmap_list[i]
                    for j in range(len(cmap)):
                        if self.transform in MSTRANSFORMS:
                            position = cmap[j][1]
                        if self.transform == 'Hogbom':
                            position = cmap[j][0]
                        ax.plot(position[0], position[1], '+', color='b')
                display(fig)
             
            #recovered image from different scales only    
            if toplot == 'reco_scale':
                fig = plt.figure()
                ax = fig.gca()                
                fields = args.get('fields')
                
                assert self.transform in MSTRANSFORMS
                indices = np.zeros((self.solver.length, self.solver.shape[0], self.solver.shape[1]))
                for i in range(len(self.cmap_list)):
                    cmap = self.cmap_list[i]
                    for j in range(len(cmap)):
                            scale = cmap[j][0]
                            position = cmap[j][1]
                            strength = cmap[j][2]
                            gain = cmap[j][3]
                            if scale == fields:
                                indices[scale, position[0], position[1]] += gain * strength / self.solver.normalization[scale]
                reco_scale = self.merger(indices.flatten())
                 
                peak = np.max((self.dmap+self.reco)[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]][::-1])
                
                ax.contour(reco_scale[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]][::-1], levels=peak*self.levels)
                display(fig)
        
        else:
            if toplot == 'dmap':
                plt.figure()
                plt.imshow(self.dmap[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]],
                    extent=(self.plotting_bounds[2], self.plotting_bounds[3], self.plotting_bounds[1], self.plotting_bounds[0]))
                plt.colorbar()
                plt.show()
                
            if toplot == 'cmap':
                plt.figure()
                plt.imshow(self.reco[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]],
                    extent=(self.plotting_bounds[2], self.plotting_bounds[3], self.plotting_bounds[1], self.plotting_bounds[0]))
                plt.colorbar()
                plt.show()

            if toplot == 'dbeam':
                plt.figure()
                plt.imshow(self.dbeam[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]],
                    extent=(self.plotting_bounds[2], self.plotting_bounds[3], self.plotting_bounds[1], self.plotting_bounds[0]))
                plt.colorbar()
                plt.show()

            if toplot == 'psf':
                plt.figure()
                plt.imshow(self.psf[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]],
                    extent=(self.plotting_bounds[2], self.plotting_bounds[3], self.plotting_bounds[1], self.plotting_bounds[0]))
                plt.colorbar()
                plt.show()
                
            if toplot == 'window':
                plt.figure()
                wind = self.solver.window.reshape(self.dmap.shape)
                plt.imshow(wind[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]],
                    extent=(self.plotting_bounds[2], self.plotting_bounds[3], self.plotting_bounds[1], self.plotting_bounds[0]))
                plt.show()
                
            if toplot == "fit":
                print("Only plottable in interactive mode")
                
            if toplot == "scales":
                print("Only plottable in interactive mode")
                
            if toplot == "cphase":
                print("Only plottable in interactive mode")
                
            if toplot == "phase":
                print("Only plottable in interactive mode")

    #Display image in contour plot
    def display(self, psf=True, mode='final', interactive=True, option="window", **args):
        self.display_option = option
        
        to_plot = np.zeros(self.dmap.shape)
        if mode=='reco':
            to_plot += self.reco
        if mode=='dmap':
            to_plot += self.dmap
        if mode=='final':
            to_plot += self.dmap + self.reco
            
        peak = np.max((self.dmap+self.reco)[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]][::-1])
        if psf:
            to_plot = self.conv_psf(to_plot)
            peak = np.max(self.conv_psf(self.dmap+self.reco)[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]][::-1])
        
        xarr = np.linspace(-self.wrapper.Prior.fovx()/2, self.wrapper.Prior.fovx()/2, self.wrapper.Prior.xdim)[self.plotting_bounds[0]:self.plotting_bounds[1]]
        yarr = np.linspace(-self.wrapper.Prior.fovy()/2, self.wrapper.Prior.fovy()/2, self.wrapper.Prior.ydim)[self.plotting_bounds[2]:self.plotting_bounds[3]]  
        xarr /= RADPERUAS
        yarr /= RADPERUAS
        fig = plt.figure()
        ax = fig.gca()
        if interactive:
            ax.contour(to_plot[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]][::-1], levels=peak*self.levels, linewidths=1)
            self.RS_wind = RectangleSelector(ax, self._line_select_callback_wind,
                                             useblit=True,
                                           button=[1, 3],  # disable middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
            fig.canvas.mpl_connect('key_press_event', self._toggle_selector_wind)                               
            plt.show()
        else:
            ax.contour(xarr, yarr, to_plot[self.plotting_bounds[0]:self.plotting_bounds[1], self.plotting_bounds[2]:self.plotting_bounds[3]][::-1], levels=peak*self.levels, linewidths=1)                              
            plt.show()

    #helper interactive selection of window from display routine
    def _line_select_callback_wind(self, eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y2 = int(eclick.xdata)+self.plotting_bounds[2], self.plotting_bounds[1]-int(eclick.ydata)
        x2, y1 = int(erelease.xdata)+self.plotting_bounds[2], self.plotting_bounds[1]-int(erelease.ydata)
        print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        self.log.info('x1: {}'.format(x1))
        self.log.info('x2: {}'.format(x2))
        self.log.info('y1: {}'.format(y1))
        self.log.info('y2: {}'.format(y2))
        
        self.log.info(self.display_option)

        if self.display_option == "window":
            self.limits = np.asarray([y1, y2, x1, x2], dtype=int)
            self.solver.window = self.solver.window.reshape(self.dmap.shape)
            self.solver.window[self.limits[0]:self.limits[1], self.limits[2]:self.limits[3]] = True
            self.solver.window = self.solver.window.flatten()
            if self.transform in MSTRANSFORMS:
                self.solver.mask = np.column_stack([self.solver.window]*self.solver.length).flatten()
                self.solver.indices = np.arange(0, self.solver.length*np.prod(self.solver.shape))[self.solver.mask]
            else:
                self.solver.indices = np.arange(0, np.prod(self.solver.shape))[self.solver.window]
                
        if self.display_option == "flat":
            self.reco[int(y1):int(y2), int(x1):int(x2)] = 0
            self.set_reco(self.reco)
            
    #helper interactive selection of window from display routine
    def _toggle_selector_wind(self, event):
        print(' Key pressed.')
        if event.key == 't':
            if self.RS_wind.active:
                print(' RectangleSelector deactivated.')
                self.RS_wind.set_active(False)
            else:
                print(' RectangleSelector activated.')
                self.RS_wind.set_active(True)

    #Clear windows
    def clrwind(self, option='last'):
        print("Clear window", option)
        if option == 'last':
            self.solver.window = self.solver.window.reshape(self.dmap.shape)
            self.solver.window[self.limits[0]:self.limits[1], self.limits[2]:self.limits[3]] = False
            self.solver.window = self.solver.window.flatten()
            
            if self.transform in MSTRANSFORMS:
                self.solver.mask = np.column_stack([self.solver.window]*self.solver.length).flatten()
                self.solver.indices = np.arange(0, self.solver.length*np.prod(self.solver.shape))[self.solver.mask]
        else:
            self.solver.indices = np.arange(0, np.prod(self.solver.shape))[self.solver.window]

        if option == 'all':
            self.solver.window = np.zeros(self.solver.window.shape, dtype=bool)
            
            if self.transform in MSTRANSFORMS:
                self.solver.mask = np.column_stack([self.solver.window]*self.solver.length).flatten()
                self.solver.indices = np.arange(0, self.solver.length*np.prod(self.solver.shape))[self.solver.mask]
        else:
            self.solver.indices = np.arange(0, np.prod(self.solver.shape))[self.solver.window]
        

    #Clear components
    def clrmod(self, option='last', scale="all"):
        self.log.info("clear mod")
        self.log.info(option)
        
        self.log.info("with scales")
        self.log.info(scale)
        
        if scale == "all":
            if option == 'last':
                self.reco -= self.reco_last
                self.dmap = self.dmap_old
    
                self.solver.counter = self.counter_old
                self.solver.dmap = self.dmap.copy()
                
                if self.transform in MSTRANSFORMS:
                    self.solver.map_list = self.map_list_old.copy()
                
                del(self.cmap_list[-1])
                
                self.obs_computed = False
    
            if option == 'all':
                self.reco = np.zeros(self.dmap.shape)
                self.dmap = self.initial_dmap.copy()
                
                self.solver.counter = 0
                
                if self.transform in MSTRANSFORMS:
                    self.solver.update_map_list(self.dmap)
                else:
                    self.solver.dmap = self.dmap.copy()
                    
                self.cmap_list = []
                
                self.obs_computed = False
                
        else:
            if option == "last":
                self.reco -= self.reco_last

                for i in range(len(self.cmap)):
                    if self.cmap[i][0] in scale:
                        self.cmap[i][2] = 0
                        
                self.reco_last = self.merge([self.cmap])

                self.reco += self.reco_last
                self.dmap = self.initial_dmap - self.conv(self.reco)
                
                self.obs_computed = False
                
                self.solver.update_map_list(self.dmap)
                
                del(self.cmap_list[-1])
                self.cmap_list.append(self.cmap)
                
            if option == "all":
                self.reco = np.zeros(self.dmap.shape)

                for i in range(len(self.cmap_list)):
                    cmap = self.cmap_list[i]
                    for j in range(len(cmap)):                      
                        if cmap[j][0] in scale:
                            self.cmap_list[i][j][2] = 0
                        
                self.reco = self.merge(self.cmap_list)

                self.dmap = self.initial_dmap - self.conv(self.reco)
                
                self.obs_computed = False
                
                self.solver.update_map_list(self.dmap)
        
                
    # Add Gauss model of missing flux
    def add_gauss(self, xfwhm, yfwhm, x=0, y=0):
        out = self.wrapper.formatoutput(self.reco)
        missing_flux = self.wrapper.flux - out.total_flux()
        out = out.add_gauss(missing_flux, (xfwhm, yfwhm, x, y, 0))

        self.reco = out.imarr().copy()

        self.dmap = self.wrapper.Obsdata.dirtyimage(self.wrapper.Prior.xdim, self.wrapper.Prior.fovx(), natural_correction=self.natural_correction).imarr().copy()
        self.dmap -= self.conv(self.reco)

        if self.transform in MSTRANSFORMS:
            self.solver.update_map_list(self.dmap)

#    def modelfit(self, components, fwhm):
#        self.log.info("Start model fitting with {} components".format(components))
#        out = self.wrapper.formatoutput(self.reco)
#        missing_flux = self.wrapper.flux - out.total_flux()
#        self.log.info("and missing flux {}".format(missing_flux))
#        flux_per_component = missing_flux / components
#        initial_dmap = self.wrapper.Obsdata.dirtyimage(self.wrapper.Prior.xdim, self.wrapper.Prior.fovx()).imarr().copy()
#
#        for i in range(components):
#            argmax = np.asarray( np.unravel_index( self.solver.indices[ np.argmax(self.dmap.flatten() [self.solver.window]) ], self.solver.shape) )
#            (x, y) = -self.wrapper.Prior.fovx() / 2 + argmax * self.wrapper.Prior.psize
#            out = out.add_gauss(flux_per_component, (fwhm, fwhm, x, y, 0))
#            self.reco = out.imarr().copy()
#            self.dmap = initial_dmap - self.conv(self.reco)
#
#        self.solver.update_map_list(self.dmap)

    #Fit with largest scale only until the flux on the shortest baseline is large enough
    def modelfit(self, uv_zblcut, maxit=10, tolerance=0.2):
        assert self.transform in MSTRANSFORMS
        uvdist = self.wrapper.unpack("uvdist")
        zero_baselines = [item < uv_zblcut for item in uvdist]
        if self.wrapper.ttype == 'direct' or self.wrapper.ttype == 'nfft':
            forward = lambda x: np.max( abs(self.wrapper.A[zero_baselines] @ x) )
        if self.wrapper.ttype == 'fast':
            forward = lambda x: np.max( abs( obsh.sampler( obsh.fft_imvec(x, self.wrapper.A[0]), self.wrapper.A[1], sample_type="vis" ) ) )
        
        old_weights = self.solver.weights.copy() * np.array(self.solver.power_scales)
        
        self.select_scales([len(self.widths)-1])
        add_only_positive_coeff = self.solver.add_only_positive_coeff
        self.solver.add_only_positive_coeff = True
        
        while forward(self.reco.flatten())<(1-tolerance)*self.wrapper.flux:
            self.run(maxit=maxit)

#        while abs(np.sum(self.reco+self.dmap) - self.wrapper.flux) > tolerance * self.wrapper.flux:
#            self.run(maxit)
            
        self.update_weights(old_weights)
        self.solver.add_only_positive_coeff = add_only_positive_coeff

    #save output        
    def save(self, path):
        np.save(path+r'cmap.npy', self.cmap_list)
        np.save(path+r'reco.npy', self.reco)
        np.save(path+r'dmap.npy', self.dmap)
        np.save(path+r'window.npy', self.solver.window)
        
        out = self.finalize()
        out.save_fits(path+r'out.fits')
        
        #self.wrapper.Obsdata.save_uvfits(path+r'obs.fits')
    
    #Prepare selection of scales, needs to be run before using select_best_scale    
    def prepare_scale_selection(self):
        dirac = np.zeros(self.dmap.shape)
        dirac[self.dmap.shape[0]//2, self.dmap.shape[1]//2] = 1
        if self.transform == "DoG":
            res = self.solver.dog_trafo_clean.decompose(dirac)
        if self.transform == "Bessel":
            res = self.solver.trafo_clean.decompose(dirac)
            
        data = np.zeros((self.solver.length, len(self.wrapper.Obsdata.data)), dtype=complex)                    
        for i in range(self.solver.length):
            scale = self.wrapper.formatoutput(res[i][0])
            obs_scale = scale.observe_same(self.wrapper.Obsdata, ttype=self.wrapper.ttype, add_th_noise=False) 
            data[i] = np.array(obs_scale.unpack('amp'), dtype=float).flatten() / self.solver.normalization[i]
        
        self.most_sensitive_scale = np.argmax(data, axis=0)
        
        self.flag_scale_selection = np.ones(len(self.wrapper.Obsdata.data))
    
    #Select the scale that is most sensitive to the data point with largest discrepancy between observation and model   
    def select_best_scale(self):
        if self.obs_computed == False:
            out = self.wrapper.formatoutput(self.reco)
            self.test_obs = out.observe_same(self.wrapper.Obsdata, ttype=self.wrapper.ttype, add_th_noise=False)
        
        data = abs( np.array(self.test_obs.unpack('amp'), dtype=float).flatten() - self.wrapper.unpack('amp') ) * self.flag_scale_selection
        index = np.argmax( data )
        self.select_scales([self.most_sensitive_scale[index]])
        
        self.obs_computed = True
    
    #flag data point with largest discrepancy from this set    
    def flag_longest(self):
        if self.obs_computed == False:
            out = self.wrapper.formatoutput(self.reco)
            self.test_obs = out.observe_same(self.wrapper.Obsdata, ttype=self.wrapper.ttype, add_th_noise=False)
        
        data = abs( np.array(self.test_obs.unpack('amp'), dtype=float).flatten() - self.wrapper.unpack('amp') ) * self.flag_scale_selection
        index = np.argmax( data )
        
        self.flag_scale_selection[index] = 0
        
        self.obs_computed = True
    
    #flag data points corresponding to smoothing scale from this selection    
    def flag_smoothing_scale(self):
        for i in range(len(self.most_sensitive_scale)):
            if int(self.most_sensitive_scale[i]) == self.solver.length-1:
                self.flag_scale_selection[i] = 0
    
    #remove all flags            
    def unflag(self):
        self.flag_scale_selection = np.ones(len(self.flag_scale_selection))
    
    #update psf from input
    def update_psf(self, fwhm1, fwhm2, angle):
        psfprior = make_square(self.wrapper.Obsdata, self.wrapper.Prior.xdim//4, self.wrapper.Prior.fovx())
        psfprior = psfprior.add_gauss(1, (fwhm1, fwhm2, angle, 0, 0))
        
        psfprior = psfprior.regrid_image(self.wrapper.Prior.fovx(), self.wrapper.Prior.xdim)
        
        self.psf = psfprior.imarr()
        self.psf /= np.sum(self.psf)
        
        self.conv_psf = Convolution(Discretization(self.psf.shape), self.psf)
    
    #update contour levels    
    def update_levels(self, levels):
        self.levels = levels
     
    #helper for finalize_uniform    
    def _add_uniform_dmap(self, snr_cut):
        self.snr_cut = snr_cut
        obs_flagged = self.wrapper.Obsdata.flag_low_snr(snr_cut)
        
        if self.power == 0 and self.uweight == 1 and len(obs_flagged.data)==len(self.wrapper.Obsdata.data):
            self.dmap_uniform = self.dmap.copy()
            self.dbeam_uniform = self.dbeam.copy()
            self.conv_uniform = deepcopy(self.conv)
        else:
            obs_flagged.uvweight(power=0, weight=1)
            
            dmap, dbeam = obs_flagged.dirty_image_beam(self.wrapper.Prior.xdim, self.wrapper.Prior.fovx(), natural_correction=self.natural_correction)
            self.dmap_uniform = dmap.imarr().copy()
            self.dbeam_uniform = dbeam.imarr().copy()
            
            self.conv_uniform = Convolution(Discretization(self.dbeam_uniform.shape), self.dbeam_uniform)
    
    #alternative to finalize, but with the dirty map computed from uniform weighting, returns a perfect fit model 
    def finalize_uniform(self, snr_cut=3):
        if 'dmap_uniform' in dir(self) and 'dbeam_uniform' in dir(self) and self.update_uniform_dmap == False and 'snr_cut' in dir(self):
            if snr_cut == self.snr_cut:
                pass
            else:
                self._add_uniform_dmap(snr_cut)
        else:
            self._add_uniform_dmap(snr_cut)
        
        self.log.info("Set restored map")
        
        dmap = self.dmap_uniform - self.conv_uniform(self.reco)

        return self.wrapper.formatoutput(self.reco+dmap)
        
    def finalize_diff(self, snr_cut=3):
        empty = self.wrapper.Prior.copy()
        empty.imvec *= 0
        obs_diff = empty.observe_same(self.wrapper.Obsdata, ttype=self.wrapper.ttype, add_th_noise=False, sgrscat=False, ampcal=True, phasecal=True)
        
        out = self.wrapper.formatoutput(self.reco)
        obs_reco = out.observe_same(self.wrapper.Obsdata, ttype=self.wrapper.ttype, add_th_noise=False, sgrscat=False, ampcal=True, phasecal=True)
        
        amp_reco = obs_reco.unpack(["amp"])["amp"]
        amp_obs = self.wrapper.unpack("amp")
        sigma_obs = self.wrapper.unpack("sigma")
        
        #for i in range(len(self.wrapper.Obsdata.data)):
        #    if np.abs(amp_reco[i]-amp_obs[i]) > snr_cut * sigma_obs[i]:
        #        obs_diff.data[i] = self.wrapper.Obsdata.data[i]
                
        #obs_diff = EhtimObsdata(obs_diff)
        #obs_diff.uvweight(power=0, weight=1)
                
        #dmap, dbeam = obs_diff.dirty_image_beam(self.wrapper.Prior.xdim, self.wrapper.Prior.fovx())
        
        #dmap_uniform = dmap.imarr().copy()
        #dbeam_uniform = dbeam.imarr().copy()
            
        #conv_uniform = Convolution(Discretization(dbeam_uniform.shape), dbeam_uniform)
        
        #dmap = dmap_uniform - conv_uniform(self.reco)
        
        #return self.wrapper.formatoutput(self.reco+dmap)
        
        for i in range(len(self.wrapper.Obsdata.data)):
            if np.abs(amp_reco[i]-amp_obs[i]) > snr_cut * sigma_obs[i]:
                obs_diff.data[i] = self.wrapper.Obsdata.data[i]
            else:
                obs_diff.data[i] = obs_reco.data[i]
         
        obs_diff = EhtimObsdata(obs_diff)
        obs_diff.uvweight(power=0, weight=1)       
         
        dmap, dbeam = obs_diff.dirty_image_beam(self.wrapper.Prior.xdim, self.wrapper.Prior.fovx(), natural_correction=self.natural_correction)
        
        dmap_uniform = dmap.imarr().copy()
        dbeam_uniform = dbeam.imarr().copy()
            
        conv_uniform = Convolution(Discretization(dbeam_uniform.shape), dbeam_uniform)
        
        dmap = dmap_uniform - conv_uniform(self.reco)
        
        return self.wrapper.formatoutput(self.reco+dmap)
    
    #finalize imaging by CLEANing the residual with Hogbom CLEAN
    def finalize_clean(self, maxit=1000, gain=0.05, psf_from_prior=False):
        self.solver_hogbom = CLEAN(self.wrapper, 'Hogbom', self.widths, power=1, uweight=0, psf_from_prior=psf_from_prior, nsigma=10, sdmap=self.initial_dmap.copy(), sdbeam=self.dbeam)

        self.solver_hogbom.set_reco(self.reco.copy())        

        self.solver_hogbom.set_bounds_from_script(self.plotting_bounds)
        
        self.solver_hogbom.solver.gain = float(gain) or self.solver.gain
        self.solver_hogbom.solver.window = self.solver.window.copy()
        self.solver_hogbom.solver.indices = np.arange(0, np.prod(self.solver_hogbom.dmap.shape))[self.solver_hogbom.solver.window]
        
        self.solver_hogbom.run(maxit)
        
        model = self.solver_hogbom.finalize()
        
        res = self.solver_hogbom.reco - self.reco
        toret_reco = self.reco+self.solver_hogbom.conv_psf(res)
        toret_dmap = self.solver_hogbom.initial_dmap - self.solver_hogbom.conv(toret_reco)
        toret = self.wrapper.formatoutput(toret_reco+toret_dmap)
        
        return [model, toret]
    
    def add_missing_scales(self, factor=0.2, skip=[]):
        indices = self._merge(self.cmap_list)
        for i in range(len(self.solver.threshold)):
            if self.solver.threshold[i] == 0 and i not in skip:
                #identify scales to interpolate from
                lower = -1
                upper = -1
                for j in range(len(self.widths)):
                    if self.solver.threshold[i-j*(self.solver.ellipticities+1)] == 1 and i-j*(self.solver.ellipticities+1)>=0:
                        lower = i-j*(self.solver.ellipticities+1)
                        break
                
                for j in range(len(self.widths)):
                    if self.solver.threshold[i+j*(self.solver.ellipticities+1)] == 1:
                        upper = i+j*(self.solver.ellipticities+1)
                        break
                
                #interpolate mssing scales    
                if lower != -1 and upper != -1:
                    indices[i] = factor * (indices[lower]+indices[upper])/2
                    
                if lower == -1 and upper != -1:
                    indices[i] = factor * indices[upper]
                    
                if lower != -1 and upper == -1:
                    indices[i] = factor * indices[lower]
                    
        return self.merger(indices.flatten())
    
    def project_to_significant(self, res, level=0.01):
        peak = np.max(self.reco)
        imarr = np.where(self.reco < peak*level, 0, self.reco)
        toret = self.wrapper.formatoutput(imarr)
        toret = toret.blur_circ(res)
        self.set_reco(toret.imarr())
        
        
                
