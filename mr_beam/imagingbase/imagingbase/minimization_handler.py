import numpy as np
import ehtim as eh

from imagingbase.ehtim_wrapper import EhtimWrapper, EhtimFunctional, EhtimOperator
from regpy.operators import CoordinateProjection
from regpy.solvers import HilbertSpaceSetting
from imagingbase.solvers.forward_backward_splitting import Forward_Backward_Splitting
from imagingbase.solvers.gradient_descent import Gradient_Descent
from regpy.hilbert import L2
from imagingbase.regpy_functionals import L0, FunctionalProductSpace
from imagingbase.operators.msi import DOGDictionary

import regpy.stoprules as rules
from imagingbase.regpy_utils import Display, power

from MSI.Image import ConversionBase

import ehtplot.color

class MinimizationHandler:
    def __init__(self, psf_fwhm, npix, fov, obs_sc, prior, zbl, rescaling, data_term, cbar_lims, threshold, repair, to_cat, **kwargs):
        self.psf_fwhm = psf_fwhm
        self.npix = npix
        self.fov = fov
        self.init_fov = prior.fovx()
        self.obs_sc = obs_sc
        self.res = self.obs_sc.res()
        self.zbl = zbl
        self.rescaling = rescaling
        self.data_term = data_term
        
        self.cfun = 'afmhot_u'
        self.cbar_lims = cbar_lims
        
        self.img = prior.regrid_image(self.fov, self.npix+1)

        #Define domain
        self.convert = ConversionBase()
        self.grid = self.convert.find_domain_ehtim(self.img)
        
        self.widths = kwargs.get('widths', self.find_widths(threshold=threshold, repair=repair, to_cat=to_cat) )

        self.domain = power(self.grid, len(self.widths))
        
        self.debias = kwargs.get('debias', False)

        #Initialize wrapper objects to ehtim
        self.wrapper = EhtimWrapper(self.obs_sc.copy(), self.img.copy(), self.img.copy(), self.zbl,
                              d='vis', maxit=100, ttype='direct', clipfloor=-100,
                              rescaling=self.rescaling, debias=self.debias)
        
        self.wrapper_amp = EhtimWrapper(self.obs_sc.copy(), self.img.copy(), self.img.copy(), self.zbl,
                              d='amp', maxit=100, ttype='direct', clipfloor=-100,
                              rescaling=self.rescaling, debias=self.debias)
        
        self.wrapper_cphase = EhtimWrapper(self.obs_sc.copy(), self.img.copy(), self.img.copy(), self.zbl,
                              d='cphase', maxit=100, ttype='direct', clipfloor=-100,
                              rescaling=self.rescaling, debias=self.debias)
        
        self.wrapper_logcamp = EhtimWrapper(self.obs_sc.copy(), self.img.copy(), self.img.copy(), self.zbl,
                              d='logcamp', maxit=100, ttype='direct', clipfloor=-100,
                              rescaling=self.rescaling, debias=self.debias)
 
        #Define forward operator 
        self.op = DOGDictionary(self.domain, self.grid, self.widths, self.grid.shape)
        #self.dog_trafo = DOGTransform(self.grid, self.widths)
        self.forward = EhtimOperator(self.wrapper, self.grid)
        
        self.setting_forward = HilbertSpaceSetting(op=self.forward, Hdomain=L2, Hcodomain=L2)
        
        #Define Hilbert Space Setting for full operator
        self.setting = HilbertSpaceSetting(op=self.op, Hdomain=L2, Hcodomain=L2)
        
        #Define data fidelity term
        self.data_fidelity_vis = EhtimFunctional(self.wrapper, self.grid) * self.op
        self.data_fidelity_amp = EhtimFunctional(self.wrapper_amp, self.grid) * self.op
        self.data_fidelity_cphase = EhtimFunctional(self.wrapper_cphase, self.grid) * self.op
        self.data_fidelity_logcamp = EhtimFunctional(self.wrapper_logcamp, self.grid) * self.op
               
        self.data_fidelity_closure = data_term['cphase'] * self.data_fidelity_cphase \
                + data_term['logcamp'] * self.data_fidelity_logcamp    
                
        #Define penalty term
        self.weights = np.ones(len(self.widths))
        self.funcs_l0 = []
        for i in range(len(self.widths)):
            self.funcs_l0.append( L0(self.setting.Hdomain.discr.summands[i]) )
            self.weights[i] = np.max(self.op.dogs[i])
        self.penalty_l0 = FunctionalProductSpace(self.funcs_l0, self.setting.Hdomain.discr, self.weights)
        
    def find_widths(self, threshold = 0.1*10**9, repair=[39.98, 47.24], to_cat=[1, 2, 4]):
        uvdist = np.array(self.obs_sc.unpack(['uvdist']), dtype=float)

        uvdist = np.sort(uvdist)
        
        jumps = np.zeros(len(uvdist)-1)
        
        for i in range(len(uvdist)-1):
            jumps[i] = uvdist[i+1]-uvdist[i]
                   
        indices = [jumps[i] > threshold for i in range(len(jumps))]
        
        indices.append(False)
                
        cuts_lower = uvdist[indices]
        cuts_upper = uvdist[1:][indices[:-1]]
        
        cuts = np.zeros(len(cuts_upper)-1)
        for i in range(len(cuts_upper)-1):
            cuts[i] = (cuts_lower[i+1]+cuts_upper[i+1])/2
        
        sigma = 1/(2*np.pi*cuts) 
        
        fwhm = 2.355 * sigma
        
        widths = fwhm / self.img.psize
        widths = widths[::-1]
        #widths = widths[:-1]
        to_cat = np.array(to_cat)
        widths = np.concatenate((to_cat, widths, np.array(repair)))
        
        return widths

    def find_initial_guess(self, data, alpha, reverse=True):
        init = np.zeros((len(self.widths), self.npix+1, self.npix+1))
        self.decision = np.zeros((len(self.widths), 10))

        guess = init.copy()

        self.noise_levels = np.ones(len(self.widths))
        
        for j in range(10):
            threshold = 0.5*j
            for i in range(len(self.widths)):
                guess[::-1][i] = data
                test = self.penalty_l0.proximal(guess.flatten(), threshold * 1/self.weights)
                self.decision[i, j] = self.data_fidelity_closure(test) + alpha * self.penalty_l0(test)
            guess *= 0
        
        nr_scales = np.unravel_index(np.argmin(np.nan_to_num(self.decision, nan=np.inf)), self.decision.shape)
        for i in range(nr_scales[0]+1):
            init[::-1][i] = data
        threshold = 0.5*nr_scales[1]    
        print("We use ", nr_scales[0], "scales with threshold", nr_scales[1]*0.5)
        

#        self.minimum = self.decision[nr_scales[0], nr_scales[1]]
#        #indices = []
#        self.thresh = threshold*1/self.weights
#        for i in range(nr_scales[0]+1-1):
#            index = i + len(self.widths) - nr_scales[0] -1
#            test_thresh = self.thresh.copy()
#            for j in range(20):
#                test_thresh[index] = 0.5*j/self.weights[index]
#                test = self.penalty_l0.proximal(init.flatten(), test_thresh)
#                objective = self.data_fidelity_closure(test) + alpha * self.penalty_l0(test)
#                if objective < self.minimum:
#                    self.minimum = objective
#                    self.thresh[index] = test_thresh[index]
         
        for i in range(len(self.widths)):
            init[i] = data
        
        self.minimum = self.decision[nr_scales[0], nr_scales[1]]
        #indices = []
        self.thresh = threshold*1/self.weights
        
        for i in range(len(self.widths) - nr_scales[0] - 1):
            self.thresh[i] = 10000*1/self.weights[i]
            
        print("Current minimum", self.minimum)
        print("Start optimization with thresholds", self.thresh*self.weights)
        print("reverse mode:", reverse)
        
        if reverse:
            for i in range(len(self.widths)):
                test_thresh = self.thresh.copy()
                for j in range(20):
                    test_thresh[i] = 0.5*j/self.weights[i]
                    test = self.penalty_l0.proximal(init.flatten(), test_thresh)
                    objective = self.data_fidelity_closure(test) + alpha * self.penalty_l0(test)
                    if objective < self.minimum and np.isfinite(objective):
                        self.minimum = objective
                        self.thresh[i] = test_thresh[i]
                        print("Updated at scale", i, "to threshold", j*0.5, "to minimum", self.minimum)
                        
        else:
            for i in range(len(self.widths)):
                test_thresh = self.thresh.copy()
                for j in range(20):
                    test_thresh[len(self.widths)-1-i] = 0.5*j/self.weights[len(self.widths)-1-i]
                    test = self.penalty_l0.proximal(init.flatten(), test_thresh)
                    objective = self.data_fidelity_closure(test) + alpha * self.penalty_l0(test)
                    if objective < self.minimum and np.isfinite(objective):
                        self.minimum = objective
                        self.thresh[len(self.widths)-i-1] = test_thresh[len(self.widths)-i-1]
                        print("Updated at scale", len(self.widths)-1-i, "to threshold", j*0.5, "to minimum", self.minimum)
                
        init = self.penalty_l0.proximal(init.flatten(), self.thresh)
         
        print("We use thresholds: ", self.thresh*self.weights)
            
        return init   
    
    def _converge(self, major=3, blur_frac=1):
        for repeat in range(major):
            init = self.imgr.out_last().blur_circ(blur_frac*self.res)
            self.imgr.init_next = init
            self.imgr.make_image_I(show_updates=False)
    
    def first_round(self, init, data_term, cycles=10, maxit=50, epochs=10, sys_noise=0.02, uv_zblcut=0.1*10**9, stop=1e-4, last_epochs=False):
        # constant regularization weights
        reg_term  = {'simple' : 0*100,    # Maximum-Entropy
                     'tv'     : 0*1.0,    # Total Variation
                     'tv2'    : 0*1.0,    # Total Squared Variation
                     'l1'     : 0.0,    # L1 sparsity prior
                     'flux'   : 1e4}    # compact flux constraint
        
        # Initialize imaging with a Gaussian image
        self.imgr = eh.imager.Imager(self.obs_sc, init, prior_im=init,
                            flux=self.zbl, data_term=data_term, maxit=maxit,
                            norm_reg=True, systematic_noise=sys_noise,
                            reg_term=reg_term, ttype=self.wrapper.ttype, cp_uv_min=uv_zblcut, stop=stop)
        
        # Imaging
        self.imgr.make_image_I(show_updates=False)
        for i in range(cycles):
            self._converge(epochs, 0)
            self._converge(1, 1)
         
        if last_epochs:    
            self._converge(epochs, 0)
        return self.imgr.out_last()
    
  
    def second_round(self, init, tau, alpha, max_iterations=25, maxiter=20, display=False):
        #Define solver
        solver = Forward_Backward_Splitting(self.setting, self.data_fidelity_closure, self.penalty_l0, init.copy(), tau = tau, regpar = alpha)
        #solver.callback = lambda x: wrapper._plotcur(dog_dictionary(x))
        stoprule = rules.CombineRules(
            [rules.CountIterations(max_iterations=max_iterations),
            Display(self.penalty_l0, 'Penalty')]
        )
        
        #Run solver
        for i in range(maxiter):
            stoprule.rules[0].iteration = 0
            stoprule.rules[0].triggered = False
            stoprule.triggered = False
            
            wtfcts, reco_data = solver.run(stoprule)
            
            reco = self.op(wtfcts)
            current_flux = np.sum(reco)
            wtfcts *= self.wrapper.flux/current_flux
            reco *= self.wrapper.flux/current_flux
            
            solver.x = wtfcts
            
            print(current_flux)
            
            print('Data Fidelity: ', self.data_fidelity_closure(solver.x))
            
            if display:
                img = self.wrapper.formatoutput(reco)
                img = img.regrid_image(self.init_fov, self.npix)
                img.display(cbar_unit=['Tb'], cfun=self.cfun, cbar_lims=self.cbar_lims)

        return wtfcts
    
    def second_round_minimization(self, init, alpha=0.1, tau=10, reverse=True, **kwargs):
        img = init.regrid_image(self.fov, self.npix+1)

        #Find initial guess by expressing img by dictionary
        #Decomposition computed by Landweber inversion
        data = self.convert.ehtim_to_numpy(img)/self.rescaling
               
        init = self.find_initial_guess(data, alpha, reverse=reverse)
        
        wtfcts = self.second_round(init, tau, alpha*self.noise_levels, **kwargs)
         
        self.find_projection(wtfcts)
        
        reco = self.op(wtfcts)
        
        img = self.wrapper.formatoutput(reco)

        return [wtfcts, img]
    
    def third_round(self, init, data_term, tau=10, **kwargs):
        data_fidelity_all = data_term['amp'] * self.data_fidelity_amp \
                + data_term['cphase'] * self.data_fidelity_cphase \
                + data_term['logcamp'] * self.data_fidelity_logcamp 

        data_fidelity_scales = data_fidelity_all * self.coordinate_proj.adjoint
        
        init = self.coordinate_proj(init.copy())
        
        #test stepsize
        start_fidelity = data_fidelity_scales(init)
        
        test_wtfcts = self.grad_desc(init.copy(), tau, data_fidelity_scales, self.setting_msi, maxit=5)
        test_fidelity = data_fidelity_scales(test_wtfcts)
        
        divergence = False
        if test_fidelity < start_fidelity:
           for i in range(3):
               test_wtfcts = self.grad_desc(test_wtfcts.copy(), tau, data_fidelity_scales, self.setting_msi, maxit=5)
               test_fidelity_new = data_fidelity_scales(test_wtfcts)
               if test_fidelity_new < test_fidelity:
                   test_fidelity = test_fidelity_new
                   divergence = False
               else:
                   divergence = True
                   break  
               
        while start_fidelity < test_fidelity or divergence:
            print('Stepsize to big, try to run with smaller stepsize')
            tau /= 2
            test_wtfcts = self.grad_desc(init.copy(), tau, data_fidelity_scales, self.setting_msi, maxit=5)
            test_fidelity = data_fidelity_scales(test_wtfcts)
            if test_fidelity < start_fidelity:
                for i in range(3):
                    test_wtfcts = self.grad_desc(test_wtfcts.copy(), tau, data_fidelity_scales, self.setting_msi, maxit=5)
                    test_fidelity_new = data_fidelity_scales(test_wtfcts)
                    if test_fidelity_new < test_fidelity:
                        test_fidelity = test_fidelity_new
                        divergence = False
                    else:
                        divergence = True
                        break
            
        wtfcts = self.grad_desc(init.copy(), tau, data_fidelity_scales, self.setting_msi, **kwargs)
        
        wtfcts = self.coordinate_proj.adjoint(wtfcts)
        reco = self.op(wtfcts)
        
        img = self.wrapper.formatoutput(reco)
        
        return [wtfcts, img, tau]
    
    def fourth_round(self, init, data_term, tau=10, **kwargs):
        #Visibility data fidelity operator
        data_fidelity_all = data_term['vis'] * self.data_fidelity_vis \
                + data_term['cphase'] * self.data_fidelity_cphase \
                + data_term['logcamp'] * self.data_fidelity_logcamp 
                
        data_fidelity_scales = data_fidelity_all * self.coordinate_proj.adjoint        

        init = self.coordinate_proj(init.copy())
        
        #test stepsize
        start_fidelity = data_fidelity_scales(init)
        
        test_wtfcts = self.grad_desc(init.copy(), tau, data_fidelity_scales, self.setting_msi, maxit=5)
        test_fidelity = data_fidelity_scales(test_wtfcts)
                
        divergence = False
        if test_fidelity < start_fidelity:
           for i in range(3):
               test_wtfcts = self.grad_desc(test_wtfcts.copy(), tau, data_fidelity_scales, self.setting_msi, maxit=5)
               test_fidelity_new = data_fidelity_scales(test_wtfcts)
               if test_fidelity_new < test_fidelity:
                   test_fidelity = test_fidelity_new
                   divergence = False
               else:
                   divergence = True
                   break  
               
        while start_fidelity < test_fidelity or divergence:
            print('Stepsize to big, try to run with smaller stepsize')
            tau /= 2
            test_wtfcts = self.grad_desc(init.copy(), tau, data_fidelity_scales, self.setting_msi, maxit=5)
            test_fidelity = data_fidelity_scales(test_wtfcts)
            if test_fidelity < start_fidelity:
                for i in range(3):
                    test_wtfcts = self.grad_desc(test_wtfcts.copy(), tau, data_fidelity_scales, self.setting_msi, maxit=5)
                    test_fidelity_new = data_fidelity_scales(test_wtfcts)
                    if test_fidelity_new < test_fidelity:
                        test_fidelity = test_fidelity_new
                        divergence = False
                    else:
                        divergence = True
                        break
        
        coeff = self.grad_desc(init, tau, data_fidelity_scales, self.setting_msi, **kwargs)
        
        coeff = self.coordinate_proj.adjoint(coeff)
        reco = self.op(coeff)
        
        img = self.wrapper.formatoutput(reco)
        
        return [coeff, img, tau]
    
    def fifth_round(self, init, data_term, tau=0.1, **kwargs):
        #Visibility data fidelity operator
        data_fidelity_all = data_term['vis'] * EhtimFunctional(self.wrapper, self.grid) \
                + data_term['cphase'] * EhtimFunctional(self.wrapper_cphase, self.grid) \
                + data_term['logcamp'] * EhtimFunctional(self.wrapper_logcamp, self.grid)
             
        #test stepsize
        start_fidelity = data_fidelity_all(init)
        
        test_wtfcts = self.grad_desc(init.copy(), tau, data_fidelity_all, self.setting_forward, maxit=5)
        test_fidelity = data_fidelity_all(test_wtfcts)
        
        divergence = False
        if test_fidelity < start_fidelity:
           for i in range(3):
               test_wtfcts = self.grad_desc(test_wtfcts.copy(), tau, data_fidelity_all, self.setting_forward, maxit=5)
               test_fidelity_new = data_fidelity_all(test_wtfcts)
               if test_fidelity_new < test_fidelity:
                   test_fidelity = test_fidelity_new
                   divergence = False
               else:
                   divergence = True
                   break          
        
        while start_fidelity < test_fidelity or divergence:
            print('Stepsize to big, try to run with smaller stepsize')
            tau /= 2
            test_wtfcts = self.grad_desc(init.copy(), tau, data_fidelity_all, self.setting_forward, maxit=5)
            test_fidelity = data_fidelity_all(test_wtfcts)
            if test_fidelity < start_fidelity:
                for i in range(3):
                    test_wtfcts = self.grad_desc(test_wtfcts.copy(), tau, data_fidelity_all, self.setting_forward, maxit=5)
                    test_fidelity_new = data_fidelity_all(test_wtfcts)
                    if test_fidelity_new < test_fidelity:
                        test_fidelity = test_fidelity_new
                        divergence = False
                    else:
                        divergence = True
                        break                
             
        reco = self.grad_desc(init, tau, data_fidelity_all, self.setting_forward, **kwargs)
                
        img = self.wrapper.formatoutput(reco)
        
        return img
    
    def find_projection(self, wtfcts):
        self.mask = [wtfcts[i] != 0 for i in range(len(wtfcts))]
        self.mask = np.asarray(self.mask, dtype=bool)
        self.coordinate_proj = CoordinateProjection(self.domain, self.mask)
        
        #data_fidelity_closure_msi = data_fidelity_closure * coordinate_proj.adjoint
        op_msi = self.op * self.coordinate_proj.adjoint
        self.setting_msi = HilbertSpaceSetting(op=op_msi, Hdomain=L2, Hcodomain=L2)
    
    def updateobs(self, obs_sc):
        self.obs_sc = obs_sc.copy()
        
        self.wrapper.updateobs(obs_sc.copy())
        self.wrapper_amp.updateobs(obs_sc.copy())
        self.wrapper_logcamp.updateobs(obs_sc.copy())
        self.wrapper_cphase.updateobs(obs_sc.copy())
        
        self.data_fidelity_vis = EhtimFunctional(self.wrapper, self.grid) * self.op
        self.data_fidelity_amp = EhtimFunctional(self.wrapper_amp, self.grid) * self.op
        self.data_fidelity_cphase = EhtimFunctional(self.wrapper_cphase, self.grid) * self.op
        self.data_fidelity_logcamp = EhtimFunctional(self.wrapper_logcamp, self.grid) * self.op

    def grad_desc(self, init, tau, data_fidelity, setting, maxit=1000, stop=0, display=True):
        
        #Define solver
        solver = Gradient_Descent(setting, data_fidelity, init, tau=tau)
        #Graphical output in every iteration
        #solver.callback = lambda x: wrapper._plotcur(dog_dictionary(shift(x)))
        #Define stopping rule
        if stop == 0:
            if display: 
                stoprule = rules.CombineRules(
                    [rules.CountIterations(max_iterations=maxit),
                    Display(data_fidelity, 'Data Fidelity')
                    ]
                )
    
            else:
                stoprule = rules.CombineRules(
                    [rules.CountIterations(max_iterations=maxit)
                    ]
                )

            
        else:
            stoprule = rules.CombineRules(
                [rules.CountIterations(max_iterations=maxit),
                 rules.DiscrepancyFunctional(data_fidelity, stop)
                ]
            )
        
        #Run solver
        wtfcts, reco_data = solver.run(stoprule)
        
        return wtfcts
    
    def update_grid(self, img):
        old_psize = self.fov / self.npix
        
        self.npix = img.xdim
        self.fov = img.fovx()
        self.init_fov = img.fovx()
        
        self.img = img

        #Define domain
        self.convert = ConversionBase()
        self.grid = self.convert.find_domain_ehtim(img)


        self.widths *= old_psize * self.npix/self.fov

        self.domain = power(self.grid, len(self.widths))
        
        #Initialize wrapper objects to ehtim
        self.wrapper = EhtimWrapper(self.obs_sc.copy(), self.img.copy(), self.img.copy(), self.zbl,
                              d='vis', maxit=100, ttype='direct', clipfloor=-100,
                              rescaling=self.rescaling, debias=self.debias)
        
        self.wrapper_amp = EhtimWrapper(self.obs_sc.copy(), self.img.copy(), self.img.copy(), self.zbl,
                              d='amp', maxit=100, ttype='direct', clipfloor=-100,
                              rescaling=self.rescaling, debias=self.debias)
        
        self.wrapper_cphase = EhtimWrapper(self.obs_sc.copy(), self.img.copy(), self.img.copy(), self.zbl,
                              d='cphase', maxit=100, ttype='direct', clipfloor=-100,
                              rescaling=self.rescaling, debias=self.debias)
        
        self.wrapper_logcamp = EhtimWrapper(self.obs_sc.copy(), self.img.copy(), self.img.copy(), self.zbl,
                              d='logcamp', maxit=100, ttype='direct', clipfloor=-100,
                              rescaling=self.rescaling, debias=self.debias)
 
        #Define forward operator 
        self.op = DOGDictionary(self.domain, self.grid, self.widths, self.grid.shape)
        #self.dog_trafo = DOGTransform(self.grid, self.widths)
        self.forward = EhtimOperator(self.wrapper, self.grid)
        
        #Define Hilbert Space Setting for full operator
        self.setting = HilbertSpaceSetting(op=self.op, Hdomain=L2, Hcodomain=L2)
        
        #Define data fidelity term
        self.data_fidelity_vis = EhtimFunctional(self.wrapper, self.grid) * self.op
        self.data_fidelity_amp = EhtimFunctional(self.wrapper_amp, self.grid) * self.op       
        self.data_fidelity_cphase = EhtimFunctional(self.wrapper_cphase, self.grid) * self.op
        self.data_fidelity_logcamp = EhtimFunctional(self.wrapper_logcamp, self.grid) * self.op
               
        self.data_fidelity_closure = self.data_term['cphase'] * self.data_fidelity_cphase \
                + self.data_term['logcamp'] * self.data_fidelity_logcamp      
                
        #Define penalty term
        self.weights = np.ones(len(self.widths))
        self.funcs_l0 = []
        for i in range(len(self.widths)):
            self.funcs_l0.append( L0(self.setting.Hdomain.discr.summands[i]) )
            self.weights[i] = np.max(self.op.dogs[i])
        self.penalty_l0 = FunctionalProductSpace(self.funcs_l0, self.setting.Hdomain.discr, self.weights)
                