import ehtim as eh
import numpy as np
from scipy.optimize import minimize
from itertools import permutations
import itertools
import multiprocessing
import math

from pyswarm import pso
import pyswarms as ps
from pyswarms.backend import topology

from imagingbase.solvers.gradient_descent import Gradient_Descent


class CooperativeGame():

    def __init__(self, data, udp, fit, solver, scipy_option, prior, use_gradient, mode='shapley', res=0, epsilon=1e-5, ideal=None):
        self.data = data
        self.udp = udp
        self.fit = fit
        self.max_weight = self.data["max_weight"]
        self.snr = self.data['snr']
        self.num_players = udp.get_nobj() # - 1 if shapley
        self.scipy_option = scipy_option
        self.shp_val = np.zeros(self.num_players)
        self.lower_bounds = np.ones(self.num_players) * data['lower_bounds']
        self.upper_bounds = np.ones(self.num_players) * data['upper_bounds']
        self.solver = solver
        self.population_size = self.data['neighbours']
        self.num_iterations = self.data['generations']
        self.num_cores = self.data['num_cores']
        self.minimization_algorithm = self.data['minimization_algorithm']
        self.use_gradient = use_gradient
        #self.data['shapley_image'] = np.ones(self.npix)
        self.mode = mode
        self.res = res
        self.epsilon = epsilon
        
        self.logweights = self.data['logweights']
        self.logim = self.data['logim']
        self.prior = prior
        
        assert ((self.fit.get_name() == 'EHT' and self.logim) or self.logim == False)
                
        #self.dirtyimage = eh.obsdata.load_uvfits(self.data['uvf']).dirtyimage(int(np.sqrt(self.npix)), self.shapley_image.fovx())
        if type(prior) is eh.image.Image:
            self.keyframes = 1
            if self.fit.get_name() == 'Polarization' or self.fit.get_name() == 'CLTrace':
                self.x0 = np.concatenate([self.prior.mvec / self.data['rescaling'], self.prior.chivec / (2*np.pi)]).flatten()
            elif self.fit.get_name() == 'FullStokes':
                self.x0 = np.concatenate([self.prior.mvec / self.data['rescaling'], self.prior.chivec / (2*np.pi), 0.5*np.ones(len(self.prior.chivec))]).flatten()
            elif self.fit.get_name() == 'Scattering':
                self.x0 = np.zeros(2*self.prior.xdim*self.prior.ydim-1)
                self.x0[0:self.prior.xdim*self.prior.ydim] = self.prior.imvec / self.data['rescaling']
            else:
                self.x0 = self.prior.imvec / self.data['rescaling']
            self.image_dummy = self.prior.copy()
            if self.logim:
                self.x0 = np.log(self.x0)
        else:
            self.keyframes = len(self.prior)
            if self.logim:
                self.x0 = np.array([np.log(self.prior[i].imvec / self.data['rescaling']) for i in range(self.keyframes)]).flatten()
            else:
                if self.fit.get_name() == 'Polarization' or self.fit.get_name() == 'CLTrace':
                    assert self.keyframes == 1
                    self.x0 = np.concatenate([self.prior[0].mvec / self.data['rescaling'], self.prior[0].chivec / (2*np.pi)]).flatten()
                elif self.fit.get_name() == 'FullStokes':
                    assert self.keyframes == 1
                    self.x0 = np.concatenate([self.prior[0].mvec / self.data['rescaling'], self.prior[0].chivec / (2*np.pi), 0.5*np.ones(len(self.prior[0].chivec))]).flatten()
                else:    
                    self.x0 = np.array([self.prior[i].imvec / self.data['rescaling'] for i in range(self.keyframes)]).flatten()
            self.image_dummy = self.prior[0].copy()  
            
        self.inits = []    
        for i in range(self.num_players):
            self.inits.append(self.x0.copy())
            
        self.shapley_image = self.image_dummy.copy() #eh.image.load_fits(self.data['img'])
        self.npix = self.shapley_image.xdim**2
        if self.fit.get_name() == 'Polarization' or self.fit.get_name() == 'CLTrace':
            self.npix *= 2
        if self.fit.get_name() == 'FullStokes':
            self.npix *= 3
        if self.fit.get_name() == 'Scattering':
            self.npix = 2*self.npix-1
       
        #Find ideal point first
        if isinstance(ideal, np.ndarray):
            assert len(ideal.shape) == self.udp.get_nobj(), 'Number of objective dont match number ideals'
            self.ideal = ideal
        else:
            self.ideal = self.get_ideal()
            self.ideal = np.diagonal(self.ideal)
        self.axis_scaling = np.max(np.abs(self.ideal), axis=0)

        ##self.ideal = np.abs(self.ideal)
        
        self.optimizer = None
       
        
        
    def _execute_optimize_shapley(self, args):
            swarm_copy, i = args
            # swarm_copy = swarm.copy()
            swarm_copy[i+1] = 0.0
            swarm_copy[i] = -3.0
            return self._optimize_shapley(swarm_copy, 0)
        
        
    def _execute_optimize_shapley_nolog(self, args):
            swarm_copy, i = args
            swarm_copy *= 0.0
            swarm_copy[i+1] = 1.0
            #print(swarm_copy, i, args, self.num_players)
            return self._optimize_shapley(swarm_copy, 0)
        
    #def fitness(self, x):
    #    return [self.fit.ob1(x), self.fit.ob2(x), self.fit.ob3(x), self.fit.ob4(x), self.fit.ob5(x), self.fit.ob6(x), self.fit.ob7(x)] #remove ob7 for shapley
    
    #def gradient(self, x):
    #    return np.concatenate(( self.fit.ob1.gradient(x), self.fit.ob2.gradient(x), self.fit.ob3.gradient(x), 
    #                          self.fit.ob4.gradient(x), self.fit.ob5.gradient(x), self.fit.ob6.gradient(x), self.fit.ob7.gradient(x))) #same with ob7 

                    
    def get_ideal(self):
        
        pool = multiprocessing.Pool(self.num_cores)#(self.num_players-1)
        results = []
        
        ideal = np.zeros((self.num_players, self.num_players)) #self.ideal.copy()
        
        if self.logweights:
            swarm = np.zeros(self.num_players)#-3.
            swarm[0] = 1.0
            result = self._optimize_shapley(swarm, 0)
            if self.logim:
                ideal[0] = self.udp.fitness(np.exp(result.x))
            else:
                ideal[0] = self.udp.fitness(result.x)

            results = pool.map(self._execute_optimize_shapley_nolog, [(swarm, i) for i in range(self.num_players - 1)])
                
            pool.close()
            pool.join()
            
            for i, result in enumerate(results):
                if self.logim:
                    ideal[i+1] = self.udp.fitness(np.exp(result.x))
                else:
                    ideal[i+1] = self.udp.fitness(result.x)
        else:
            swarm = np.zeros(self.num_players)
            swarm[0] = 1.0
            result = self._optimize_shapley(swarm, 0)
            if self.logim:
                ideal[0] = self.udp.fitness(np.exp(result.x))
            else:
                ideal[0] = self.udp.fitness(result.x)

            results = pool.map(self._execute_optimize_shapley_nolog, [(swarm, i) for i in range(self.num_players - 1)])
            
            pool.close()
            pool.join()
            

            for i, result in enumerate(results):
                if self.logim:
                    ideal[i+1] = self.udp.fitness(np.exp(result.x))
                else:
                    ideal[i+1] = self.udp.fitness(result.x)
        
        #if shapley only the line before, for the ideal way, all line:   
        if self.mode == 'pareto':
            ideal = np.transpose(ideal.transpose()[:-1]-ideal[:-1,-1])
        #self.ideal[:,-1] = ideal[:,-1]
        #print(self.ideal, ideal)
        #self.ideal = np.abs(self.ideal)
        return ideal            
        
        
    def update_shapley(self, x):
        #self.data["shapley_values"] = x
        self.shp_val = x

    #Fitness function, distance to ideal point for every weight combination
    def fitness_function(self, swarm, funs):
        funs = []
        distances = []
        result = []
        for i in range(len(swarm)):
            particle = swarm[i]
            
            if self.logweights:
                result.append(self._optimize_shapley(10**particle, i, compute_shapley=False))
            else:
                result.append(self._optimize_shapley(particle, i, compute_shapley=False))
                
            #for j in range(self.keyframes):
                #if self.logim:
                    #self.image_dummy.imvec = np.exp(result.x.copy()[j*self.npix:(j+1)*self.npix])
                    #self.inits[i][j*self.npix:(j+1)*self.npix] = np.log(self.image_dummy.blur_circ(self.res).imvec)
                #else:
                    #self.image_dummy.imvec = result.x.copy()[j*self.npix:(j+1)*self.npix]
                    #self.inits[i][j*self.npix:(j+1)*self.npix] = self.image_dummy.blur_circ(self.res).imvec
            
        

        if self.logim:
            for res in result:
                evals = self.udp.fitness(res.x)
                distances.append(self._compute_distance(evals))
        else:
            for res in result:
                evals = self.udp.fitness(res.x)
                distances.append(self._compute_distance(evals))
    
        return distances
        
    def _compute_distance(self, evals):
        if self.mode == 'pareto':
            shapley_evals = evals-evals[-1]
            shapley_evals[-1] = evals[-1]
        else:
            shapley_evals = evals.copy()
        if self.fit.get_name() == 'Polarization' or self.fit.get_name() == 'CLTrace' or self.fit.get_name() == 'FullStokes':
            return np.linalg.norm((shapley_evals[1:]-self.ideal[1:])/self.axis_scaling[1:])
        return np.linalg.norm((shapley_evals-self.ideal)/self.axis_scaling)
        
    
    def _chisq(self, x):
        if self.logim:
            return self.fit.ob7(np.exp(x)) - 1.0
        else:
            return self.fit.ob7(x) - 1.0
        
    def _optimize_shapley(self, x, i, compute_shapley=False):
        
        if compute_shapley:
            print("ERROR!!! YOU SHOULDN'T BE HERE!!!!! (YET)")
            shapley_values = self.calculate_shapley_values(x, self.num_players)
            self.normalized_weights = shapley_values / np.sum(shapley_values)
         
        else: 
            self.normalized_weights = x / np.sum(x)
            if self.fit.get_name() == 'Polarization' or self.fit.get_name() == 'CLTrace' or self.fit.get_name() == 'FullStokes':
                self.normalized_weights = self.normalized_weights / np.sum(self.normalized_weights[1:])
            
        if self.data['parallel'] == True:
            x0 = self.inits[0].copy() #inits has dimen. of players, but swarm dimension can be greater
        else:
            x0 = self.image_dummy #np.random.rand(self.npix)#self.inits[0].copy()

        if self.use_gradient:
            jac = self.objective_grad
        else:
            jac = None
            
        if self.logim and (self.fit.get_name() == 'EHT' or self.fit.get_name() == 'Movie'):
            bounds = [(None, None)]*self.npix*self.keyframes
        elif self.logim == False and (self.fit.get_name() == 'EHT' or self.fit.get_name() == 'Movie'):
            bounds = [(self.epsilon,None)]*self.npix*self.keyframes
        elif self.fit.get_name() == 'Polarization' or self.fit.get_name() == 'CLTrace':
            bounds = [(0.001,0.999)]*(self.npix//2) + [(None,None)]*(self.npix//2)
        elif self.fit.get_name() == 'FullStokes':
            bounds = [(0.001,0.999)]*(self.npix//3) + [(None,None)]*(self.npix//3) + [(0.001,0.999)]*(self.npix//3)
        elif self.fit.get_name() == 'Scattering':
            assert(self.logim == False)
            bounds = ([(self.epsilon,None)]*((self.npix+1)//2)+[(None,None)]*((self.npix+1)//2-1))*self.keyframes
        else:
            bounds = [(None, None)]*self.npix*self.keyframes

        #Minimization by SLSQP
        if compute_shapley:
            constraints = {'type': 'eq', 'fun': self._chisq}
            method = 'SLSQP'
            result = minimize(self.objective, x0, jac=jac, method=self.minimization_algorithm, options=self.scipy_option, bounds=bounds, constraints=[constraints])
        else:
            if self.minimization_algorithm == 'gradient':
                x = x0.copy()
                for i in range(self.scipy_option['maxiter']):
                   x -= self.scipy_option['tau'] * self.objective_grad(x) 
                class results():
                    def __init__(self):
                        return
                result = results()
                result.x = x
            else:
                result = minimize(self.objective, x0, jac=jac, method=self.minimization_algorithm, options=self.scipy_option, bounds=bounds)
            #print(self.normalized_weights, self.objective(x0), np.max(result.x))

        return result
    
    def objective(self, imvec):
        if self.logim:
            imvec = np.exp(imvec)
        #This is a workaround to avoid these infinity values
        if self.fit.get_name() == 'Polarization' or self.fit.get_name() == 'CLTrace' or self.fit.get_name() == 'FullStokes':
            return np.dot(self.normalized_weights[1:], self.udp.fitness(imvec)[1:])
        return np.dot(self.normalized_weights, self.udp.fitness(imvec))
            
    def objective_grad(self, imvec): # to delete if works, just to keep the one working without being touched
        if self.logim:
            mat = np.exp(imvec)*self.gradient(np.exp(imvec)).reshape(self.num_players,self.npix * self.keyframes)
            return np.nan_to_num(np.array([mat[i] * self.normalized_weights[i] for i in range(self.num_players)]).sum(axis=0)).flatten()
        else:
            mat = self.udp.gradient(imvec).reshape(self.num_players,self.npix * self.keyframes)
            if self.fit.get_name() == 'Polarization' or self.fit.get_name() == 'CLTrace' or self.fit.get_name() == 'FullStokes':
                return np.nan_to_num(np.array([mat[i+1] * self.normalized_weights[i+1] for i in range(self.num_players-1)]).sum(axis=0)).flatten() 
            return np.nan_to_num(np.array([mat[i] * self.normalized_weights[i] for i in range(self.num_players)]).sum(axis=0)).flatten() 
    
    def cooperative_particles(self):
        if self.solver == 'pyswarms':
            # Call instance of PSO
            print("--------------------- PYSWARMS selected: GlobalBestPSO used")
            options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
            #options = {'c1': 2, 'c2': 2, 'w': 0.9}
            #options = {'c1': 2, 'c2': 0.1, 'w': 0.5}
            
            optimizer = ps.single.GlobalBestPSO(n_particles=self.population_size, dimensions=self.num_players, options=options, bounds=(self.lower_bounds, self.upper_bounds))
    
            kwargs={"funs": []}
            # Perform optimization
            best_fitness, best_position = optimizer.optimize(self.fitness_function, iters=self.num_iterations, n_processes=self.num_cores, **kwargs)
            
            self.optimizer = optimizer
            
            result = self._optimize_shapley(best_position, 0, compute_shapley=False)
            self.image_dummy = result.x
            
            return best_fitness, best_position
            
            
        elif self.solver == 'pyswarm':
            # Use PSO to optimize the fitness function
            print("--------------------- PYSWARM selected: pso used")
            best_position, best_fitness = pso(self.fitness_function, self.lower_bounds, self.upper_bounds, swarmsize=self.population_size, maxiter=self.num_iterations, debug=True)
            
            return best_fitness, best_position
    
        else:
            print("Wrong solver")
            
            return None, None
        
    def v(self, S, cS=1):
        if len(S) == 0:
            return 0
        elif len(S) == 1:
            return self.ideal[list(S)]
        else:
            vS = np.sum(self.ideal[i] for i in S)
            return (1 + np.random.rand()*cS/len(S)) * vS
        
    def calculate_shapley_values_old(self, c, num_players):
        shapley_values = np.zeros(num_players)
        for i in range(num_players):
            for coalition_size in range(1, num_players + 1):
                for m in range(1, coalition_size + 1):
                    coalition = set([i] + list(range(m)))
                    v_im = self.v(coalition, np.random.rand())
                    shapley_values[i] += v_im + (c[coalition_size - 1] / coalition_size) * v_im
                        
        return shapley_values
    
    def calculate_shapley_values(self, c, num_players):
        # TODO: adapt the c properly
        
        shapley_values = np.zeros(num_players+1)
        # Generate all permutations of players
        
        all_subsets = []
        for i in range(1, num_players + 1):
            all_subsets.extend(itertools.combinations(list(range(len(c))), i))
            
        for i in range(1,num_players+1):
            for S in all_subsets:
                if i in S:
                    coalition = set(S)
                    v_S = self.v(coalition)
                    coalition_minus_i = coalition - {i}
                    v_S_minus_i = self.v(coalition_minus_i)
                    num_permutations = math.factorial(len(coalition_minus_i)) * math.factorial(num_players - len(coalition))
                    shapley_values[i] += num_permutations * (v_S - v_S_minus_i)

        shapley_values /= math.factorial(num_players)

        return np.abs(shapley_values[1:])
