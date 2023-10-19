import pygmo as pg
import numpy as np
import sys
sys.path.append('/')

STRINGKEYS = ['img', 'uvf', 'decomposition_method', 'minimization_algorithm', 'mode']
FLOATKEYS = ['rescaling', 'rescalingV', 'zbl', 'prior_fwhm', 'CR', 'F', 'eta_m', 'realb', 'blur_circ', 'C', 'tau', 'pcut', 'max_weight', 'snr']
INTKEYS = ['grid_size', 'seed_initial', 'num_cores', 'generations', 'neighbours', 'decomposition_seed', 'limit']
REGKEYS = ['l1w', 'simple', 'tv', 'tv2', 'lA', 'flux', 'ngmem', 'entr', 'msimple', 'hw', 'ptv']
DATKEYS = ['vis', 'amp', 'cphase', 'logcamp', 'pvis']
BOOLKEYS = ['preserve_diversity', 'use_gradient','parallel']

def calculate_pop_size_7(grid_size):
    x = np.arange(grid_size+1)
    return int(np.sum(np.sum(np.meshgrid(x, x, x, x, x, x, x), axis=0) == grid_size))

def calculate_pop_size_5(grid_size):
    x = np.arange(grid_size+1)
    return int(np.sum(np.sum(np.meshgrid(x, x, x, x, x), axis=0) == grid_size))

def calculate_pop_size_4(grid_size):
    x = np.arange(grid_size+1)
    return int(np.sum(np.sum(np.meshgrid(x, x, x, x), axis=0) == grid_size))

def read_config_params(file, probl):
    data = open(file, 'r').readlines()
    
    dictionary = {}
    dictionary['reg_term'] = {}
    dictionary['data_term'] = {}
    
    for i in range(len(data)):
        key = data[i].split('=')[0]
        field = data[i].split('=')[1].split('\n')[0]
        if key in STRINGKEYS:
            dictionary[key] = field
        if key in FLOATKEYS:
            dictionary[key] = float(field)
        if key in INTKEYS:
            dictionary[key] = int(field)
        if key in REGKEYS:
            dictionary['reg_term'][key] = float(field)
        if key in DATKEYS:
            dictionary['data_term'][key] = float(field)
        if key in BOOLKEYS:
            dictionary[key] = bool(field)
    
    if probl == 'EHT':
        dictionary['pop_size'] = calculate_pop_size_7(dictionary['grid_size'])
    if probl == 'Pol':
        dictionary['pop_size'] = calculate_pop_size_4(dictionary['grid_size'])
    if probl == 'Full_Stokes':
        dictionary['pop_size'] = calculate_pop_size_5(dictionary['grid_size'])
            
    return dictionary
        

def solve(**kwargs):
    
    probl = kwargs.pop('probl')
    print (probl)
    
    config = kwargs.get('config')
    
    
    if probl == 'EHT':
        
        from GA.problems import EHT
        
        import ehtim as eh
        
        dictionary = read_config_params(config, probl)
        
        reg_term = dictionary['reg_term']
        data_term = dictionary['data_term']
           
        rescaling = dictionary['rescaling']
        zbl = dictionary['zbl']

        #Load true sky brightness distribution
        img = eh.image.load_fits(dictionary['img'])
        
        #img = img.regrid_image(img.fovx(), 16)

        target_im = img.imarr()
        target_im = np.asarray(target_im/rescaling, dtype=float)

        #Load synthetic data 
        obs = eh.obsdata.load_uvfits(dictionary['uvf'])

        npix = img.xdim
        fov = img.fovx()
        
        # regrid
        #npix = 16
        #img = img.regrid_image(img.fovx(), npix)
        #fov = img.fovx()


        #prior image, leave it blank for now
        prior = eh.image.make_square(obs, npix, fov)
        prior_fwhm = dictionary['prior_fwhm']*eh.RADPERUAS
        prior = prior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
        
        num_cores = dictionary['num_cores']
        EHTfit = EHT.EHT(obs, prior, data_term, reg_term, rescaling, zbl, npix*npix, num_cores=num_cores)
        EHTfit.setFit()
        
        #prior = EHTfit.wrapper.Obsdata.dirtyimage(npix, fov).imarr()
        
        udp = pg.problem(EHTfit)
        
        pop_size = dictionary['pop_size']
        seed = dictionary['seed_initial']
        pop = pg.population(udp, size=pop_size, seed=seed)
        
#        for i in range(pop_size):
#            pop.set_x(i,prior.imarr().flatten())
        
        return [udp, EHTfit, pop, dictionary]
    
    if probl == 'Movie':
        from GA.problems import Movie
        import ehtim as eh
        
        dictionary = read_config_params(config, probl)
        
        reg_term = dictionary['reg_term']
        data_term = dictionary['data_term']
           
        rescaling = dictionary['rescaling']
        zbl = dictionary['zbl']
        C = dictionary['C']
        tau = dictionary['tau']

        #Load true sky brightness distribution
        img = eh.image.load_fits(dictionary['img'])
        
        #img = img.regrid_image(img.fovx(), 16)

        target_im = img.imarr()
        target_im = np.asarray(target_im/rescaling, dtype=np.float)

        #Load synthetic data 
        obs = eh.obsdata.load_uvfits(dictionary['uvf'])

        npix = img.xdim
        fov = img.fovx()
        
        # regrid
        #npix = 16
        #img = img.regrid_image(img.fovx(), npix)
        #fov = img.fovx()

        #prior image, leave it blank for now
        prior = eh.image.make_square(obs, npix, fov)
        prior_fwhm = dictionary['prior_fwhm']*eh.RADPERUAS
        prior = prior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
        
        num_cores = dictionary['num_cores']
        Moviefit = Movie.Movie(obs, prior, data_term, reg_term, rescaling, zbl, npix*npix, num_cores=num_cores, C=C, tau=tau)
        Moviefit.setFit()
        
        #prior = EHTfit.wrapper.Obsdata.dirtyimage(npix, fov).imarr()
        
        udp = pg.problem(Moviefit)
        
        pop_size = dictionary['pop_size']
        seed = dictionary['seed_initial']
        pop = pg.population(udp, size=pop_size, seed=seed)
        
#        for i in range(pop_size):
#            pop.set_x(i,prior.imarr().flatten())
        
        return [udp, Moviefit, pop, dictionary]       
    
    if probl == 'Pol':
        from GA.problems import Pol
        
        import ehtim as eh
        
        dictionary = read_config_params(config, probl)
        
        reg_term = dictionary['reg_term']
        data_term = dictionary['data_term']
           
        rescaling = dictionary['rescaling']
        zbl = dictionary['zbl']
        
        pcut = dictionary['pcut']

        #Load true sky brightness distribution
        img = eh.image.load_fits(dictionary['img'])
        
        #img = img.regrid_image(img.fovx(), 16)

        target_im = img.imarr()
        target_im = np.asarray(target_im, dtype=float)

        #Load synthetic data 
        obs = eh.obsdata.load_uvfits(dictionary['uvf'])

        npix = img.xdim
        fov = img.fovx()
        
        # regrid
        #npix = 16
        #img = img.regrid_image(img.fovx(), npix)
        #fov = img.fovx()


        #prior image, leave it blank for now
        #prior = eh.image.make_square(obs, npix, fov)
        #prior_fwhm = dictionary['prior_fwhm']*eh.RADPERUAS
        #prior = prior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
        prior = img.copy()
        
        num_cores = dictionary['num_cores']
        Polfit = Pol.Pol(obs, prior, data_term, reg_term, rescaling, zbl, 2*npix*npix, num_cores=num_cores, pcut=pcut)
        Polfit.setFit()
        
        #prior = EHTfit.wrapper.Obsdata.dirtyimage(npix, fov).imarr()
        
        udp = pg.problem(Polfit)
        
        pop_size = dictionary['pop_size']
        seed = dictionary['seed_initial']
        pop = pg.population(udp, size=pop_size, seed=seed)
        
#        for i in range(pop_size):
#            pop.set_x(i,prior.imarr().flatten())
        
        return [udp, Polfit, pop, dictionary]
    
    if probl == 'Full_Stokes':
        from GA.problems import Full_Stokes
        
        import ehtim as eh
        
        dictionary = read_config_params(config, probl)
        
        reg_term = dictionary['reg_term']
        data_term = dictionary['data_term']
           
        rescaling = dictionary['rescaling']
        zbl = dictionary['zbl']
        
        pcut = dictionary['pcut']

        #Load true sky brightness distribution
        img = eh.image.load_fits(dictionary['img'])
        
        #img = img.regrid_image(img.fovx(), 16)

        target_im = img.imarr()
        target_im = np.asarray(target_im, dtype=float)

        #Load synthetic data 
        obs = eh.obsdata.load_uvfits(dictionary['uvf'])

        npix = img.xdim
        fov = img.fovx()
        
        # regrid
        #npix = 16
        #img = img.regrid_image(img.fovx(), npix)
        #fov = img.fovx()


        #prior image, leave it blank for now
        #prior = eh.image.make_square(obs, npix, fov)
        #prior_fwhm = dictionary['prior_fwhm']*eh.RADPERUAS
        #prior = prior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
        prior = img.copy()
        
        num_cores = dictionary['num_cores']
        Polfit = Full_Stokes.FullStokes(obs, prior, data_term, reg_term, rescaling, zbl, 3*npix*npix, num_cores=num_cores, pcut=pcut, rescalingV=dictionary['rescalingV'])
        Polfit.setFit()
        
        #prior = EHTfit.wrapper.Obsdata.dirtyimage(npix, fov).imarr()
        
        udp = pg.problem(Polfit)
        
        pop_size = dictionary['pop_size']
        seed = dictionary['seed_initial']
        pop = pg.population(udp, size=pop_size, seed=seed)
        
#        for i in range(pop_size):
#            pop.set_x(i,prior.imarr().flatten())
        
        return [udp, Polfit, pop, dictionary]
        
    else:
        
        from ga.problems import Entropy, Polynomial
        
        dim = kwargs.pop('dim')
        print(dim, '%s.%s(%i)'%(probl,probl,dim))
        
        
        return [pg.problem(eval('%s.%s(%i)'%(probl,probl,dim))), 0, 0]

    
    

    
