import pygmo as pg
import numpy as np
import GA.solver as solver
import ehtim as eh
import sys

config_file = sys.argv[1]
outfile = sys.argv[2]

inputs = {'probl':'EHT', 
         'dim':'not needed',
         'config':config_file
         }

udp, fit, pop, config = solver.solve(**inputs)

obs = fit.obs.copy()
res = obs.res()

reg_term  = {'simple' : 0*100,    # Maximum-Entropy
             'tv'     : 0*1.0,    # Total Variation
             'tv2'    : 0*1.0,    # Total Squared Variation
             'l1'     : 0.0,    # L1 sparsity prior
             'flux'   : 1e4}    # compact flux constraint

data_term = config['data_term']

maxit = 100
sys_noise = 0.02
stop = 1e-4
uv_zblcut = 0.1*10**9

prior = fit.prior.copy()
prior = prior.add_gauss(1e-2*config['zbl'], (100*eh.RADPERUAS, 100*eh.RADPERUAS, 0, 100*eh.RADPERUAS, 100*eh.RADPERUAS))
# Initialize imaging with a Gaussian image
imgr = eh.imager.Imager(obs, prior, prior_im=prior,
                    flux=config['zbl'], data_term=data_term, maxit=maxit,
                    norm_reg=True, systematic_noise=sys_noise,
                    reg_term=reg_term, ttype='direct', cp_uv_min=uv_zblcut, stop=stop)

# Imaging
imgr.make_image_I(show_updates=False)

for i in range(10):
        init = imgr.out_last().blur_circ(res)
        imgr.init_next = init
        imgr.make_image_I(show_updates=False)

weights = pg.decomposition_weights(n_f=udp.get_nf(), n_w=len(pop), method=config['decomposition_method'], seed=config['decomposition_seed'])

gen = config['generations']
neighbours = config['neighbours']
CR = config['CR']
F = config['F']
eta_m = config['eta_m']
realb = config['realb']
limit = config['limit']
preserve_diversity = config['preserve_diversity']

algo = pg.algorithm(pg.moead(gen=gen, neighbours=neighbours, decomposition="weighted", weight_generation=config["decomposition_method"], seed=config['decomposition_seed'], CR=CR, F=F, eta_m=eta_m, realb=realb, limit=limit, preserve_diversity=preserve_diversity))

algo.set_verbosity(1)

for i in range(len(pop)):
    pop.set_x(i, imgr.out_last().imvec/config['rescaling'])

pop = algo.evolve(pop)

fits, vectors = pop.get_f(), pop.get_x() 
npix = int(np.sqrt(len(vectors[0])))

save = True

if save:
    #d = np.asarray([fits,vectors,weights], dtype=object)
    #np.save(outfile, d)
    np.save(outfile+r'_fits.npy', fits)
    np.save(outfile+r'_vectors.npy', vectors)
    np.save(outfile+r'_weights.npy', weights)
    
