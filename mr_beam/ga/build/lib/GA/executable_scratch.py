import pygmo as pg
import numpy as np
import GA.solver as solver
import sys

config_file = sys.argv[1]
outfile = sys.argv[2]

inputs = {'probl':'EHT', 
         'dim':'not needed',
         'config':config_file
         }

udp, fit, pop, config = solver.solve(**inputs)

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

pop_array = pop.get_x() * len(pop) * config["zbl"] / (np.sum(pop.get_x() * config["rescaling"]) )
for i in range(len(pop)):
    pop.set_x(i, pop_array[i])

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
    
