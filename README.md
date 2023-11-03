# mrbeam

This is the second release of the MrBeam software tool.

Caution! This is alpha quality software, do expect bugs and sparse documentation!

MrBeam is a software written for VLBI imaging, in particular sparse mm-VLBI data.

## Features:

The main features of this release of MrBeam are as follows:

-> a general VLBI interface for regpy

-> a support for convex solvers, evolutionary algorithms and non-Euclidean spaces adding a more diverse set of solver options to ehtim

-> an implementation of multiscalar decompositions for imaging, in particular:

    -> DoG-HiT (Mueller, Lobanov 2022a, https://ui.adsabs.harvard.edu/abs/2022arXiv220609501M/abstract)
    
    -> DoB-CLEAN (Mueller, Lobanov 2023a, https://ui.adsabs.harvard.edu/abs/2023A%26A...672A..26M/abstract)
    
    -> Dynamic polarimetry with the multiresolution support (Mueller, Lobanov 2023b, https://ui.adsabs.harvard.edu/abs/2023A%26A...673A.151M/abstract)

-> an implementation of multiobjective VLBI imaging, in particular

    -> MOEAD (Mueller, Mus, Lobanov 2023, https://ui.adsabs.harvard.edu/abs/2023A%26A...675A..60M/abstract)

    -> MOEAD II (Mus, Mueller, Marti-Vidal, Lobanov, submitted to A&A) 

-> an extension point to neural networks

We added some jupyter-notebook tutorials for a quick start with MrBeam. Some tutorials are still under construction, but will be added soon.

## Installation guide:

The installation is easiest with anaconda. Create a new environment with python 3 (<3.12):

                conda create -n mrbeam
   
                conda activate mrbeam

Then clone the repo:

	        git clone https://github.com/hmuellergoe/mrbeam.git
         
Finally run the installation script:

                cd mrbeam
         
	        bash -x build.sh
         
All the commands for manual implementation are contained in installation_guide.txt. The implementation is tested on ehtim version 1.2.6, python=3.9-3.11, python=3.12 will not allow to install the dependency pygmo.

## References

If you use MrBeam for a manuscript, please include a reference to [Mueller,Lobanov 2022, A&A 666, A137], [Mueller,Lobanov 2023a, A&A 672, A26], and [Mueller, Mus, Lobanov 2023, A&A 675, A60]. If you use the polarimetric, time-dynamic functionalities of MrBeam we also ask you to put a reference to [Mueller, Lobanov 2023b, A&A 673, A151] (for DoG-HiT) and/or [Mus, Mueller, Marti, Lobanov 2023, submitted] (for MOEA/D).

Please do not forget to acknowledge the dependencies appropriately. MrBeam makes explicit use of source code contained in:

        -> ehtim (Chael, Johnson, Narayan et. al. 2016, https://ui.adsabs.harvard.edu/abs/2016ApJ...829...11C/abstract; Chael, Johnson, Bouman et. al. 2018, https://ui.adsabs.harvard.edu/abs/2018ApJ...857...23C/abstract; source code: https://github.com/achael/eht-imaging)

        -> WISE (Mertens, Lobanov 2015, https://ui.adsabs.harvard.edu/abs/2015A%26A...574A..67M/abstract; source code: https://github.com/flomertens/wise), we provide a light version of libwise (lightwise) ported to python 3

        -> regpy (https://num.math.uni-goettingen.de/regpy/), we provide a local version of regpy with minor changes to provide adaptability

Moreover, the multiobjective functionalities of MrBeam use the pygmo software tool:

        -> pygmo (Biscani, Izzo 2020, https://joss.theoj.org/papers/10.21105/joss.02338, website: https://esa.github.io/pygmo2/)

MrBeam is currently under developement and in application to many sparse VLBI data analysis projects, such as for the EHT, RadioAstron and the EVN. Feel free to contribute.


