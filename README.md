# mrbeam

This is the second release of the MrBeam software tool.

Caution! This is alpha quality software, do expect bugs and sparse documentation!

MrBeam is a software written for VLBI imaging, in particular sparse mm-VLBI data. 

The main features of this release of MrBeam are as follows:

-> a general VLBI interface for regpy

-> a support for convex solvers and non-L2 spaces adding a more diverse set of solver options to ehtim

-> an implementation of multiscalar decompositions for imaging, in particular:

        -> DoG-HiT (Mueller, Lobanov 2022a, https://ui.adsabs.harvard.edu/abs/2022arXiv220609501M/abstract)
        
        -> DoB-CLEAN (Mueller, Lobanov 2023a, https://ui.adsabs.harvard.edu/abs/2023A%26A...672A..26M/abstract)
        
        -> Dynamic polarimetry with the multiresolution support (Mueller, Lobanov 2023b, https://ui.adsabs.harvard.edu/abs/2023A%26A...673A.151M/abstract)
        
-> an extension point to neural networks

This is a light version of MrBeam. Expect much more to come in consecutive releases.

We added some jupyter-notebook tutorials for a quick start with MrBeam. Some tutorials are still under construction, but will be added soon.

MrBeam makes explicit use of:

-> ehtim (Chael, Johnson, Narayan et. al. 2016, https://ui.adsabs.harvard.edu/abs/2016ApJ...829...11C/abstract; Chael, Johnson, Bouman et. al. 2018, https://ui.adsabs.harvard.edu/abs/2018ApJ...857...23C/abstract; source code: https://github.com/achael/eht-imaging)

-> WISE (Mertens, Lobanov 2015, https://ui.adsabs.harvard.edu/abs/2015A%26A...574A..67M/abstract; source code: https://github.com/flomertens/wise), we provide a light version of libwise (lightwise)

-> regpy (https://num.math.uni-goettingen.de/regpy/), we provide a local version of regpy with minor changes to provide adaptability

Installation guide:
The installation is easiest with anaconda. Create a new environment with python 3 and install all dependencies first:

-> numpy

-> scipy

-> matplotlib

-> numba

-> joblib

-> astropy

-> ephem

-> future

-> h5py

-> pandas

-> skimage (install by conda with install scikit-image)

-> pynfft (install by conda with install -c conda-forge pynfft)

-> ehtplot (pip install from https://github.com/liamedeiros/ehtplot)

-> ehtim (pip install from https://github.com/achael/eht-imaging)

Download and unpack MrBeam, cd in the subfolders itreg, libwise_0.4.7_light, MSI and imagingbase and install the packages by "pip install .".

MrBeam is currently under developement and in application to many sparse VLBI data analysis projects, such as for the EHT, RadioAstron and the EVN. Feel free to contribute.
