conda install python=3.11
conda install pip
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install pagmo pagmo-devel
pip install cloudpickle
conda install pygmo
pip install numpy scipy matplotlib numba pandas
pip install joblib ephem future
pip install astropy h5py
pip install scikit-image
wget https://github.com/liamedeiros/ehtplot/archive/refs/heads/master.zip
unzip master.zip -d .
pip install ./ehtplot-master
wget https://github.com/achael/eht-imaging/archive/refs/tags/v1.2.6.zip
unzip v1.2.6.zip -d .
pip install ./eht-imaging-1.2.6
pip install ./mr_beam/itreg
pip install ./mr_beam/libwise-0.4.7-light
pip install ./mr_beam/MSI
pip install ./mr_beam/imagingbase
pip install ./mr_beam/ga
