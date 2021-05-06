# InfoTracking

Time-lapse microscopy and simulation tracking using entropy and mutual information.

Requires:
* numpy
* scipy
* matplotlib

Jupyter required for notebook based data exploration and plotting. 

`conda create -n infotracking python=3.6 numpy scipy scikit-image matplotlib jupyter`

`source activate infotracking`

`conda install -c conda-forge opencv`

`python EM.py <filename_pattern> <start_frame> <number_frames>`

where `<filename_pattern>` is a format string with full path, e.g. `/Users/tim/data/image%03d.tif`.

