MH370_MCMC
==========

Modeling the Last Flight of MH370 with a Markov Chain Monte Carlo Method

The original source is hosted here:
https://github.com/myhrvold/MH370_MCMC

Fast Lab Company stories:

http://www.fastcolabs.com/3028265/how-i-narrowed-down-the-location-of-malaysia-air-using-monte-carlo-data-models

http://www.fastcolabs.com/3028331/more-about-our-methodology-tracking-mh370-with-monte-carlo-data-models

http://www.fastcolabs.com/3028569/this-data-model-shows-mh370-could-not-have-flown-accidentally-to-its-destination

Updates
----------------

2014-03-27 (myhrvold):

Accompanying Fast Company Labs story: http://www.fastcolabs.com/3028265/how-i-narrowed-down-the-location-of-malaysia-air-using-monte-carlo-data-models

Feel free to reach out with comments or suggestions -- especially critiques and criticisms.

I've noticed several typos in the IPython notebook; I plan on addressing in an updated version (separate notebook).

2014-03-27 (natlaughlin):

mh370.py is the standalone python script version of the above work by Conor L. Myhrvold (myhrvold).  

To run it, you will need to unzip runways.txt.zip to the same directory as the script.

This script outputs the plots to SVG instead of displaying them, but otherwise everything is the same.

2014-03-29 (natlaughlin):

The latest bugfixed version is now updated according to:
http://nbviewer.ipython.org/github/myhrvold/MH370_MCMC/blob/master/MH370_MC_ConorMyhrvold-bug_fixes.ipynb?create=1

I've moved the previous version to the archive directory.

2014-04-01 (natlaughlin):

Moved v1 script to the archive.  Created a standalone script according to:

http://qa.nbviewer.ipython.org/github/myhrvold/MH370_MCMC/blob/master/MH370_MC_ConorMyhrvold-V2-Part1.ipynb?create=1

http://qa.nbviewer.ipython.org/github/myhrvold/MH370_MCMC/blob/master/MH370_MC_ConorMyhrvold-V2-Part2.ipynb?create=1

http://qa.nbviewer.ipython.org/github/myhrvold/MH370_MCMC/blob/master/MH370_MC_ConorMyhrvold-V2-Part3.ipynb?create=1



It is now called:
mh370_mcmc_v2.py

There is now a command line interface to change the monte carlo parameters and specify an output directory for the plots (see Examples).  

The runways file is now in the archive/v1 directory, as the v2 script no longer requires it.

2014-04-02 (natlaughlin):



Installation Requirements
----------------

- python (https://www.python.org/download/releases/2.7.6/)
- GEOS (http://trac.osgeo.org/geos/)
- numpy (http://www.numpy.org/)
- scipy (http://www.scipy.org/)
- matplotlib (http://matplotlib.org/)
- basemap (https://github.com/matplotlib/basemap)
- seaborn (https://github.com/mwaskom/seaborn)

Mac OSX
```
ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/homebrew/go/install)"
brew install python geos
pip install virtualenv
virtualenv env
source ./env/bin/activate
export ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future
pip install numpy scipy matplotlib seaborn
easy_install basemap
```
	
Help
----------------
```
python ./mh370_mcmc_v2.py -h
```	
	
Examples
----------------
This will put the plots in the directory "./examples/default"
```
python ./mh370_mcmc_v2.py -o examples/default
```

How to change the monte carlo parameters:

This will change the last_known_heading to north
```
python ./mh370_mcmc_v2.py -l 0
```

This will change the km_hop to 400 km/h
```
python ./mh370_mcmc_v2.py -k 400
```

This will change the number of simulations to 10
```
python ./mh370_mcmc_v2.py -n 10
```
