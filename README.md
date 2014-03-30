MH370_MCMC
==========

Modeling the Last Flight of MH370 with a Markov Chain Monte Carlo Method

See: http://nbviewer.ipython.org/github/myhrvold/MH370_MCMC/blob/master/MH370_MC_ConorMyhrvold.ipynb?create=1

For viewing IPython notebook in the web (or you can copy the url of the file yourself, in here: http://nbviewer.ipython.org/ )

Updates Mar 27 2014 (myhrvold):

Accompanying Fast Company Labs story: http://www.fastcolabs.com/3028265/how-i-narrowed-down-the-location-of-malaysia-air-using-monte-carlo-data-models

Feel free to reach out with comments or suggestions -- especially critiques and criticisms.

I've noticed several typos in the IPython notebook; I plan on addressing in an updated version (separate notebook).

Updates Mar 27 2014 (natlaughlin):

mh370.py is the standalone python script version of the above work by Conor L. Myhrvold (myhrvold).  

To run it, you will need to unzip runways.txt.zip to the same directory as the script.

This script outputs the plots to SVG instead of displaying them, but otherwise everything is the same.

Updates Mar 30 2014 (natlaughlin):

The latest bugfixed version is now updated according to:
http://nbviewer.ipython.org/github/myhrvold/MH370_MCMC/blob/master/MH370_MC_ConorMyhrvold-bug_fixes.ipynb?create=1

I've moved the previous version to the archive directory.

==========

ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip install numpy scipy matplotlib seaborn