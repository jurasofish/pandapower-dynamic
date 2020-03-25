# idk what this is yet #

## Installing scikits.odes ##
### windows ###

get the wheel off that website where that one guy packages heaps of stuff.

### mac ###

writing this up cause it was a pita.
These instructions are different to the scikits.odes docs.

1. create new conda environment `conda create -n ppd python=3.7`
2. Activate it `conda activate ppd`
3. Set it to use conda-forge for this environment
   4. `conda config --env --add channels conda-forge`
   5. `conda config --env --set channel_priority strict`
4. Install sundials `conda install sundials`
5. `conda install numpy pandas scipy ///whatever`
6. `pip install scikits.odes`
7. that's what worked for me ¯\\_(ツ)_/¯
