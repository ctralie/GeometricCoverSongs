# GeometricCoverSongs


## Getting Started
First, you will need to install the dependencies
* numpy/scipy/matplotlib
* [librosa]: Used for audio reading, dynamic programming beat tracking, MFCCs
* [Essentia] with Python bindings.  Used for HPCP features
* [Madmom] (optional): State of the art beat tracking, best if only one beat level is used

Once you have the dependencies installed, you can checkout the code
~~~~~ bash
git clone https://github.com/ctralie/GeometricCoverSongs.git
git submodule update --init
git submodule update --remote
~~~~~

If this worked properly, you should see a subdirectory pyMIRBasic populated with various Python files.  You will also need to compile the C files used for Smith Waterman sequence alignment

~~~~~ bash
cd SequenceAlignment
python setup.py build_ext --inplace
~~~~~


## Quick Comparison of Two Songs with Detailed Plots
The file *SongComparator.py* is a quickstart for running the pipeline on a pair of songs.  The code will run and output cross-similarity matrices and Smith Waterman matrices for HPCPs, MFCCs, MFCC SSMs, and similarity fusion on all of the above.  Right now, it is set to compare an example from the [Covers80] dataset (see below), but you can modify the __main__ function to load in any two songs of your choosing.


## Quick Comparison of Two Songs with GUI


## [Covers80] Dataset

[click here] to download the [Covers80] dataset (164MB).  Extract the *covers32k* directory at the root of this repository.  Then you can run the file *Covers80.py* to replicate the experiments reported in the [paper].  The file will output an HTML table with mean rank, mean reciprocal rank, median rank, and covers80 score (see [paper] for more details).


[Chris Tralie]: <http://www.ctralie.com>
[paper]: <http://www.covers1000.net/ctralie2017_EarlyMFCC_HPCPFusion.pdf>
[librosa]: <http://librosa.github.io/librosa/install.html>
[Essentia]: <http://essentia.upf.edu/documentation/installing.html>
[Madmom]: <http://madmom.readthedocs.io/en/latest/>
[Covers80]: <https://labrosa.ee.columbia.edu/projects/coversongs/covers80/>
[click here]: <https://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tgz>
