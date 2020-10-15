.

Our codebase is organized as follows. (*) denotes key parts.

- models/           : Contains the model neural network (ResNet18). You can add one if you want to, but make sure to modify '__init__.py'
- tools/            : Contains some functions.
- tools/loaders.py  : Contains dataloaders. You may add additional datasets.
- tools/risks.py    : (*) Contains functions to compute (batch)-CVaR and SVP.
- tools/train.py    : Ugly-looking train/test modules.
- longtake.py       : (*) Our main function. In the file, you would see what kind of arguments you want to type into 'run.sh' file.
- run.sh            : (*) Script to run 'longtake.py' but requires some un-commenting.
- print.py          : (*) After the main script is done running, you can use this to take summarize seed-wise results to get the csv file with average/std.

Have fun!
Authors.
--------------------------------------------
P.S. "betabar" is simply an "alpha" for CVaR.