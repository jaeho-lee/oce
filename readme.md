# Learning bounds for risk-sensitive learning (NeurIPS 2020)

This repo contains a PyTorch codebase for the NeurIPS 2020 paper [Learning bounds for risk-sensitive learning](https://arxiv.org/abs/2006.08138).

(Disclaimer: It should be straightforward to implement these codes by yourself, perhaps in a more efficient way. Still, these codes may be helpful to beginners? :nerd_face:)

Our codebase is organized as follows. (*) denotes key parts.

- `models/`           : Contains the model neural network (ResNet18). You can add one if you want to, but make sure to modify `__init__.py`
- `tools/`            : Contains some functions.
- `tools/loaders.py`  : Contains dataloaders. You may add additional datasets.
- `tools/risks.py`    : (*) Contains functions to compute (batch)-CVaR and SVP.
- `tools/train.py`    : Ugly-looking train/test modules.
- `longtake.py`       : (*) Our main function. In the file, you would see what kind of arguments you want to use with `longtake.py`.
- `print.py`          : (*) After the main script is done running, you can use this to take summarize seed-wise results to get the csv file with average/std.

For quick runs, you may try typing:

`python longtake.py --cuda 0 --target_risk avg
python longtake.py --cuda 0 --target_risk meanstd --stdmult 1.0
python longtake.py --cuda 0 --target_risk cvar --betabar 0.4`


Have fun!
Jaeho.

P.S. "betabar" is simply an "alpha" for CVaR.