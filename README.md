# Federated Neural Bandits

This repository is the official implementation of our paper "Federated Neural Bandits" accepted to ICLR 2023. We have implemented both the communication-efficient variant: FN-UCB (Less Comm.) (see Sec. 3.4 of the main paper) and the main algorithm FN-UCB. FN-UCB (Less Comm.) is the one by default (see Sec. 5.1 of the main paper). Our implementation, especially the baseline algorithms, makes use of the official implementation of the paper "Neural Thompson Sampling", which can be found at: https://github.com/ZeroWeight/NeuralTS.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Run scripts:

To run our federated neural-upper confidence bound (FN-UCB) algorithm:
```run
python run_fn_ucb.py
```

To run different baseline algorithms (Neural UCB, Neural TS, Linear UCB, Linear TS, Kernelized UCB, Kernelized TS):
```run
python neural_ucb_ts.py
```

## Results:

The results are saved in the directory "results_fn_ucb". To visualize the results, use the jupyter notebook: plot_figures.ipynb.

