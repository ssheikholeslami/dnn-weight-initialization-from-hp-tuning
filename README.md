# Deep Neural Network Weight Initialization from Hyperparameter Tuning Trials
Code and results from our paper, "Deep Neural Network Weight Initialization from Hyperparameter Tuning Trials", published at ICONIP 2024.

# Read This First
The implementation of our proposed approach is *very* simple (and that's what we like about it the most) - considering you have your own hyperparameter tuning and model training code, you just have to:

- In your hyperparameter tuning code: add one line of code to the end of the "training loop", to save the model weights of an epoch. In PyTorch, this would be something like: `torch.save(model.state_dict(), save_dir+f"/trial_{trial_id}_epoch_{epoch}_weights.pth")`.

- In your model training code: add a few lines of code immediately after where you initialize your model. In PyTorch, this would look similar to:
```
if weight_initialization in ["hp-init", "hp-final", "hp-epoch"]:
    model.load_state_dict(torch.load(weights_path))
```
And that's it! you can add your own logic for when to save or load the weights, but what mentioned above is essentially everything you need to do to initialize your model weights from your hyperparameter tuning trials.

The rest of this document describes what you need to do if you want to *replicate/reproduce* the results from our paper. The `PaperCSVs` folder contains the raw results from our experiments and you can use `plots-latex.ipynb` to create the same plots and tables that you can see in our paper, but you can also run our scripts and re-do the hyperparameter tuning and model training experiments. Note that, due to the stochastic nature of training DNNs, and especially if you use version of frameworks other than what we have specified in `requirements.txt`, you'll most probably get slightly different results from what we got, but the trends and overall results should be the same.

# Prerequisites

## Python Version
To run the code without any issues, we recommend that you create a Python virtual environment with Python 3.9, e.g., with `python3.9 -m venv env-name`, activate the environment using `source env-name/bin/activate` and then clone the repo and use `pip install -r requirements.txt` to install the specific versions of the required libraries.


## GPUs and Parallelization
The provided code uses Ray, and allows for running each training and hyperparameter tuning trial in parallel with each other, i.e., if you have *n* GPUs, you can run *n* trials at the same time (note that this is different from *data-parallel training*). The experiments in the paper were done using 4 Nvidia RTX 2070S GPUs on a single machine running Ubuntu 22.04.4 LTS.

# Reproducing the Plots

The `PaperCSVs` folder contains the raw results from our experiments and you can use `plots-latex.ipynb` to create the same plots and tables that you can see in our paper (plus a few more plots that are not included in the paper).

If you run your own experiments, you can prepare your own CSV files and then reuse the plots. We have used **Weights & Biases** for experiment tracking and used its API to get these CSV files. If you have a W&B account, you may uncomment the lines of code in the `main.py` that correspond to `wandb` tracking.

# Running the Experiments

## Random Seeds
For the experiments included in the paper, we have used one set of random seeds for hyperparameter tuning trials and one set for model training. These were obtained using master seeds passed to `generate_seeds.py`:

- Tuning (master seed 3): `31190, 77678, 71333, 17094, 48490, 79157, 62135, 82014, 76133, 8588, 79377, 1725, 61503, 33994, 72192, 30714, 25132, 93998, 61638, 70906`
- Training (master seed 17): `68417, 54285, 39770, 47923, 37961, 22896, 92424, 92303, 70803, 86688, 36474, 14439, 3530, 32624, 50354, 97831, 54993, 33080, 65746, 41606`

## Run All Experiments

Use `run_all.sh` to run all the experiments included in the paper.

**Remember** that for each experiment, you need to first run the hyperparameter tuning script, and then set the "BASE_HP_PATH" variable in the script to run the corresponding training runs. If you're using Ray, the path would look similar to, e.g., `'~/ray_results/tune_hyperparameters_2024-04-18_19-20-06'`.

Thanks for checking and using our code! If you have any questions or comments about the approach or the paper, or notice any issues, or would like to work on more experiments, feel free to reach out to me over GitHub Issues or `sinash *at* k t h *dot* s e`.

