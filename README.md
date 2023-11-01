# State-Action Similarity-Based Representations for Off-Policy Evaluation

This is the source code accompanying the paper [***State-Action Similarity-Based Representations for Off-Policy Evaluation***](https://arxiv.org/abs/2310.18409) by Brahma S. Pavse and Josiah P. Hanna.

## Directory Structure
```
.
├── cheetah
│   └── policies
│       ├── cheetah_model_sac_420000_steps.zip (custom dataset evaluation/expert pi)
│       ├── cheetah_model_sac_60000_steps.zip (custom dataset behavior/medium pi)
│       ├── halfcheetah_online_10.pkl (d4rl dataset evaluation/expert pi)
│       └── halfcheetah_online_5.pkl (d4rl dataset behavior/medium pi)
├── hopper
│   └── policies
│       ├── hopper_online_10.pkl (d4rl dataset evaluation/expert pi)
│       └── hopper_online_5.pkl (d4rl dataset behavior/medium pi)
├── humanoidstandup
│   └── policies
│       ├── humstd_model_sac_200000_steps.zip (custom dataset behavior/medium pi) 
│       └── humstd_model_sac_350000_steps.zip (custom dataset evaluation/expert pi)
├── mujoco_xmls
│   └── custom_swimmer.xml (custom body design of swimmer)
├── swimmer
│   └── policies
│       ├── swimmer_model_sac_220000_steps.zip (custom dataset behavior/medium pi) 
│       └── swimmer_model_sac_350000_steps.zip (custom dataset evaluation/expert pi)
├── walker
│   └── policies
│       ├── walker_online_10.pkl (d4rl dataset evaluation/expert pi)
│       └── walker_online_5.pkl (d4rl dataset behavior/medium pi)
├── behavior_dataset.py (functionality to manage offline dataset such as sampling, normalization)
├── cfg.yaml (specifies policies, dataset mixtures etc.)
├── custom_cheetah.py (can be ignored since same as native v4 implementation)
├── custom_humstd.py (can be ignored since same as native v4 implementation)
├── custom_swimmer.py (loads a custom body design for swimmer; see above)
├── estimators.py (produces policy value estimates using dataset)
├── fqe.py (core FQE code)
├── gen_offline_dataset.py (generates dataset according to specification)
├── learn_phi.py (core ROPE code)
├── plot.py (plotting code for reference)
├── policies.py (includes policy wrapper code and torch neural network wrappers)
├── README.md
├── requirements.txt
├── run_single_learn_phi_ope_main.py (main launch script)
└── utils.py (supporting utility code)
```

## Requirements
The required libraries can be installed as follows:

```
pip install -r requirements.txt
```

## Policies

All evaluation and behavior policies are in the appropriate directories: `cheetah`, `humanoidstandup`, `swimmer`, `hopper`, and `walker`. The d4rl-based ones were taken from [here](https://github.com/google-research/deep_ope).

## Datasets

### Downloading Datasets

* All the datasets used in the paper (custom and d4rl generated) are available [here](https://drive.google.com/file/d/1eTnwgz-lvtxu6jRnFmfA_a8rtAPtY6yJ/view?usp=sharing)

### Generating Datasets

In case a user wants to generate the datasets, they can do the following.

For custom datasets:

```
python gen_offline_dataset.py --env_name <env> --oracle_num_traj 300 --gamma 0.99  --seed 2347  --d4rl_dataset false --dataset_name medium-expert --samples_to_collect 1e5
```
where "env" is from `Swimmer`, `HumanoidStandup`, `Cheetah`.

For d4rl-based datasets:

```
python gen_offline_dataset.py --env_name <env> --oracle_num_traj 300 --gamma 0.99  --seed 2347  --d4rl_dataset true --dataset_name <name> --samples_to_collect 1e6
```
where "env" is from `Cheetah`, `Hopper`, `Walker`, and "name" is from `random`, `medium`, `medium-expert`.

## Training

Ensure that there is a directory folder called `datasets/` and the above `.npy` dataset file is in this directory. Note that seed generation for the 20 trials was done by picking a random integer between 0 and 1M. 

Common information: 
- "result-file-name" is the name of the `.npy` to save the results of the single run in.
- custom datasets "env" is `Swimmer`, `HumanoidStandup`, `Cheetah`; "ds-name" is `medium-expert`; "ds-size" is `1e5`; "d4rl-flag" is `false`
- d4rl datasets "env" is `Cheetah`, `Hopper`, `Walker`; "ds-name" is `random`, `medium`, or `medium-expert`; "ds-size" is `1e6`; "d4rl-flag" is `true`
- "fqe-clip-flag" will clip the bootstrapping target for FQE. Note in the paper, this was set to `false`.
- Following commands train a single run for the specified algorithm and setting.


**1. Training FQE**
Executes FQE using the original/native state-action pairs.
```
python3 run_single_learn_phi_ope_main.py --outfile <result-file-name> --seed 0 --env_name <env> --gamma 0.99 --epochs 300000 --exp_name fqe --normalize_states true --normalize_rewards false  --Q_hidden_dim 256 --dataset_name <ds-name> --samples_to_collect <ds-size> --d4rl_dataset <d4rl-flag> --fqe_clip_target <fqe-clip-flag>
```

**2. Training ROPE + FQE**
This command will pre-train ROPE representations, and then execute FQE using the pre-trained representations.
```
python3 run_single_learn_phi_ope_main.py --outfile <result-file-name> --seed 0 --env_name <env> --gamma 0.99 --epochs 300000 --encoder_name off-policy-sa --phi_epochs 300000 --exp_name fqe --normalize_states true --normalize_rewards false --rep_layer_norm true --phi_hidden_dim 256 --Q_hidden_dim 256 --dataset_name <ds-name> --samples_to_collect <ds-size> --d4rl_dataset <d4rl-flag> --fqe_clip_target false
```

The results can be viewed in "result-file-name". The data is stored as a python dictionary. See `run_single_learn_phi_ope_main.py` to see the format.


## Citation
If you found any part of this code useful, please consider citing our paper:

```
@inproceedings{
  pavse2023rope,
  title={State-Action Similarity-Based Representations for Off-Policy Evaluation},
  author={Brahma S. Pavse, Josiah P. Hanna},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
}

```
## Contact

If you have any questions, please feel free to email: pavse@wisc.edu!
