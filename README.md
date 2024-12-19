# 🚗STAGE: STyle driving Action GEneration with preference imitation learning


<!--#### Project Website: https://tonyzhaozh.github.io/aloha/-->


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate STAGE
- ``policy.py`` Policy module of STAGE, with loss functions(MAE, KL, preference) setting
- ``detr`` Model definitions of STAGE, modified from ACT and DETR. The ablation study could change the value of latent_sample and latent_input to achieve, such as remove VAE or remove VAE and preference.
- ``sim_env.py`` Add metadrive env setting
- ``collect_dataset_manual.py`` Collect data from metadrive, by PPO expert or human steering.
- ``constants.py`` Constants shared across files, such as task name, task data length
- ``utils.py`` Utils such as data loading and helper functions, such as extract trajectories and parse them to train data and preference data
- ``drive_style_gui`` This dir is our drive style GUI, by pysimplegui
- ``metadrive_util`` This dir includes some data process of vehicle and top_down_view render


### Installation

    conda create -n aloha python=3.9
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    pip install pysimplegui
    cd act/detr && pip install -e .
    git clone https://github.com/metadriverse/metadrive.git

change metadrive repo reference in metadrive_change.png, to differ color of ego car and other cars

    cd metadrive && pip install -e .

### Example Usages

To set up a new terminal, run:

    conda activate xxx
    cd <path to repo>

### Simulated experiments

First, use metadrive_util/collect_dataset_manual.py to collect drive dataset, user can use PPO expert to collect expert data or use steering wheel to collect human drive data. Run:

    python3 metadrive_util/collect_dataset_manual.py

To train STAGE:
    
    python3 imitate_episodes.py \
    "--task_name", "sim_drive",
    "--ckpt_dir", "ckpt_dir",
    "--policy_class", "ACT",
    "--kl_weight", "10",
    "--chunk_size", "10",
    "--hidden_dim", "512",
    "--batch_size", "1024",
    "--dim_feedforward", "3200",
    "--num_epochs", "1000",
    "--lr", "5e-5",
    "--seed", "0"


To evaluate the policy, run the same command but add ``--eval``. This loads some checkpoint of ckpt dir. And render setting can change `sim_env.py` and `get_topdown_config()` of `imitate_episodes.py`

### Acknowledgement

This repo is built on top of [ACT](https://github.com/tonyzhaozh/act), which already has a imitation transformer backbone and vae action modality encoder.

The env is built on top of [metadrive](https://github.com/metadriverse/metadrive), and author Quanyi Li provided much help for the env setting.