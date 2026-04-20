# Algebraic Priors for Approximately Equivariant Networks

This repository hosts the code for Algebraic Priors for Approximately Equivariant Networks (GrAM@ICLR 2026). If you find this work useful in your research, please cite:

```bibtex
@inproceedings{ali2026algebraic,
  title={Algebraic priors for approximately equivariant networks},
  author={Ali, Riccardo and Lio, Pietro and Vicary, Jamie},
  booktitle={ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling},
  year={2026},
  url={[https://openreview.net/forum?id=4ig6Rj4aio](https://openreview.net/forum?id=4ig6Rj4aio)}
}
```

We packaged the code into three self-contained directories to reproduce the corresponding experiments in the paper. Each directory contains its own `requirements.txt` file.


### Exploratory experiments to find the optimal representation on the latent space
1. Navigate to the directory and install the requirements
```
cd exploration
pip install -r requirements.txt
```

2. To run the TMNIST, MNIST and CIFAR10 experiments, run the following commands (respectively)
```
python3 train.py --dataset=TMNIST --latent_dim=6 --run_id=2 --model=RegularizedFunctor --lambda_t=0.5 --lambda_W=1 --x2_transformation=font --W_exponent_algebra=2 

python3 train.py --data=MNIST --dataset=MultiWMNIST --x2_angle=60 --latent_dim=18 --run_id=-1 --model=MultiWFunctorMNIST --lambda_t=0.5 --lambda_W=0.5

python3 train.py --model=EncoderClassifierFunctor --lambda_t=0.5 --lambda_W=1.0 --latent_dim=64 --run_id=1000 --x2_angle=90 --dataset=ClassificationDataset --data=CIFAR
```

### DDMNIST experiments
1. Navigate to the directory and install the requirements
```
cd DDMNIST_MedMNIST3d
pip install -r requirements.txt
```

2. To run the $C_4, D_1, D_4$ experiments, run the following commands (respectively)
```
python3 lightning_train_and_eval.py --group=C4xC4 --model=GxGregularfunctor --dataset=ddmnist_c4 --lambda_t=2 --lr=0.001 --fast

python3 lightning_train_and_eval.py --group=D1xD1 --model=GxGregularfunctor --dataset=ddmnist_d1 --lambda_t=1 --lr=0.001 --fast

python3 lightning_train_and_eval.py --group=D4xD4 --model=GxGregularfunctor --dataset=ddmnist_d4 --lambda_t=1 --lr=0.0005 --fast
```
Optionally, you can set `--seed=d` with d $\in\{0,1,2,3,4\}$ for reproducibility.
Add `--fast` for faster training with reduced logging overhead.

Simlarly, to run the experiments for the baseline CNN, run the following commands
```
python3 lightning_train_and_eval.py --group=C4xC4 --model=GxGregularfunctor --dataset=ddmnist_c4 --lambda_t=0 --lr=0.0005 --fast

python3 lightning_train_and_eval.py --group=D1xD1 --model=GxGregularfunctor --dataset=ddmnist_d1 --lambda_t=0 --lr=0.001 --fast

python3 lightning_train_and_eval.py --group=D4xD4 --model=GxGregularfunctor --dataset=ddmnist_d4 --lambda_t=0 --lr=0.0005 --fast
``` 


### MedMNIST experiments
1. Navigate to the directory and install the requirements
```
cd DDMNIST_MedMNIST3d
pip install -r requirements.txt
```

2. To run the Nodule3D, Synapse3D, Organ3D experiments, run the following commands (respectively)
```
python3 lightning_train_and_eval.py --group=S4 --model=MedMNISTGxGregularfunctor --dataset=S4 --data_flag=nodulemnist3d --lambda_t=1.0 --lr=0.00005

python3 lightning_train_and_eval.py --group=S4 --model=MedMNISTGxGregularfunctor --dataset=S4 --data_flag=synapsemnist3d --lambda_t=1.0 --lr=0.0001

python3 lightning_train_and_eval.py --group=S4 --model=MedMNISTGxGregularfunctor --dataset=S4 --data_flag=organmnist3d --lambda_t=2.0 --lr=0.0001
```

Optionally, you can set `--seed=d` with d $\in\{0,1,2,3,4\}$ for reproducibility.

Simlarly, to run the experiments for the baseline CNN, run the same commands with `--lambda_t=0`


### SMOKE experiments
1. Download the smoke plume dataset from https://github.com/Rose-STL-Lab/Approximately-Equivariant-Nets/tree/master

2. Navigate to the directory and install the requirements (conda required)
```
cd SMOKE
conda create -n "smoke" python=3.9.16
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

3. Run the following command:
```
python3 run_model.py --dataset=PhiFlow --relaxed_symmetry=Rotation --hidden_dim=92 --num_layers=5 --out_length=6 --alpha=1e-5 --batch_size=16 --learning_rate=0.001 --decay_rate=0.95 --latent_action=grid --get_latents --model=CNN --lambda_t=0.005
```
Simlarly, to run the experiments for the baseline CNN, run the same command with `--lambda_t=0`

### SHREC 11 experiments
1. Navigate to the directory and install the requirements
```
cd SHREC
pip install -r requirements.txt
```

2. Download and preprocess the data following the instructions at https://github.com/vsitzmann/neural-isometries/tree/main <br> The following command assumes that the preprocessed data is in data/SHREC_11/processed


3. Create the directories ./cshrec11_encode/weights and ./cshrec11_pred/weights/ Run the following command
```
python3 experiments/cshrec11_encode/train.py --in "data/SHREC_11/processed/" --out "./cshrec11_encode/weights/" --linear --augmentation "oh" && python3 experiments/cshrec11_pred/train.py --in "data/SHREC_11/processed/" --weights "./cshrec11_encode/weights/linear_oh/0/checkpoints-0/" --out "./cshrec11_pred/weights/" --linear --projection=matrix
```
