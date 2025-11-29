# EMPI

The Source code for paper _A Novel Population Initialization Method via Adaptive Experience Transfer for General-Purpose Binary Evolutionary Optimization_

This page will tell you how to config the environment for the source code and run it.

# Quick Start
## Setup Environment
### Install Package and Build Library
Modify the path setting in the lines 2 of file `configs/env.sh` as following:
```shell
export EMPI="The Path of this project on your server"
```
Then, run `configs/env.sh`
```shell
./configs/env.sh
```

### Download Packages for Golang Executable File
Enter the path `src/distribution`, and then execute the following commands:
```shell
go mod init main
go mod tidy
```

## Dataset
There are 6 problem classes in this repo and 2 of them are generated according to the existing datasets. The datasets of 
complementary influence maximization problem and compiler arguments optimization problem have upload into the folder `data/dataset`.

#### Complementary Influence Maximization Problem
The dataset of Facebook/Wiki/Epinions for Complementary Influence Maximization Problem is located in the folder **data/dataset/com_imp**.

#### Compiler Arguments Optimization problem
The dataset for Compiler Arguments Optimization problem is located in the folder **data/dataset/compiler_args**.



## Run the Project

### Set the PYTHONPATH
Set the env_variable PYTHONPATH as: 
```shell
# You need to set the $EMPI as the root path of this project
# export EMPI="The Path of this project on your server"

export PYTHONPATH=$EMPI:$EMPI/src
```
While `$EMPI` is the root path of this project.

### Start Distribution System

**It is worth noting that to enhance computational power, EMPI runs on a distributed platform with a master-slave architecture. Therefore, the distributed system must be launched prior to starting the experiments.**

#### Start the master node of the distribution system:
Execute the following commands to start the master node.
```shell
cd $EMPI/src/distribution
go run distribution_master.go
```

#### Start the CPU task evaluation nodes of the distribution system
For each cpu node in the distribution system, execute the following commands:
```shell
cd $EMPI
python src/distribution/distribution_eval.py --task_capacity 512 --task_type "cpu" --master_host 10.16.104.19:1088
```
The `task_capacity` argument indicates the computational capability of the node. It is advisable not to set the value of `task_capacity` beyond the total number of CPU cores in the node. The `master_host` argument indicates the hostname of the master node, which depends on the IP address of your master node.

#### Start the GPU task evaluation nodes of the distribution system
For each gpu node in the distribution system, execute the following commands:
```shell
python src/distribution/distribution_eval.py --task_capacity 40 --task_type "gpu" --gpu_list 0 1 2 3 --master_host 10.16.104.19:1088
```
The `task_capacity` argument indicates the computational capability of the node. It is advisable not to set the value of `task_capacity` beyond the total number of CPU cores in the node. The `gpu_list` argument specifies the indices of available GPUs. The `master_host` argument indicates the hostname of the master node, which depends on the IP address of your master node.

### Run EMPI
Before running, you need to set the IP address of the master node in the distribution system. You can set the value in the file `src/experiments_setting.py`:
```Python
master_host = "127.0.0.1:1088"
```
To ensure efficient communication, the correct master hostname must be set. It is recommended to run EMPI directly on the same machine as the master node, which will assign the hostname as "127.0.0.1:1088". This setup enables EMPI to communicate with the master node through memory, enhancing performance and reliability.

#### Generate Problem Instances and Collect Experience
Run the `src/experiment_problem.py`
```shell
cd $EMPI/src/experiments
python experiment_problem.py
```

#### Represent Solving Experience
Represent each experience of the instance in the training set $S$ in a neural network form.

Run the `src/experiment_surrogate_training.py`
```shell
cd $EMPI
python src/experiments/experiment_surrogate_training.py
```

#### Calculate the Correlation
To streamline subsequent processes, we precompute the correlation coefficients between instances in the training set ($S$) and those in both the test set and the training set of the gating network ($S_G$).

Run the `src/experiments/experiment_calculate_correlation.py`
```shell
cd $EMPI
python src/experiments/experiment_calculate_correlation.py
```

#### Finetuning the Decoder and Generate Solutions
To streamline subsequent processes, we precompute experience transfers from instances in the training set ($S$) to those in both the test set and the training set of the gating network ($S_G$), generating the corresponding solution sets $X^{gen}$. Each pairwise experience transfer between instances is repeated three times.

Run the `src/experiments/experiment_decoder_mapping.py` and `src/experiments/experiment_init_eval.py`
```shell
cd $EMPI
python src/experiments/experiment_decoder_mapping.py
python src/experiments/experiment_init_eval.py
```

#### Train the Gating Network
Run the `src/experiments/experiment_train_gate.py`
```shell
cd $EMPI
python src/experiments/experiment_train_gate.py 
```
#### Evaluate EMPI on the Test Set

Run the `src/experiments/experiment_eval_init_ea.py`
```shell
cd $EMPI
python src/experiments/experiment_eval_init_ea.py 
```

#### Evaluate Baselines on the Test Set

Run the `src/experiments/experiment_baseline_init_ea.py`
```shell
cd $EMPI
python src/experiments/experiment_baseline_init_ea.py 
```

### Result Analysis:

Run interactively in the `src/data_analysis.ipynb`.

