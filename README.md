# pong-competition

## Install

Clone the repository

Create a Python virtual environment (must be at least Python 3.9.21):

```shell
git clone https://github.com/chrissi-b/pong-competition.git
cd pong-competition
python3 -m venv pongvenv
source pongvenv/bin/activate
pip3 install -r requirements.txt
```

Run the python file `setup.py` to create the Julia virtual environment within your python virtual environment

```shell
python3 setup.py
```

Move julia folders inside the Julia virtual environment

```shell
cp -r juliafolders/* pongvenv/julia_env/
```

## Repository Content

## Run files

Files take different environment variables that have to be provided for successful execution:
- **optimization.py** no environment variables needed, runs with static variables for demonstration purposes
- **run_from_pickle.py** takes the folder, the checkpoint of optimization (modulo of 5), and the seed range that needs to be evaluated (start and end seed):
	- python3 run_from_pickle.py best_fitness 1500 1 10
- **run_from_pseudocode.py** takes the seed range that needs to be evaluated (start and end seed):
	- python3 run_from_pseudocode.py 1 10
