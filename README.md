# gecco-pong-competition

## Install

Clone the repository

Create a Python virtual environment (must be at least Python 3.9.21):

```shell
git clone https://github.com/chrissi-b/gecco-pong-competition.git
cd gecco-pong-competition
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

## Policy Breakdown

This is an example of the intermediary outputs of nodes (functions) in the graph.

### Inputs Seed 7 - Action 17

frame1 | frame2 | frame3 | frame4 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](TODO add img) | ![](TODO add img) | ![](TODO add img) | ![](TODO add img) 


### Program for NOOP prediction
function | output
:-------------------------:|:-------------------------:
horizontal_argmax(frame3) | 32
horizontal_notmaskfromto(frame3, 0.0, 30.0) | TODO add img
opening(horizontal_notmaskfromto, horizontal_argmax) | TODO add img
center_of_mass(frame2) | (31, 42)
argmin_position(opening) | (31, 1)
true_greatherthan(center_of_mass, argmin_position) | 0.0
**output_noop** | **0** 

### Program for DOWN prediction
function | output
:-------------------------:|:-------------------------:
erosion(frame3, 60.0) | TODO add img
relative_horizontal_notmaskfromto(erosion, 1.0, 2.0) | TODO add value
reduce_maximum(relative_horizontal_notmaskfromto) | TODO add value
**output_down** | **TODO** 

### Program for UP prediction
function | output
:-------------------------:|:-------------------------:
sobely_filter(frame3, 2.0) | TODO add img
reative_vertical_argmax(sobely_filter) | TODO add value
**output_up** | **TODO** 
