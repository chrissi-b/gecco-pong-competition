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

### Run files

Files take different environment variables that have to be provided for successful execution:
- **optimization.py** no environment variables needed, runs with static variables that were used to generate the presented policy for demonstration purposes (see paper for evolution parameters). Running this file requires 12 available threads.
- **run_from_pickle.py** takes the folder, the checkpoint of optimization (modulo of 5), and the seed range that needs to be evaluated (start and end seed). The graph presented in the paper was generated after 1500 generations and can be tested as follows:
	- python3 run_from_pickle.py best_policy 1500 1 10
- **run_from_pseudocode.py** takes the seed range that needs to be evaluated (start and end seed). Pseudocode is generated based on the graph after 1500 generations:
	- python3 run_from_pseudocode.py 1 10

### Optimization log

Three different files can be used to verify the progression of the policy during the performed optimization
- **optimization_log.html** is an exported jupyter notebook that shows the convergence curve of the best policy, as well as other runs performed for this project
- **ga_metrics/pong/best_policy/metrics123.json** is the input that was used to plot the convergence curve in optimization_log.html
- **ga_metrics/pong/best_policy/log.out** and **ga_metrics/pong/best_policy/log.err** contains the raw output generated during optimization 


## Policy Breakdown

This is an example of the intermediary outputs of nodes (functions) in the graph.

### Inputs Seed 7 - Action 17

frame1 | frame2 | frame3 | frame4 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="/best-policy-screens/input1.png" width="400"> | <img src="/best-policy-screens/input2.png" width="400">  | <img src="/best-policy-screens/input3.png" width="400"> | <img src="/best-policy-screens/input4.png" width="400"> 


### Program for NOOP prediction
function | output
:-------------------------:|:-------------------------:
horizontal_argmax(frame3) | 32
horizontal_notmaskfromto(frame3, 0.0, 30.0) | <img src="/best-policy-screens/notmaskfromtoh.png" width="300">
opening(horizontal_notmaskfromto, horizontal_argmax) | <img src="/best-policy-screens/opening.png" width="300">
center_of_mass(frame2) | (31, 42)
argmin_position(opening) | (31, 1)
true_greatherthan(center_of_mass, argmin_position) | 0.0
**output_noop** | **0** 

### Program for DOWN prediction
function | output
:-------------------------:|:-------------------------:
erosion(frame3, 60.0) | <img src="/best-policy-screens/erosion.png" width="300">
relative_horizontal_notmaskfromto(erosion, 1.0, 2.0) | <img src="/best-policy-screens/rel_notmaskfromtoh.png" width="300">
reduce_maximum(relative_horizontal_notmaskfromto) | 0.3412
**output_down** | **0.3412** 

### Program for UP prediction
function | output
:-------------------------:|:-------------------------:
sobely_filter(frame3, 2.0) | <img src="/best-policy-screens/sobely.png" width="300">
reative_vertical_argmax(sobely_filter) | 0.6667
**output_up** | **0.6667** 
