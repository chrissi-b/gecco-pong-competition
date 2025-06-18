# gecco-pong-competition

## Install

This repository runs on Linux (Ubuntu) and MacOS. If you use Debian or Windows Subsystem for Linux (WSL), you might encounter some errors with the opencv library that will be installed in the following step - this code can help:
```shell
sudo apt update && sudo apt upgrade
sudo apt install libgl libglib2.0-0
```

Clone the repository and reate a Python virtual environment (must be at least Python 3.9.21).

```shell
git clone https://github.com/chrissi-b/gecco-pong-competition.git
cd gecco-pong-competition
python3 -m venv pongvenv
source pongvenv/bin/activate
pip3 install -r requirements.txt
```

Run the python file `setup.py` to create the Julia virtual environment within your python virtual environment. This might take a few minutes.

```shell
python3 setup.py
```

Move julia folders inside the Julia virtual environment.

```shell
cp -r juliafolders/* pongvenv/julia_env/
```

## Repository Content

### Run files

Th run files can be used to replicate the optimization process and test the policy performance. The policy is available as a serialized file (pickle) that was directly generated during optimization, and as pseudocode, which was created manually after policy generation and uses the exact same functions that were used to generate the graph. Running from pseudocode is faster than running from pickle. The first execution can take considerably longer than the subsequent ones.

Files take different environment variables that have to be provided for successful execution, however fallbacks are implemented:
- **optimization.py** takes the number of threads (population size) and the Random seed. Fallbacks are the variables that were used to generate the presented policy (12 threads, seed 123). The number of threads needs to be minimum 3. Running with fallbacks requires 12 available threads. 
	- with fallback: ```python3 optimization.py ```
	- without fallback: ```python3 optimization.py -t 12 -rs 123 ```
- **run_from_pickle.py** takes the folder, the checkpoint of optimization (modulo of 5 in range 5 to 1725), and the seed range that needs to be evaluated (start and end seed). The fallback values evaluates the graph presented in the paper (best_policy at generation 1500) on the environments 1 to 10:
	- with fallback: ```python3 run_from_pickle.py ```
	- without fallback: ```python3 run_from_pickle.py -p best_policy -cp 1500 -s 1 -e 10 ```
- **run_from_pseudocode.py** takes the seed range that needs to be evaluated (start and end seed). The pseudocode is generated based on the graph after 1500 generations, and the fallback values evaluates the graph on the environments 1 to 10:
	- with fallback: ```python3 run_from_pseudocode.py ```
	- without fallback: ```python3 run_from_pseudocode.py -s 1 -e 10```

#### Remarks 

You can run ```python3 {run_file} --help``` for more information on environment variables.

Running *optimization.py* will create a new folder with pickle files of the generated policies in the *ga_metrics/pong* folder. You can test your own graphs with the *run_from_pickle.py* file by defining -p {your run ID} and -cp the checkpoint you want to evaluate (modulo of 5, range depends on how long you run the optimization for). The number of generations defined in *optimization.py* is set to 2500, so finishing the entire evolution is costly. You can early stop with Ctrl+C and still have access to the generated pickles up to that point.

Example: ``` python3 run_from_pickle.py -p 287834d2-f9c9-4cac-8655-a2e51c8739ae -cp 5 -s 1 -e 5 ```

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


## GIFs

Below are two GIFs showing the first few points for seed 1 and 7 generated with the presented policy

S1 | S7 
:-------------------------:|:-------------------------:
![](/best-policy-screens/seed1.gif)  |  ![](/best-policy-screens/seed7.gif)
