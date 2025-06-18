# optimization file, i.e., a Python script, from which the optimization process can be reproduced

import os
import argparse

parser = argparse.ArgumentParser(description='gecco-pong-competition')
parser.add_argument('-t', '--threads', default='12', required=False,
                    help='Number of threads, needs to be minimum 3.')
                    
parser.add_argument('-rs', '--randomseed', default = '123', required=False,
                    help='Seed for Random generator.')                 
    
args = parser.parse_args()

os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'
os.environ['JULIA_NUM_THREADS'] = args.threads

from juliacall import Main as jl

jl.seval("""
         using InteractiveUtils
         println(versioninfo())
         """)

path = "pongvenv/julia_env/scripts/run_pong.jl"
with open(path) as f:
    juliacode = f.read()

seed_change_threshold = '"2501"'
random_seed = f'"{args.randomseed}"'
module_path = '"pongvenv/julia_env/src/PongCompetition.jl"'
utils_path = f'"pongvenv/julia_env/scripts/utils.jl"'
extra_fns_path = f'"pongvenv/julia_env/scripts/extra_fns.jl"'

replace_tuples = [("ARGS[1]", seed_change_threshold),
                  ("ARGS[2]", random_seed),
                  ('"scripts", "utils.jl"', utils_path),
                  ('"scripts", "extra_fns.jl"', extra_fns_path),
                  ('"src/PongCompetition.jl"', module_path)
                  
                 ]

juliacode_formatted = juliacode
for t in replace_tuples:
    juliacode_formatted = juliacode_formatted.replace(t[0], t[1])
print(f"Formatted juliacode: {juliacode_formatted}")

jl.seval(f""" {juliacode_formatted} """)
