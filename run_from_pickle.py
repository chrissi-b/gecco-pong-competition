# run file, i.e., a Python script, from which the submitted policy can be assessed on the environment

import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='gecco-pong-competition')

parser.add_argument('-p', '--policy', default='best_policy', required=False,
                    help='Folder name of policy.')
                    
parser.add_argument('-cp', '--checkpoint', default = '1500', required=False,
                    help='Checkpoint to be evaluated. Must be a modulo of 5 in the range 5 to 1725 when evaluating best_policy.')
                    
parser.add_argument('-s', '--startseed', default = '1', required=False,
                    help='Start of seed range to be evaluated.')
                    
parser.add_argument('-e', '--endseed', default = '10', required=False,
                    help='End of seed range to be evaluated. Must be greather or equal to startseed.')                   
    
args = parser.parse_args()

folder = args.policy
checkpoint = int(args.checkpoint)
seed_start = int(args.startseed)
seed_end = int(args.endseed)

print(f"Getting payload for folder {folder} at checkpoint {checkpoint}")

os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'
os.environ['JULIA_NUM_THREADS'] = '1'

from juliacall import Main as jl
import numpy as np

jl.seval("""
         using InteractiveUtils
         println(versioninfo())
         """)

path = "pongvenv/julia_env/scripts/run_pong_short.jl"
with open(path) as f:
    juliacode = f.read()

seed_change_threshold = '"2501"'
random_seed = '"123"'
module_path = '"pongvenv/julia_env/src/PongCompetition.jl"'
utils_path = f'"pongvenv/julia_env/scripts/utils.jl"'
extra_fns_path = f'"pongvenv/julia_env/scripts/extra_fns.jl"'
nt = "12"

replace_tuples = [("ARGS[1]", seed_change_threshold),
                  ("ARGS[2]", random_seed),
                  ('"scripts", "utils.jl"', utils_path),
                  ('"scripts", "extra_fns.jl"', extra_fns_path),
                  ("nthreads()", nt),
                  ('"src/PongCompetition.jl"', module_path)
                 ]


juliacode_formatted = juliacode
for t in replace_tuples:
    juliacode_formatted = juliacode_formatted.replace(t[0], t[1])


jl.seval(f""" {juliacode_formatted} """)

jl.seval(f"""
    using Serialization
    load_folder = "ga_metrics/pong/{folder}/checkpoint_{checkpoint}.pickle"
    payloads = deserialize(load_folder);
    best_program = UTCGP.decode_with_output_nodes(payloads["best_genome"], payloads["ml"], model_arch, shared_inputs)
"""
)

jl.seval(f"""
            F = []; ST = []; INDS = []; ACTIONS_ = []; OWN_POINTS = []; OPP_POINTS = []
                for S in {seed_start}:{seed_end}
                    f, s, inds, actions, own_points, opp_points = fixed_game_without_unlock(
                            deepcopy(best_program),
                            S, model_arch, payloads["ml"], 1)
                    println("Seed: $S")
                    println("Fitness: $f")
                    @show f
                    push!(F, f)
                    push!(ST, s)
                    push!(INDS, inds)
                    push!(ACTIONS_, actions)
                    push!(OWN_POINTS, own_points)
                    push!(OPP_POINTS, opp_points)
                end
            """)


fitnesses = [round(p[0][0]) for p in jl.F]
own_points = [round(p[0][0]) for p in jl.OWN_POINTS]
opp_points = [round(p[0][0]) for p in jl.OPP_POINTS]

print("Fitness all seeds after a max of 30 000 frames:")
print(fitnesses)
print("Own points all seeds after a max of 30 000 frames:")
print(own_points)
print("Opponent points all seeds after a max of 30 000 frames:")
print(opp_points)
print("Termination points all seeds:")
jl.seval('[i[end]["it"] for i in ST]')


print(f"Unique values own points: {list(set(own_points))}")
print(f"Unique values opponent points: {list(set(opp_points))}")
print(f"# of games won: {len([p for p in own_points if p == 21])}")
print(f"# of games lost: {len([p for p in own_points if p != 21])}")
print(f"Mean reward (the lower the better): {float(np.mean(fitnesses))}")
print(f"Std reward: {float(np.std(fitnesses))}")
print(f"Mean own points: {float(np.mean(own_points))}")
print(f"Std own points: {float(np.std(own_points))}")
print(f"Mean opponent points: {float(np.mean(opp_points))}")
print(f"Std own points: {float(np.std(opp_points))}")



