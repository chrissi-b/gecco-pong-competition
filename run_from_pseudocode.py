# run file, i.e., a Python script, from which the submitted policy can be assessed on the environment

import os
import sys

os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'
os.environ['JULIA_NUM_THREADS'] = '1'

seed_start = int(sys.argv[1])
seed_end = int(sys.argv[2])

from juliacall import Main as jl
import numpy as np

jl.seval("""
         using InteractiveUtils
         println(versioninfo())
         """)


#seed_start = 1
#seed_end = 10

path = "pongvenv/julia_env/scripts/setup_function_libraries.jl"
with open(path) as f:
    juliacode = f.read()

module_path = '"pongvenv/julia_env/src/PongCompetition.jl"'
utils_path = f'"pongvenv/julia_env/scripts/utils.jl"'
extra_fns_path = f'"pongvenv/julia_env/scripts/extra_fns.jl"'

replace_tuples = [('"scripts", "utils.jl"', utils_path),
                  ('"scripts", "extra_fns.jl"', extra_fns_path),
                  ('"src/PongCompetition.jl"', module_path)
                 ]

juliacode_formatted = juliacode
for t in replace_tuples:
    juliacode_formatted = juliacode_formatted.replace(t[0], t[1])

jl.seval(f""" {juliacode_formatted} """)

jl.seval("""
	experimental_horizontal_argmax = ml[2][:experimental_horizontal_argmax].fn
	notmaskfromtoh_image2D = ml[1][:experimental_notmaskfromtoh_image2D].fn
	experimental_opening_image2D_factory = ml[1][:experimental_opening_image2D_factory].fn
	center_of_mass = ml[3][:center_of_mass].fn
	argmin_position = ml[3][:argmin_position].fn
	true_gt = ml[2][:true_gt].fn

	sobely_image2D = ml[1][:sobely_image2D].fn
	experimental_vertical_relative_argmax = ml[2][:experimental_vertical_relative_argmax].fn
	
	experimental_is_gt = ml[2][:experimental_is_gt].fn
	erosion_image2D_factory = ml[1][:erosion_image2D_factory].fn
	experimental_notmaskfromtoh_relative_image2D = ml[1][:experimental_notmaskfromtoh_relative_image2D].fn
	reduce_maximum = ml[2][:reduce_maximum].fn
	
	identity_float = ml[2][:identity_float].fn
""")


jl.seval("""
	function evolved_pong_policy(frame1, frame2, frame3, frame4)
	    
	    # output NO OP
	    h_argmax = experimental_horizontal_argmax(frame3) # 33,2
	    notmaskfromtoh_img = notmaskfromtoh_image2D(frame3, 0.0, 30.0) # 30,1
	    opening_img = experimental_opening_image2D_factory(notmaskfromtoh_img, h_argmax) # 40,1
	    center = center_of_mass(frame2) # 48,3
	    argmin = argmin_position(opening_img) # 57,3
	    tr_gt = true_gt(center, argmin) # 114,2 
	    output_noop = identity_float(tr_gt) # 115 2

	    # Output RIGHTFIRE (UP)     
	    sobely_img = sobely_image2D(frame3, 2.0) # 25,1    
	    v_rel_argmax = experimental_vertical_relative_argmax(sobely_img) # 113,2
	    output_up = identity_float(v_rel_argmax) # 116 2

	    # Output LEFTFIRE (DOWN)
	    is_gt1 = experimental_is_gt(50.0, 0.0) # 28,2
	    is_gt2 = experimental_is_gt(20.0, is_gt1) # 39,2
	    erosion_img = erosion_image2D_factory(frame3, 60.0) # 20,1
	    notmaskfromtoh_rel_img = experimental_notmaskfromtoh_relative_image2D(erosion_img, is_gt2, 2.0) # (58,1)
	    reduce_max = reduce_maximum(notmaskfromtoh_rel_img) # 112,2)
	    output_down = identity_float(reduce_max) # 117 2

	    outputs = (output_noop, output_up, output_down)
	    return ACTIONS[argmax(outputs)]

	end
""")

jl.seval("""
	function get_final_score_per_seed(env)

	    obs_, rew_, term_, trunc_, info_ = PongCompetition.step([env], [0]);

	    # Initialise variables
	    state_view = view(obs_, [1])
	    previous_obs = state_view
	    
	    s = size(previous_obs[1][1, CROP[1], CROP[2]])
	    S = Tuple{s[1],s[2]}

	    # PLAY
	    num_actions = 10000
	    
	    actions = []
	    
	    reward_total = 0
	    points_own = 0
	    points_own_idx = []
	    points_opponent = 0
	    points_opponent_idx = []
	    termination_point = 0
	    
	    for i in 1:num_actions
	    
		inputs = previous_obs[1]
		ins = Any[SImageND(inputs[t, CROP[1], CROP[2]], S) for t in axes(inputs, 1)]
	    
		action = evolved_pong_policy(ins[1], ins[2], ins[3], ins[4])
		push!(actions, action)
	    
		obs_, rew_, term_, trunc_, info_ = PongCompetition.step([env], [action]);
	    
		if rew_[1] == 1
		    reward_total = reward_total + 1
		    points_own = points_own + 1
		    push!(points_own_idx, i)
		elseif rew_[1] == -1
		    reward_total = reward_total - 1
		    points_opponent = points_opponent + 1
		    push!(points_opponent_idx, i)
		end
		
		if term_[1] == true || trunc_[1] == true
		    termination_point = i
		    break
		end
	    
		state_view = view(obs_, [1])
		previous_obs = state_view
		s = size(previous_obs[1][1, CROP[1], CROP[2]])
		S = Tuple{s[1],s[2]}
	    end

	    return termination_point, reward_total, points_own, points_own_idx, points_opponent, points_opponent_idx, actions
	end
""")

jl.seval("""
	termination_points = []
	rewards_total = []
	points_own = []
	points_own_idx = []
	points_opponent = []
	points_opponent_idx = []
	actions = []
""")

jl.seval(f"""
	seeds = [x for x in {seed_start}:{seed_end}]
	envs = pool(GAMENAME, seeds)

	for env in envs
	    tp, rew, p_ow, p_ow_idx, p_op, p_op_idx, acts = get_final_score_per_seed(env)

	    push!(termination_points, tp)
	    push!(rewards_total, rew)
	    push!(points_own, p_ow)
	    push!(points_own_idx, p_ow_idx)
	    push!(points_opponent, p_op)
	    push!(points_opponent_idx, p_op_idx)
	    push!(actions, acts)
	end
""")

jl.seval(f"""
	using DataFrames
	
	data_final_results = DataFrame(
	    "seed" => {seed_start}:{seed_end},
	    "termination_point" => termination_points,
	    "reward_total" => rewards_total,
	    "points_own" => points_own,
	    "points_own_idx" => points_own_idx,
	    "points_opponent" => points_opponent,
	    "points_opponent_idx" => points_opponent_idx
	)
	
	println("Unique values own points: $(unique(data_final_results.points_own))")
	println("Unique values opponent points: $(unique(data_final_results.points_opponent))")
	println("# of games won: $(size(filter(row -> row.points_own == 21, data_final_results))[1])")
	println("# of games lost: $(size(filter(row -> row.points_opponent == 21, data_final_results))[1])")
	
	println("Mean reward: $(mean(data_final_results.reward_total))")
	println("Std reward: $(round(std(data_final_results.reward_total), sigdigits=3))")
	println("Mean own points: $(mean(data_final_results.points_own))")
	println("Std own points: $(round(std(data_final_results.points_own), sigdigits=3))")
	println("Mean termination point: $(mean(data_final_results.termination_point))")
	println("Std termination point: $(round(std(data_final_results.termination_point), sigdigits=3))")

""")



