using ErrorTypes
using UTCGP
using UTCGP: Optional_FN, Mandatory_FN, PopulationPrograms
using Random
using LinearAlgebra
using FileIO

##########################################
# MISC
##########################################

@inline function relu(x)
    x > 0 ? x : 0
end

function protected_norm(x::Vector{Float64})
    nrm = norm(x)
    if nrm > 0
        return nrm
    else
        return 1.0
    end
end

##########################################
# ACTIONS
##########################################
abstract type AbstractProb end
struct prob <: AbstractProb end
struct notprob <: AbstractProb end

function pong_action_mapping(outputs::Vector, p::Type{notprob})
    global ACTIONS
    ACTIONS[argmax(outputs)]
end
function pong_action_mapping_one_out(outputs::Vector, p::Type{notprob})
    if outputs[1] > 0
        return 4
    end
    return 5
end
function pong_action_mapping(outputs::Vector, p::Type{prob}, mt)
    global ACTIONS
    os = relu.(outputs) .* 100
    w = Weights(os)
    action = sample(mt, ACTIONS, w)
    action
end

##########################################
# Evaluate INDS
##########################################
function get_tape_from_ind(ind_prog::UTCGP.IndividualPrograms)
    n_outputs = length(ind_prog)
    tape = Vector{Float64}[]
    for (ith_prog, program) in enumerate(ind_prog)
        program_tape = Float64[]
        for (ith_op, op) in enumerate(program)
            if !isnothing(lib_number_to_img[op.fn.name])
                push!(program_tape, op.calling_node.value)
            end
        end
        push!(tape, program_tape)
    end
    tape
end

"""
puts in matrix form (frames, ops)
"""
function tapes_to_matrix(tape::Vector{Vector{Float64}})
    r, c = length(tape), length(tape[begin])
    pre_outputs = Matrix{Float64}(undef, r, c)
    for (i, outputs_in_one_prog) in enumerate(tape)
        pre_outputs[i, :] .= outputs_in_one_prog
    end
    pre_outputs
end
function tapes_to_matrices(tapes::Vector{Vector{Vector{Float64}}})
    example = tapes[1]
    frames, n_outputs = length(tapes), length(example) # constants
    Matrices = Matrix{Float64}[]
    for i in 1:n_outputs
        n_prelim_outputs = length(example[i])
        pre_outputs = Matrix{Float64}(undef, frames, n_prelim_outputs)
        push!(Matrices, pre_outputs)
    end
    for (ith_tape, tape) in enumerate(tapes)
        # each tape => the outputs from a single frame eval
        for (i, outputs_in_one_prog) in enumerate(tape)
            Matrices[i][ith_tape, :] .= outputs_in_one_prog
        end
    end
    Matrices
end

function evaluate_with_tape_individual_programs!(
    ind_prog::UTCGP.IndividualPrograms,
    buffer,
    model_arch::modelArchitecture,
    meta_library::MetaLibrary
)
    in_types = model_arch.inputs_types_idx
    UTCGP.reset_programs!(ind_prog)
    push!(buffer, -1.0, 0.5, 2.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    input_nodes = [
        InputNode(value, pos, pos, in_types[pos]) for
        (pos, value) in enumerate(buffer)
    ]
    UTCGP.replace_shared_inputs!(ind_prog, input_nodes)
    # Eval
    outputs = UTCGP.evaluate_individual_programs(
        ind_prog,
        model_arch.chromosomes_types,
        meta_library,
    )
    # Get Tape
    tape = get_tape_from_ind(ind_prog)
    outputs, tape
end

function evaluate_individual_programs!(
    ind_prog::UTCGP.IndividualPrograms,
    buffer,
    model_arch::modelArchitecture,
    meta_library::MetaLibrary
)
    in_types = model_arch.inputs_types_idx
    UTCGP.reset_programs!(ind_prog)
    push!(buffer, -1.0, 0.5, 2.0, 0.0)
    input_nodes = [
        InputNode(value, pos, pos, in_types[pos]) for
        (pos, value) in enumerate(buffer)
    ]
    UTCGP.replace_shared_inputs!(ind_prog, input_nodes)

    # Eval
    outputs = UTCGP.evaluate_individual_programs(
        ind_prog,
        model_arch.chromosomes_types,
        meta_library,
    )
    UTCGP.reset_programs!(ind_prog)
    outputs
end


# function atari_fitness(ind::UTCGP.IndividualPrograms, ind_idx::Int, seed::Int, model_arch::modelArchitecture, meta_library::MetaLibrary)
#     # TODO remove this line
#     seed = 1
#     Random.seed!(seed)
#     global GAME
#     global ATARI_LOCK
#     global PROB_ACTION
#     global DIMOVER
#     global NIMAGES
#     global GAMES
#     # game = AtariEnv(GAME, seed, ATARI_LOCK)
#     game = GAMES[ind_idx]
#     mt = MersenneTwister(seed)
#     MAGE_ATARI.reset!(game)
#     # max_frames = 300
#     max_frames = clamp(NIMAGES[], 10, 5000)
#     stickiness = 0.25
#     reward = 0.0
#     frames = 0
#     prev_action = Int32(0)
#     actions_counter = Dict(
#         [action => 0 for action in ACTIONS]...
#     )
#     action_changes = 0
#     output_sequence = Vector{Vector{Float64}}(undef, 0)
#     q = Queue{IMAGE_TYPE}()
#     TAPES = Vector{Vector{Float64}}[]
#     @info "GAME BEGIN at $(Threads.threadid())"
#     start = time()
#     while ~game_over(game.ale)
#         if rand(mt) > stickiness || frames == 0
#             MAGE_ATARI.update_screen(game)
#             current_gray_screen = N0f8.(Gray.(game.screen))
#             cur_frame = SImageND(current_gray_screen)
#             if isempty(q)
#                 enqueue!(q, cur_frame)
#                 enqueue!(q, cur_frame)
#                 enqueue!(q, cur_frame)
#                 enqueue!(q, cur_frame)
#             end
#             dequeue!(q) # removes the first
#             enqueue!(q, cur_frame) # adds to last
#             v = [i for i in q]
#             o_copy = [v[1],
#                 v[2],
#                 v[3],
#                 v[4],
#                 0.1, -0.1, 2.0, -2.0]
#             outputs, tape = evaluate_with_tape_individual_programs!(ind, o_copy, model_arch, meta_library)
#             push!(TAPES, tape)
#             push!(output_sequence, outputs)

#             # action = ACTION_MAPPER(outputs)
#             if PROB_ACTION
#                 action = ACTION_MAPPER(outputs, prob, mt)
#             else
#                 action = ACTION_MAPPER(outputs, notprob)
#             end

#         else
#             action = prev_action
#         end
#         # reward += act(game.ale, action)
#         r, s = MAGE_ATARI.step!(game, game.state, action)
#         if action != prev_action
#             action_changes += 1
#         end
#         actions_counter[action] += 1
#         reward += r
#         frames += 1
#         prev_action = action
#         if frames > max_frames
#             # @info "Game finished because of max_frames"
#             break
#         end
#     end
#     elapsed = time() - start
#     @info "GAME END at $(Threads.threadid()) : $elapsed"
#     # MAGE_ATARI.close(game, ATARI_LOCK)
#     tape_matrices = tapes_to_matrices(TAPES) # [(frames, ops), ...]
#     variance_per_matrix = [mean(std(m; dims=DIMOVER)) for m in tape_matrices]
#     variance_over_all_episodes = mean(filter(!isnan, variance_per_matrix))
#     variance_over_all_episodes = isnan(variance_over_all_episodes) ? -100 : variance_over_all_episodes

#     # calculate the descriptors
#     # tot = sum(values(actions_counter))
#     # action_tot = sum([actions_counter[a] for a in ACTIONS[2:end]])
#     # activity_descriptor = action_tot / tot
#     # action_changes_descriptor = action_changes / tot
#     first_objective = reward * -1.0

#     # TODO minimize the inverse of the average variance of image to float fns in the graph (variance over the episode = game) 
#     # for now we maximize the variance of each output over the episode
#     # note: we first divide the vector by its norm to ensure coherence across the values
#     # transposed_output_signals = [[v[i] for v in output_sequence] for i = 1:length(output_sequence[1])]
#     # variances = [var(v / protected_norm(v)) for v in transposed_output_signals]
#     # second_objective = -mean(variances)
#     # if isnan(second_objective)
#     # second_objective = 0.0
#     # end
#     return [first_objective, variance_over_all_episodes * -1.0]
# end

function step_pop!(
    population::SubArray{IndividualPrograms},
    actions::Vector{<:Int},
    model_arch::modelArchitecture,
    ml::MetaLibrary, generation::Int, it::Int,
    thread_partitions::Iterators.PartitionIterator,
    mts::SubArray{MersenneTwister},
    previous_obs::SubArray{N0f8,4,Array{N0f8,4}},#
    TAPES
)
    @debug "GAME Step $it"
    start = now()
    global PROB_ACTION, CROP
    global ACTION_MAPPER
    # global DIMOVER
    # global NIMAGES
    tasks = Vector{Task}(undef, length(thread_partitions))
    # x = pyconvert(Array, previous_obs)
    s = size(previous_obs[1, 1, CROP[1], CROP[2]])
    S = Tuple{s[1],s[2]}

    for (ithread, ith_x) in enumerate(thread_partitions)
        t = @spawn begin
            @debug "Spawn the task of $ith_x to thread $(threadid())"
            for i in ith_x
                ind_programs = deepcopy(population[i])
                mt = mts[i]
                inputs = previous_obs[i, :, :, :] # T, W, H
                ins = Any[SImageND(inputs[t, CROP[1], CROP[2]], S) for t in axes(inputs, 1)]
                outputs, tape = evaluate_with_tape_individual_programs!(ind_programs,
                    ins, model_arch, ml)
                push!(TAPES[i], reduce(vcat, tape))
                if PROB_ACTION
                    action = ACTION_MAPPER(outputs, prob, mt)
                else
                    action = ACTION_MAPPER(outputs, notprob)
                end
                actions[i] = action
            end
        end
        tasks[ithread] = t
    end
    fetch.(tasks)
    ended = now()
    @debug "GAME Step End $(ended - start)"
end

function step_pop_st!(
    population::SubArray{IndividualPrograms},
    actions::Vector{<:Int},
    model_arch::modelArchitecture,
    ml::MetaLibrary, generation::Int, it::Int,
    thread_partitions::Iterators.PartitionIterator,
    mts::SubArray{MersenneTwister},
    previous_obs::SubArray,#
    TAPES
)
    @debug "GAME Step $it"
    start = now()
    global PROB_ACTION, CROP
    global ACTION_MAPPER
    s = size(previous_obs[1][1, CROP[1], CROP[2]])
    S = Tuple{s[1],s[2]}
    for (ithread, ith_x) in enumerate(thread_partitions)
        @debug "Spawn the task of $ith_x to thread $(threadid())"
        for i in ith_x
            ind_programs = deepcopy(population[i])
            mt = mts[i]
            inputs = previous_obs[i] # T, W, H
            ins = Any[SImageND(inputs[t, CROP[1], CROP[2]], S) for t in axes(inputs, 1)]
            outputs, tape = evaluate_with_tape_individual_programs!(ind_programs,
                ins, model_arch, ml)
            push!(TAPES[i], reduce(vcat, tape))
            if PROB_ACTION
                action = ACTION_MAPPER(outputs, prob, mt)
            else
                action = ACTION_MAPPER(outputs, notprob)
            end
            actions[i] = action
        end
    end
    ended = now()
    @debug "GAME Step End $(ended - start)"
end

function step_pop_st_and_log!(
    population::SubArray{IndividualPrograms},
    actions::Vector{<:Int},
    model_arch::modelArchitecture,
    ml::MetaLibrary, generation::Int, it::Int,
    thread_partitions::Iterators.PartitionIterator,
    mts::SubArray{MersenneTwister},
    previous_obs::SubArray,#
    TAPES,
    INDS
)
    @debug "GAME Step $it"
    start = now()
    global PROB_ACTION, CROP
    global ACTION_MAPPER
    s = size(previous_obs[1][1, CROP[1], CROP[2]])
    S = Tuple{s[1],s[2]}
    for (ithread, ith_x) in enumerate(thread_partitions)
        @debug "Spawn the task of $ith_x to thread $(threadid())"
        for i in ith_x
            ind_programs = deepcopy(population[i])
            mt = mts[i]
            inputs = previous_obs[i] # T, W, H
            ins = Any[SImageND(inputs[t, CROP[1], CROP[2]], S) for t in axes(inputs, 1)]
            outputs, tape = evaluate_with_tape_individual_programs!(ind_programs,
                ins, model_arch, ml)
            push!(INDS, deepcopy(ind_programs))
            push!(TAPES[i], reduce(vcat, tape))
            if PROB_ACTION
                action = ACTION_MAPPER(outputs, prob, mt)
            else
                action = ACTION_MAPPER(outputs, notprob)
            end
            actions[i] = action
        end
    end
    ended = now()
    @debug "GAME Step End $(ended - start)"
end

function resolve_partitions_for_the_round(n_ind_to_eval::Int, nthreads::Int)
    indices = collect(1:n_ind_to_eval)
    BS = ceil(Int, n_ind_to_eval / nthreads)
    Iterators.partition(indices, BS)
end

function fixed_game(program, seed, model_arch, meta_library, generation, frames::Int=30_000)
    global NP, GAMENAME
    NSEEDS = 1
    pop = UTCGP.PopulationPrograms([program])
    n, nt = length(pop), 1 #nthreads()
    seeds_for_envs = [seed]
    local ENVS
    PongCompetition.pycall_lock() do
        pyexec("import numpy as np; np.random.seed(0)", PongCompetition)
        ENVS = pool_of_pools(GAMENAME, n, [seed], frames) # N pools with seeds
        PongCompetition.reset(ENVS, [seed])
    end

    UTCGP.reset_programs!.(pop)
    pop_fits = Vector{Vector{Float64}}(undef, n)
    tasks = []
    pop_partitions = resolve_partitions_for_the_round(n, nt)
    STATES = []
    INDS = []
    ACTIONS_ = []
    for (ith_partition, pop_partition) in enumerate(pop_partitions)
        t = @spawn begin
            tid = threadid()
            for (ith_program) in pop_partition
                ind_program = pop[ith_program]
                ENVS_IND = ENVS[ith_program] # is a pool with seeds
                println("Running ind n° $ith_program")
                local state, rew, term, truncated, info
                local still_playing
                PongCompetition.pycall_lock() do
                    PongCompetition.reset(ENVS_IND, seeds_for_envs)
                    state, rew, term, truncated, info = PongCompetition.step(ENVS_IND, zeros(Int, NSEEDS))
                end
                still_playing = .~(term .|| truncated)
                indices_jl = collect(1:NSEEDS)
                players_py = collect(0:NSEEDS-1)
                ACTIONS = [[] for i in 1:NSEEDS]
                TAPES = [[] for i in 1:NSEEDS]

                mts = identity.([MersenneTwister(i) for i in 1:NSEEDS])
                it = 0
                pop_for_ind = PopulationPrograms([deepcopy(ind_program) for i in 1:NSEEDS])
                local ind_fitness = Vector{Float64}(undef, NSEEDS)
                R = []
                while true
                    n_playing = sum(still_playing)
                    actions_for_players_playing = Vector{Int}(undef, n_playing)
                    state_view = view(state, still_playing)
                    nstates = size(state_view, 1)
                    @assert nstates == n_playing == length(indices_jl) == length(players_py) "N states : $nstates vs N playing : $n_playing Indices : $(length(players_py))"
                    partitions = resolve_partitions_for_the_round(n_playing, nt)
                    step_pop_st_and_log!(
                        view(pop_for_ind.population_programs, indices_jl),
                        actions_for_players_playing,
                        model_arch, meta_library, generation, it, partitions,
                        view(mts, indices_jl), state_view, TAPES, INDS)
                    push!(STATES, state_view)
                    PongCompetition.pycall_lock() do
                        state, rew, term, truncated, info = PongCompetition.step(ENVS_IND, actions_for_players_playing,
                            indices_jl)
                        # @show info
                    end

                    # Reward
                    # push!(R, deepcopy(rew))
                    @assert length(rew) == length(indices_jl) == n_playing
                    for (i, ind_index) in enumerate(indices_jl)
                        r = rew[i]
                        # r = isnan(r) ? 100 : r
                        if it == 0
                            ind_fitness[ind_index] = r
                        else
                            ind_fitness[ind_index] += r
                        end
                    end

                    # determine who is still playing
                    still_playing = .~(term .|| truncated)
                    if sum(still_playing) == 0
                        @info "Breaking cause all in time limit or game over"
                        println("last reward : $rew")
                        # @info term
                        # @info truncated
                        # if isdefined(Main, :Infiltrator)
                        #     Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
                        # end
                        # println(info)
                        break
                    end

                    players_py = players_py[still_playing]
                    indices_jl = indices_jl[still_playing]

                    # save actions 
                    for (i, idx) in enumerate(indices_jl)
                        push!(ACTIONS[idx], actions_for_players_playing[i])
                    end
                    it += 1
                end

                # local entropy
                # entropies = Float64[]
                # try
                #     for tape in TAPES # loop over diff seeds
                #         m = tapes_to_matrix(identity.(tape))
                #         for col_idx in 1:size(m, 2) # loop over ops
                #             try
                #                 h = StatsBase.fit(StatsBase.Histogram, m[:, col_idx], nbins=50)
                #                 push!(entropies, StatsBase.entropy(h.weights / length(m[:, col_idx])))
                #             catch
                #                 push!(entropies, 0.0)
                #             end
                #         end
                #     end
                #     replace!(entropies, NaN => 0.0)
                #     entropy = mean(entropies)
                #     entropy = isnan(entropy) ? 0.0 : entropy
                # catch
                #     entropy = 0.0
                # end
                # @show ind_fitness

                # Fitness per seed
                F = Float64[]
                for s in 1:NSEEDS
                    f = ind_fitness[s]
                    e = try
                        entropies[s]
                    catch
                        0.0
                    end
                    # e = clamp(e, 0.0, 0.99) # A point is always better than e
                    push!(F, f * -1.0 + e * -1.0)
                end
                push!(ACTIONS_, ACTIONS...)
                pop_fits[ith_program] = [mean(F)]
            end
        end
        push!(tasks, t)
    end
    fetch.(tasks)

    GC.gc(true)
    PongCompetition.pycall_lock() do
        for pool in ENVS
            for env in pool
                env.close()
            end
        end
    end
    GC.gc(true)
    ENVS = nothing
    GC.gc(true)
    PongCompetition.gc_()
    PythonCall.GC.gc()
    GC.gc(true)
    return pop_fits, STATES, INDS, ACTIONS_
end

function step_wise_frames!(ENVS, generation)
    global ENVS, ENVS2, ENVS3
    if generation < 500
        @info "Using 600 frames"
    elseif generation == 500
        empty!(ENVS)
        push!(ENVS, ENVS2...)
        @info "Using 2_000 frames for the first time"
    elseif generation > 500 && generation < 1500
        @info "Using 2_000 frames"
    elseif generation == 1500
        empty!(ENVS)
        push!(ENVS, ENVS3...)
        @info "Using 36_000 frames for the first time"
    else
        @info "Using 36_000 frames"
    end
end

function step_wise_frames!(pop_size, generation)
    #global GAMENAME, CUSTOM_SEEDS
    global GAMENAME, CURRENT_SEEDS
    local frames
     if generation < 500
        @info "Using 600 frames on seeds $CURRENT_SEEDS"
        frames = 600
    elseif generation >= 500 && generation < 1500
        @info "Using 2_000 frames on seeds $CURRENT_SEEDS"
        frames = 2000
    else
        #@info "Using 36_000 frames on seeds $CURRENT_SEEDS"
        #frames = 36_000
        @info "Using 18_000 frames on seeds $CURRENT_SEEDS"
        frames = 18_000
    end
    #frames = 18_000
    #@info "Using 18_000 frames manually changed"

    ENVS = PongCompetition.pycall_lock() do
        #pool_of_pools(GAMENAME, pop_size, CUSTOM_SEEDS, frames)
        pool_of_pools(GAMENAME, pop_size, CURRENT_SEEDS, frames)
    end
    ENVS
end

function modify_seeds_half!(generation)
    global CURRENT_SEEDS, SEED_CHANGE_THRESHOLD

    if (generation == 1 || generation % SEED_CHANGE_THRESHOLD == 0)
        CURRENT_SEEDS = [i <= ceil(Int, NSEEDS/2) ? rand(SEEDS_LEFT) : rand(SEEDS_RIGHT) for i in 1:NSEEDS]
        @info "New CURRENT_SEEDS: of type $(typeof(CURRENT_SEEDS))"
        return CURRENT_SEEDS
    end
end

function modify_seeds_quarter!(generation)
    global CURRENT_SEEDS, SEED_CHANGE_THRESHOLD
    if NSEEDS < 4
        @info "Number of seeds to small for quarterly seed randomization, fallback to modify_seeds_half!()"
        modify_seeds_half!(generation)
    else
        if (generation == 1 || generation % SEED_CHANGE_THRESHOLD == 0)
            if length(CURRENT_SEEDS) % 4 > 0
                @info "Consider modifying the number of seeds to a multiple of 4 to ensure balanced training"
            end
            
            half = ceil(Int, NSEEDS/2)
            quarter = ceil(Int, half/2)
            third = ceil(Int, (NSEEDS-half)/2 + half)
            
            for i in 1:NSEEDS
                if i <= quarter
                    CURRENT_SEEDS[i] = rand(SEEDS_UPPERLEFT)
                elseif i <= half
                    CURRENT_SEEDS[i] = rand(SEEDS_LOWERLEFT)
                elseif i <= third
                    CURRENT_SEEDS[i] = rand(SEEDS_UPPERRIGHT)
                else
                    CURRENT_SEEDS[i] = rand(SEEDS_LOWERRIGHT)
                end
            end
            @info "New CURRENT_SEEDS: $CURRENT_SEEDS of type $(typeof(CURRENT_SEEDS))"
        end
    end
    return CURRENT_SEEDS
end

struct EnvpoolAtariEndpoint <: UTCGP.BatchEndpoint
    fitness_results::Vector{Vector{Float64}}
    function EnvpoolAtariEndpoint(
        pop::UTCGP.PopulationPrograms,
        model_arch::modelArchitecture,
        meta_library::MetaLibrary,
        generation::Int
    )
        global NP, CUSTOM_SEEDS, NSEEDS, GAMENAME, CURRENT_SEEDS, SEEDS_UPPERLEFT, SEEDS_LOWERLEFT, SEEDS_LEFT, SEEDS_UPPERRIGHT, SEEDS_LOWERRIGHT, SEEDS_RIGHT
        # global ENVS
        n, nt = length(pop), nthreads()
        # seeds_for_envs = CUSTOM_SEEDS
        seeds_for_envs = modify_seeds_quarter!(generation)
        ENVS = step_wise_frames!(n, generation) # needs to be moved after seed modification because it refers to global var CURRENT_SEEDS
        @info "Pools created"
        @info "Var seeds_for_envs: $seeds_for_envs"
        UTCGP.reset_programs!.(pop)
        pop_fits = Vector{Vector{Float64}}(undef, n)
        tasks = []
        pop_partitions = resolve_partitions_for_the_round(n, nt)
        @show collect(pop_partitions)
        for (ith_partition, pop_partition) in enumerate(pop_partitions)
            t = @spawn begin
                tid = threadid()
                for (ith_program) in pop_partition
                    ind_program = pop[ith_program]
                    ENVS_IND = ENVS[ith_program] # is a pool with seeds
                    println("Running ind n° $ith_program")
                    local state, rew, term, truncated, info
                    local still_playing
                    #@info "NOW TWO VALUES"
                    #println(ENVS_IND)
                    #println(seeds_for_envs)
                    PongCompetition.pycall_lock() do
                        PongCompetition.reset(ENVS_IND, seeds_for_envs)
                        state, rew, term, truncated, info = PongCompetition.step(ENVS_IND, zeros(Int, NSEEDS))
                    end
                    still_playing = .~(term .|| truncated)
                    indices_jl = collect(1:NSEEDS)
                    players_py = collect(0:NSEEDS-1)
                    local frames_per_seed = zeros(NSEEDS)
                    ACTIONS = [[] for i in 1:NSEEDS]
                    TAPES = [[] for i in 1:NSEEDS]

                    mts = identity.([MersenneTwister(i) for i in 1:NSEEDS])
                    it = 0
                    pop_for_ind = PopulationPrograms([deepcopy(ind_program) for i in 1:NSEEDS])
                    local ind_fitness = Vector{Float64}(undef, NSEEDS)
                    while true
                        n_playing = sum(still_playing)
                        actions_for_players_playing = Vector{Int}(undef, n_playing)
                        state_view = view(state, still_playing)
                        nstates = size(state_view, 1)
                        @assert nstates == n_playing == length(indices_jl) == length(players_py) "N states : $nstates vs N playing : $n_playing Indices : $(length(players_py))"
                        partitions = resolve_partitions_for_the_round(n_playing, nt)
                        step_pop_st!(
                            view(pop_for_ind.population_programs, indices_jl),
                            actions_for_players_playing,
                            model_arch, meta_library, generation, it, partitions,
                            view(mts, indices_jl), state_view, TAPES)

                        # increment +1 frames
                        frames_per_seed[indices_jl] .+= 1

                        PongCompetition.pycall_lock() do
                            state, rew, term, truncated, info = PongCompetition.step(ENVS_IND, actions_for_players_playing,
                                indices_jl)
                        end

                        # Reward
                        @assert length(rew) == length(indices_jl) == n_playing
                        for (i, ind_index) in enumerate(indices_jl)
                            r = rew[i]
                            r = isnan(r) ? 100 : r
                            if it == 0
                                ind_fitness[ind_index] = r
                            else
                                ind_fitness[ind_index] += r
                            end
                        end

                        # determine who is still playing
                        still_playing = .~(term .|| truncated)
                        if sum(still_playing) == 0
                            @info "Breaking cause all in time limit or game over"
                            break
                        end

                        players_py = players_py[still_playing]
                        indices_jl = indices_jl[still_playing]

                        # save actions 
                        for (i, idx) in enumerate(indices_jl)
                            push!(ACTIONS[idx], actions_for_players_playing[i])
                        end
                        it += 1
                    end

                    entropies_seeds = Float64[]
                    for tape in TAPES # loop over diff seeds
                        entropies_in_tape = Float64[]
                        m = tapes_to_matrix(identity.(tape))
                        for col_idx in 1:size(m, 2) # loop over ops
                            try
                                h = StatsBase.fit(StatsBase.Histogram, m[:, col_idx], nbins=50)
                                push!(entropies_in_tape, StatsBase.entropy(h.weights / length(m[:, col_idx])))
                            catch
                                push!(entropies_in_tape, 0.0)
                            end
                        end
                        try
                            replace!(entropies_in_tape, NaN => 0.0)
                            entropy = mean(entropies_in_tape)
                            entropy = isnan(entropy) ? 0.0 : entropy
                            push!(entropies_seeds, entropy)

                        catch
                            push!(entropies_seeds, 0.0)
                        end
                    end
                    @show ind_fitness

                    # Fitness per seed
                    F = Float64[]
                    for s in 1:NSEEDS
                        f = ind_fitness[s]
                        e = try
                            entropies_seeds[s]
                        catch
                            0.0
                        end
                        e = clamp(e, 0.0, 0.99) # A point is always better than e
                        push!(F, f * -1.0 + e * -1.0)
                    end
                    @show entropies_seeds
                    @show frames_per_seed

                    gc_t_inner = @elapsed GC.gc(true)
                    @show gc_t_inner
                    GC.gc(true)
                    # TODO changed here from mean(F) to maximum(F)
                    pop_fits[ith_program] = [maximum(F)]
                    PongCompetition.pycall_lock() do
                        PongCompetition.gc_()
                        PythonCall.GC.gc()
                    end
                    GC.gc()
                end
                # gc_t = @elapsed GC.gc(true)
                # @show gc_t
                # PongCompetition.gc_()
            end
            push!(tasks, t)
        end
        fetch.(tasks)


        PongCompetition.pycall_lock() do
            for pool in ENVS
                for env in pool
                    env.close()
                end
            end
        end

        PongCompetition.pycall_lock() do
            GC.gc(true)
            PongCompetition.gc_()
            PythonCall.GC.gc()
            GC.gc(true)
        end
        return new(pop_fits)
    end
end

struct EnvpoolAtariNSGA2Endpoint <: UTCGP.BatchEndpoint
    fitness_results::Vector{Vector{Float64}}
    function EnvpoolAtariNSGA2Endpoint(
        pop::UTCGP.PopulationPrograms,
        model_arch::modelArchitecture,
        meta_library::MetaLibrary,
        generation::Int
    )
        global NP, SEEDS, GAMENAME
        n, nt = length(pop), nthreads()
        seeds_for_envs = collect(1:SEEDS)
        local ENVS
        PongCompetition.pycall_lock() do
            pyexec("import numpy as np; np.random.seed(0)", PongCompetition)
            ENVS = pool_of_pools(GAMENAME, n, seeds_for_envs, 3000) # N pools with seeds
            PongCompetition.reset(ENVS, seeds_for_envs)
        end

        UTCGP.reset_programs!.(pop)
        pop_fits = Vector{Vector{Float64}}(undef, n)
        tasks = []
        pop_partitions = resolve_partitions_for_the_round(n, nt)
        @show collect(pop_partitions)
        for (ith_partition, pop_partition) in enumerate(pop_partitions)
            t = @spawn begin
                tid = threadid()
                for (ith_program) in pop_partition
                    ind_program = pop[ith_program]
                    ENVS_IND = ENVS[ith_program] # is a pool with seeds
                    println("Running ind n° $ith_program")
                    local state, rew, term, truncated, info
                    local still_playing
                    PongCompetition.pycall_lock() do
                        PongCompetition.reset(ENVS_IND, seeds_for_envs)
                        state, rew, term, truncated, info = PongCompetition.step(ENVS_IND, zeros(Int, SEEDS))
                    end
                    still_playing = .~(term .|| truncated)
                    indices_jl = collect(1:SEEDS)
                    players_py = collect(0:SEEDS-1)
                    ACTIONS = [[] for i in 1:SEEDS]
                    TAPES = [[] for i in 1:SEEDS]

                    mts = identity.([MersenneTwister(i) for i in 1:SEEDS])
                    it = 0
                    pop_for_ind = PopulationPrograms([deepcopy(ind_program) for i in 1:SEEDS])
                    local ind_fitness = Vector{Float64}(undef, SEEDS)
                    local frames_played = zeros(Float64, SEEDS)
                    while true
                        n_playing = sum(still_playing)
                        actions_for_players_playing = Vector{Int}(undef, n_playing)
                        state_view = view(state, still_playing)
                        nstates = size(state_view, 1)
                        @assert nstates == n_playing == length(indices_jl) == length(players_py) "N states : $nstates vs N playing : $n_playing Indices : $(length(players_py))"
                        partitions = resolve_partitions_for_the_round(n_playing, nt)
                        step_pop_st!(
                            view(pop_for_ind.population_programs, indices_jl),
                            actions_for_players_playing,
                            model_arch, meta_library, generation, it, partitions,
                            view(mts, indices_jl), state_view, TAPES)

                        PongCompetition.pycall_lock() do
                            state, rew, term, truncated, info = PongCompetition.step(ENVS_IND, actions_for_players_playing,
                                indices_jl)
                        end

                        # Reward
                        @assert length(rew) == length(indices_jl) == n_playing
                        for (i, ind_index) in enumerate(indices_jl)
                            r = rew[i]
                            r = isnan(r) ? 100 : r
                            if it == 0
                                ind_fitness[ind_index] = r
                            else
                                ind_fitness[ind_index] += r
                            end

                            # Add one frame for seed still playing
                            frames_played[ind_index] += 1
                        end

                        # determine who is still playing
                        still_playing = .~(term .|| truncated)
                        if sum(still_playing) == 0
                            @info "Breaking cause all in time limit or game over"
                            break
                        end

                        players_py = players_py[still_playing]
                        indices_jl = indices_jl[still_playing]

                        # save actions 
                        for (i, idx) in enumerate(indices_jl)
                            push!(ACTIONS[idx], actions_for_players_playing[i])
                        end
                        it += 1
                    end

                    # local entropy
                    # entropies = Float64[]
                    # try
                    #     for tape in TAPES # loop over diff seeds
                    #         m = tapes_to_matrix(identity.(tape))
                    #         for col_idx in 1:size(m, 2) # loop over ops
                    #             try
                    #                 h = StatsBase.fit(StatsBase.Histogram, m[:, col_idx], nbins=50)
                    #                 push!(entropies, StatsBase.entropy(h.weights / length(m[:, col_idx])))
                    #             catch
                    #                 push!(entropies, 0.0)
                    #             end
                    #         end
                    #     end
                    #     replace!(entropies, NaN => 0.0)
                    #     entropy = mean(entropies)
                    #     entropy = isnan(entropy) ? 0.0 : entropy
                    # catch
                    #     entropy = 0.0
                    # end
                    # @show ind_fitness

                    # Fitness per seed
                    F = Float64[mean(ind_fitness)*-1.0, mean(frames_played)*-1.0]
                    # for s in 1:SEEDS
                    #     f = ind_fitness[s]
                    #     # e = try
                    #     #     entropies[s]
                    #     # catch
                    #     #     0.0
                    #     # end
                    #     # e = clamp(e, 0.0, 0.99) # A point is always better than e
                    #     frames_score = mean(frames_played)
                    #     push!(F, f * -1.0 + frames_score * -1.0)
                    # end
                    # @show entropies
                    @show frames_played
                    pop_fits[ith_program] = F
                end
            end
            push!(tasks, t)
        end
        fetch.(tasks)
        GC.gc(true)
        PongCompetition.pycall_lock() do
            for pool in ENVS
                for env in pool
                    env.close()
                end
            end
        end
        GC.gc(true)
        ENVS = nothing
        GC.gc(true)
        PongCompetition.gc_()
        PythonCall.GC.gc()
        GC.gc(true)
        @show pop_fits
        return new(pop_fits)
    end
end

function UTCGP.get_endpoint_results(e::EnvpoolAtariNSGA2Endpoint)
    return e.fitness_results
end

##########################################
# FIT NSGA II 
##########################################
"""
Initializes n inds and returns them.
"""
function random_pop(n)
    pop = UTGenome[]
    for i in 1:n
        shared_inputs, ut_genome = make_evolvable_utgenome(
            model_arch, ml, node_config
        )
        initialize_genome!(ut_genome)
        correct_all_nodes!(ut_genome, model_arch, ml, shared_inputs)
        UTCGP.fix_all_output_nodes!(ut_genome)
        push!(pop, ut_genome)
    end
    pop
end

"""
Copies the main parent and mutates it
"""
function non_random_pop(n::Int, parent::UTGenome)
    @assert n > 0
    pop = UTGenome[]
    for i in 1:n
        new_ind = deepcopy(parent)
        new_material_mutation!(
            new_ind,
            run_conf,
            model_arch,
            ml,
            shared_inputs,)
        push!(pop, new_ind)
    end
    pop
end

function _nsga2_init_params(run_config::RunConfNSGA2)
    early_stop = false
    best_programs = nothing
    pareto_front_idx = nothing
    # population = UTCGP.Population([deepcopy(genome) for i = 1:run_config.pop_size]) # initial pop 
    ranks = [1 for i = 1:run_config.pop_size]
    distances = [0.0 for i = 1:run_config.pop_size]
    return early_stop, best_programs, pareto_front_idx, ranks, distances
end

function fit_nsga2_mt(
    population::Vector{UTGenome},
    shared_inputs::SharedInput,
    # genome::UTGenome,
    model_architecture::modelArchitecture,
    node_config::nodeConfig,
    run_config::RunConfNSGA2,
    meta_library::MetaLibrary,
    # Callbacks before training
    pre_callbacks::Optional_FN,
    # Callbacks before step (before looping through data)
    population_callbacks::Mandatory_FN,
    mutation_callbacks::Mandatory_FN,
    output_mutation_callbacks::Mandatory_FN,
    decoding_callbacks::Mandatory_FN,
    # Callbacks per step (while looping through data)
    endpoint_callback::Type{<:BatchEndpoint},
    final_step_callbacks::Optional_FN,
    # Callbacks after step ::
    survival_selection_callbacks::Optional_FN,
    epoch_callbacks::Optional_FN,
    early_stop_callbacks::Optional_FN,
    last_callback::Optional_FN,
)
    println("FIT NSGAII")
    local early_stop, best_programs, pareto_front_idx, ranks, distances = _nsga2_init_params(run_config)
    ind_performances = Vector{Vector{Float64}}()
    population = UTCGP.Population(population)
    local pareto_front_individuals
    # PRE CALLBACKS
    # _make_pre_callbacks_calls(pre_callbacks)
    M_gen_loss_tracker = UTCGP.GenerationMultiObjectiveLossTracker()
    for iteration = 1:run_config.generations
        early_stop ? break : nothing
        @warn "Iteration : $iteration"

        # Population
        nsga2_pop_args = UTCGP.NSGA2_POP_ARGS(
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            ranks,
            distances,
        )
        if iteration > 1
            # Compute offspring 
            offspring, time_pop =
                @unwrap_or UTCGP._make_nsga2_population(nsga2_pop_args, population_callbacks) throw(
                    "Could not unwrap make_population",
                )
            # Offspring mutation
            nsga2_mutation_args = UTCGP.NSGA2_MUTATION_ARGS(
                offspring,
                iteration,
                run_config,
                model_architecture,
                node_config,
                meta_library,
                shared_inputs,
            )
            offspring, time_mut =
                @unwrap_or UTCGP._make_nsga2_mutations!(nsga2_mutation_args, mutation_callbacks) throw(
                    "Could not unwrap make_me_mutations",
                )

            # Output mutations ---
            # offspring, time_out_mut = @unwrap_or _make_ga_output_mutations!(
            #     ga_mutation_args,
            #     output_mutation_callbacks,
            # ) throw("Could not unwrap make_ga_output_mutations")
        else
            offspring = population
        end

        # Genotype to Phenotype mapping --- 
        offspring_programs, time_pop_prog = UTCGP._make_decoding(
            offspring,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            shared_inputs,
            decoding_callbacks,
        )

        UTCGP.reset_programs!.(offspring_programs)

        # TODO implement tracker for vector fitnesses
        # M_individual_loss_tracker = IndividualLossTracker() # size of []

        @warn "MT Graphs evals"
        endpoint_holder = PongCompetition.PythonCall.GIL.unlock() do
            endpoint_callback(
                offspring_programs,
                model_architecture,
                meta_library,
                iteration,
            )
        end
        offspring_fitness_values = UTCGP.get_endpoint_results(endpoint_holder) # Vector{Vector{Float64}}

        # TODO implement tracker for vector fitnesses
        # UTCGP.add_pop_loss_to_ind_tracker!(M_individual_loss_tracker, fitness_values)  # appends the loss for the ith x sample to the

        # Resetting the population (removes node values)
        [UTCGP.reset_genome!(g) for g in offspring]

        # final step call...
        if !isnothing(final_step_callbacks)
            for final_step_callback in final_step_callbacks
                UTCGP.get_fn_from_symbol(final_step_callback)()
            end
        end

        # Merge parents and offspring
        if iteration > 1
            fitness_values = vcat(ind_performances, offspring_fitness_values)
            full_population = UTCGP.Population(vcat(population.pop, offspring.pop))
        else
            fitness_values = offspring_fitness_values
            full_population = offspring
        end

        ranks, distances = UTCGP._ranks_and_crowding_distances(fitness_values)

        # Survival selection
        nsga2_selection_args =
            UTCGP.NSGA2_SELECTION_ARGS(ranks, distances, full_population, run_config)
        survival_idx, time_survival = @unwrap_or UTCGP._make_nsga2_survival_selection(
            nsga2_selection_args,
            survival_selection_callbacks,
        ) throw("Could not unwrap make_nsga2_selection")

        ind_performances = fitness_values[survival_idx]
        elite_ranks = ranks[survival_idx]
        population = Population(full_population.pop[survival_idx])

        # @show ind_performances

        # TODO loss trackers
        # ind_performances = UTCGP.resolve_ind_loss_tracker(M_individual_loss_tracker)
        # try
        #     # histogram(ind_performances) |> println
        #     # histogram([d[1] for d in descriptor_values]) |> println
        #     # @show ARCHIVE.descriptors
        #     histogram(collect(skipmissing(ARCHIVE.fitness_values)))
        # catch e
        #     @error "Could not drawn histogram"
        # end

        pareto_front_individuals =
            population[findall(==(minimum(elite_ranks)), elite_ranks)]
        pareto_front_fitnesses =
            ind_performances[findall(==(minimum(elite_ranks)), elite_ranks)]
        best_programs = [
            UTCGP.decode_with_output_nodes(pfi, meta_library, model_architecture, shared_inputs) for pfi in pareto_front_individuals
        ]

        try
            scatterplot(
                [i[1] for i in pareto_front_fitnesses],
                [i[2] for i in pareto_front_fitnesses],
                title="Pareto Front"
            ) |> println
        catch e
            @show e
        end

        # EPOCH CALLBACK
        if !isnothing(epoch_callbacks)
            UTCGP._make_epoch_callbacks_calls(
                pareto_front_fitnesses,
                ranks,
                Population(pareto_front_individuals),
                iteration,
                run_config,
                model_architecture,
                node_config,
                meta_library,
                shared_inputs,
                PopulationPrograms(best_programs),
                nothing, # []
                nothing, # best_programs,
                nothing,# elite_idx,
                nothing,
                epoch_callbacks,
            )
        end

        for bp in best_programs
            @show bp
        end

        # store iteration loss/fitness
        UTCGP.affect_fitness_to_loss_tracker!(
            M_gen_loss_tracker,
            iteration,
            pareto_front_fitnesses,
        )
        println("Iteration $iteration. 
                 Pareto front fitness values : $pareto_front_fitnesses")
        # @warn pareto_front_fitnesses
        flush(stdout)
        flush(stderr)

        # EARLY STOP CALLBACK 
        if !isnothing(early_stop_callbacks) && length(early_stop_callbacks) != 0
            # early_stop_args = GA_EARLYSTOP_ARGS(
            #     M_gen_loss_tracker,
            #     M_individual_loss_tracker,
            #     ind_performances,
            #     population,
            #     iteration,
            #     run_config,
            #     model_architecture,
            #     node_config,
            #     meta_library,
            #     shared_inputs,
            #     population_programs,
            #     elite_fitnesses,
            #     best_programs,
            #     elite_idx,
            # )
            # early_stop =
            #     _make_ga_early_stop_callbacks_calls(early_stop_args, early_stop_callbacks) # true if any
        end

        if early_stop
            g = run_config.generations
            @warn "Early returning at iteration : $iteration from $g total iterations"
            if !isnothing(last_callback)
                # last_callback(
                #     ind_performances,
                #     population,
                #     iteration,
                #     run_config,
                #     model_architecture,
                #     node_config,
                #     meta_library,
                #     population_programs,
                #     elite_fitnesses,
                #     best_programs,
                #     elite_idx,
                # )
            end
            # UTCGP.show_program(program)
            return tuple(pareto_front_individuals, best_programs, M_gen_loss_tracker)
        end
        gct = @elapsed GC.gc(true)
        GC.gc(false)
        @warn "Running GC at the end of iteration. GC time : $gct"
    end
    return (pareto_front_individuals, best_programs, M_gen_loss_tracker)
end

function fixed_game_without_unlock(program, seed, model_arch, meta_library, generation, frames::Int=30_000)
    global NP, GAMENAME
    NSEEDS = 1
    pop = UTCGP.PopulationPrograms([program])
    n, nt = length(pop), 1 #nthreads()
    seeds_for_envs = [seed]
    local ENVS
    pyexec("import numpy as np; np.random.seed(0)", PongCompetition)
    ENVS = pool_of_pools(GAMENAME, n, [seed], frames) # N pools with seeds
    PongCompetition.reset(ENVS, [seed])

    UTCGP.reset_programs!.(pop)
    pop_fits = Vector{Vector{Float64}}(undef, n)
    pop_own_points = Vector{Vector{Float64}}(undef, n)
    pop_opponent_points = Vector{Vector{Float64}}(undef, n)

    tasks = []
    pop_partitions = resolve_partitions_for_the_round(n, nt)
    STATES = []
    INDS = []
    ACTIONS_ = []
    for (ith_partition, pop_partition) in enumerate(pop_partitions)
        t = @spawn begin
            tid = threadid()
            println("THREAD ID IS $tid")
            for (ith_program) in pop_partition
                ind_program = pop[ith_program]
                ENVS_IND = ENVS[ith_program] # is a pool with seeds
                println("Running ind n° $ith_program")
                local state, rew, term, truncated, info
                local still_playing
                PongCompetition.reset(ENVS_IND, seeds_for_envs)
                state, rew, term, truncated, info = PongCompetition.step(ENVS_IND, zeros(Int, NSEEDS))
                still_playing = .~(term .|| truncated)
                indices_jl = collect(1:NSEEDS)
                players_py = collect(0:NSEEDS-1)
                ACTIONS = [[] for i in 1:NSEEDS]
                TAPES = [[] for i in 1:NSEEDS]

                mts = identity.([MersenneTwister(i) for i in 1:NSEEDS])
                it = 0
                pop_for_ind = PopulationPrograms([deepcopy(ind_program) for i in 1:NSEEDS])
                local ind_fitness = Vector{Float64}(undef, NSEEDS)
                local ind_own_points = Vector{Float64}(undef, NSEEDS)
                local ind_opponent_points = Vector{Float64}(undef, NSEEDS)
                while true
                    n_playing = sum(still_playing)
                    actions_for_players_playing = Vector{Int}(undef, n_playing)
                    state_view = view(state, still_playing)
                    nstates = size(state_view, 1)
                    @assert nstates == n_playing == length(indices_jl) == length(players_py) "N states : $nstates vs N playing : $n_playing Indices : $(length(players_py))"
                    partitions = resolve_partitions_for_the_round(n_playing, nt)
                    step_pop_st_and_log!(
                        view(pop_for_ind.population_programs, indices_jl),
                        actions_for_players_playing,
                        model_arch, meta_library, generation, it, partitions,
                        view(mts, indices_jl), state_view, TAPES, INDS)
                    payload = Dict()
                    payload["action"] = deepcopy(actions_for_players_playing)
                    payload["seen"] = deepcopy(state_view)
                    payload["seen_state"] = deepcopy(state)
                    # if it == 50
                    #     actions_for_players_playing = Int[0]
                    #     payload["action"] = deepcopy(actions_for_players_playing)
                    # end
                    state, rew, term, truncated, info = PongCompetition.step(ENVS_IND, actions_for_players_playing,
                        indices_jl)

                    payload["result_state"] = deepcopy(state)
                    payload["it"] = it


                    # Reward
                    @assert length(rew) == length(indices_jl) == n_playing
                    for (i, ind_index) in enumerate(indices_jl)
                        r = rew[i]
                        r = isnan(r) ? 100 : r
                        if it == 0
                            ind_fitness[ind_index] = r

                            if r == 1
                                ind_own_points[ind_index] = 1
                            elseif r == -1
                                ind_opponent_points[ind_index] = 1
                            end  
                        else
                            ind_fitness[ind_index] += r

                            if r == 1
                                ind_own_points[ind_index] += 1
                            elseif r == -1
                                ind_opponent_points[ind_index] += 1
                            end  
                        end
                    end

                    # determine who is still playing
                    still_playing = .~(term .|| truncated)
                    if sum(still_playing) == 0
                        @info "Breaking cause all in time limit or game over"
                        break
                    end

                    players_py = players_py[still_playing]
                    indices_jl = indices_jl[still_playing]

                    # save actions 
                    for (i, idx) in enumerate(indices_jl)
                        push!(ACTIONS[idx], actions_for_players_playing[i])
                    end
                    it += 1
                    push!(STATES, payload)
                end

                # local entropy
                # entropies = Float64[]
                # try
                #     for tape in TAPES # loop over diff seeds
                #         m = tapes_to_matrix(identity.(tape))
                #         for col_idx in 1:size(m, 2) # loop over ops
                #             try
                #                 h = StatsBase.fit(StatsBase.Histogram, m[:, col_idx], nbins=50)
                #                 push!(entropies, StatsBase.entropy(h.weights / length(m[:, col_idx])))
                #             catch
                #                 push!(entropies, 0.0)
                #             end
                #         end
                #     end
                #     replace!(entropies, NaN => 0.0)
                #     entropy = mean(entropies)
                #     entropy = isnan(entropy) ? 0.0 : entropy
                # catch
                #     entropy = 0.0
                # end
                # @show ind_fitness

                # Fitness per seed
                F = Float64[]
                OW_P = Float64[]
                OP_P = Float64[]
                for s in 1:NSEEDS
                    f = ind_fitness[s]
                    owp = ind_own_points[s]
                    opp = ind_opponent_points[s]
                    e = try
                        entropies[s]
                    catch
                        0.0
                    end
                    # e = clamp(e, 0.0, 0.99) # A point is always better than e
                    push!(F, f * -1.0 + e * -1.0)
                    push!(OW_P, owp)
                    push!(OP_P, opp)
                end
                push!(ACTIONS_, ACTIONS...)
                pop_fits[ith_program] = [mean(F)]
                pop_own_points[ith_program] = [mean(OW_P)]
                pop_opponent_points[ith_program] = [mean(OP_P)]

            end
        end
        push!(tasks, t)
    end
    fetch.(tasks)

    GC.gc(true)
    for pool in ENVS
        for env in pool
            env.close()
        end
    end
    GC.gc(true)
    ENVS = nothing
    GC.gc(true)
    PongCompetition.gc_()
    PythonCall.GC.gc()
    GC.gc(true)
    return pop_fits, STATES, INDS, ACTIONS_, pop_own_points, pop_opponent_points
end

# GAME PLAYED BY FIXED PROGRAM 
function run_program_with_inputs!(
    pop_for_ind,
    state_view,
    actions_for_players_playing
)
    inputs = state_view[1]
    ins = Any[SImageND(inputs[t, CROP[1], CROP[2]]) for t in axes(inputs, 1)]
    push!(ins, -1.0, 0.5, 2.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    out = pop_for_ind[1](ins)
    actions_for_players_playing[1] = out
end

function fixed_game_fixed_program(program, seed, generation, frames::Int=10_000)
    global NP, GAMENAME
    NSEEDS = 1
    pop = [program]
    n, nt = length(pop), 1 #nthreads()
    seeds_for_envs = [seed]
    local ENVS
    PongCompetition.pycall_lock() do
        pyexec("import numpy as np; np.random.seed(0)", PongCompetition)
        ENVS = pool_of_pools(GAMENAME, n, [seed], frames) # N pools with seeds
        PongCompetition.reset(ENVS, [seed])
    end
    pop_fits = Vector{Vector{Float64}}(undef, n)
    tasks = []
    pop_partitions = resolve_partitions_for_the_round(n, nt)
    STATES = []
    INDS = []
    ACTIONS_ = []
    for (ith_partition, pop_partition) in enumerate(pop_partitions)
        t = @spawn begin
            tid = threadid()
            for (ith_program) in pop_partition
                ind_program = pop[ith_program]
                ENVS_IND = ENVS[ith_program] # is a pool with seeds
                println("Running ind n° $ith_program")
                local state, rew, term, truncated, info
                local still_playing
                PongCompetition.pycall_lock() do
                    PongCompetition.reset(ENVS_IND, seeds_for_envs)
                    state, rew, term, truncated, info = PongCompetition.step(ENVS_IND, zeros(Int, NSEEDS))
                end
                still_playing = .~(term .|| truncated)
                indices_jl = collect(1:NSEEDS)
                players_py = collect(0:NSEEDS-1)
                ACTIONS = [[] for i in 1:NSEEDS]
                TAPES = [[] for i in 1:NSEEDS]

                mts = identity.([MersenneTwister(i) for i in 1:NSEEDS])
                it = 0
                pop_for_ind = [deepcopy(ind_program) for i in 1:NSEEDS]
                local ind_fitness = Vector{Float64}(undef, NSEEDS)
                while true
                    n_playing = sum(still_playing)
                    actions_for_players_playing = Vector{Int}(undef, n_playing)
                    state_view = view(state, still_playing)
                    nstates = size(state_view, 1)
                    @assert nstates == n_playing == length(indices_jl) == length(players_py) "N states : $nstates vs N playing : $n_playing Indices : $(length(players_py))"
                    partitions = resolve_partitions_for_the_round(n_playing, nt)
                    run_program_with_inputs!(
                        pop_for_ind,
                        state_view,
                        actions_for_players_playing
                    )
                    push!(STATES, state_view)
                    PongCompetition.pycall_lock() do
                        state, rew, term, truncated, info = PongCompetition.step(ENVS_IND, actions_for_players_playing,
                            indices_jl)
                    end

                    # Reward
                    @assert length(rew) == length(indices_jl) == n_playing
                    for (i, ind_index) in enumerate(indices_jl)
                        r = rew[i]
                        r = isnan(r) ? 100 : r
                        if it == 0
                            ind_fitness[ind_index] = r
                        else
                            ind_fitness[ind_index] += r
                        end
                    end

                    # determine who is still playing
                    still_playing = .~(term .|| truncated)
                    if sum(still_playing) == 0
                        @info "Breaking cause all in time limit or game over"
                        break
                    end

                    players_py = players_py[still_playing]
                    indices_jl = indices_jl[still_playing]

                    # save actions 
                    for (i, idx) in enumerate(indices_jl)
                        push!(ACTIONS[idx], actions_for_players_playing[i])
                    end
                    it += 1
                end

                # Fitness per seed
                F = Float64[]
                for s in 1:NSEEDS
                    f = ind_fitness[s]
                    push!(F, f * -1.0)
                end
                push!(ACTIONS_, ACTIONS...)
                pop_fits[ith_program] = [mean(F)]
            end
        end
        push!(tasks, t)
    end
    fetch.(tasks)

    GC.gc(true)
    PongCompetition.pycall_lock() do
        for pool in ENVS
            for env in pool
                env.close()
            end
        end
    end
    GC.gc(true)
    ENVS = nothing
    GC.gc(true)
    PongCompetition.gc_()
    PythonCall.GC.gc()
    GC.gc(true)
    return pop_fits, STATES, INDS, ACTIONS_
end
