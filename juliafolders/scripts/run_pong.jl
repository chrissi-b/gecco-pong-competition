JULIA_CONDAPKG_OFFLINE = true

include("src/PongCompetition.jl")
using .PongCompetition

using PythonCall
NP = pyimport("numpy")

println("This is the run_pong.jl script")

greet()

using Logging
using Base.Threads
import JSON
using UnicodePlots
using ErrorTypes
using Revise
using Dates
using UTCGP
using UTCGP: jsonTracker, save_json_tracker, repeatJsonTracker, jsonTestTracker
using UTCGP: get_image2D_factory_bundles, SImageND, DataFrames
using UTCGP: AbstractCallable, Population, RunConfNSGA2, PopulationPrograms, IndividualPrograms, get_float_bundles, replace_shared_inputs!, evaluate_individual_programs, reset_programs!
using DataStructures
using UUIDs
using ImageCore
using Statistics
using StatsBase
using UTCGP: SImage2D, BatchEndpoint
using Random
using ImageIO
using ThreadPinning
using ThreadPools

# pinthreads(:cores)
println(ThreadPinning.threadinfo())

const NIMAGES::Ref{Int} = Ref{Int}(12)
disable_logging(Logging.Debug - 2000)

dir = @__DIR__
pwd_dir = pwd()
println("pwd_dir $pwd_dir")

file = @__FILE__
home = dirname(dirname(file))
println("home $home")


if occursin("REPL", file)
    home = "./"
end

# EXAMPLE OF GAME #
CROPS = Dict(
    "pong" => (15:77, 1:84)
)
GAMENAME = "PongNoFrameskip-v4"
GAME = "pong"
const CROP = CROPS[GAME]
env_ex = pool(GAMENAME, [1])
obs_, rew_, term_, trunc_, info_ = PongCompetition.step(env_ex, [0]);
x_example = obs_[1]
example = x_example[1, :, :]
example = example[CROP[1], CROP[2]]
IMAGE_TYPE = typeof(SImageND(example))
IMG_SIZE = example |> size
PROB_ACTION = false

# DON'T USE
CUSTOM_SEEDS = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 16, 19, 21, 22, 23, 25, 27, 28, 29, 30, 32, 38, 57, 65, 67, 73, 106, 130]

SEED_CHANGE_THRESHOLD = parse(Int, ARGS[1])

SEEDS_UPPERLEFT = [10, 130, 2, 12, 11, 13, 38] 
SEEDS_LOWERLEFT = [5, 3, 6, 16, 106, 22, 1, 4] 
SEEDS_LEFT = [SEEDS_UPPERLEFT; SEEDS_LOWERLEFT]

SEEDS_UPPERRIGHT = [14, 21, 67, 23, 25, 27, 28] 
SEEDS_LOWERRIGHT = [19, 73, 65, 57, 7, 32, 30, 29]
SEEDS_RIGHT = [SEEDS_UPPERRIGHT; SEEDS_LOWERRIGHT]

CURRENT_SEEDS = [1, 1, 1, 1]

#NSEEDS = length(CUSTOM_SEEDS)
NSEEDS = length(CURRENT_SEEDS)
folder = "ga_metrics/$GAME"

include(joinpath(home, "scripts", "utils.jl"))
include(joinpath(home, "scripts", "extra_fns.jl"))

action_mapping_dict = Dict(
    "pong" => [3, pong_action_mapping],
)

NACTIONS = 3
ACTIONS = [0, 4, 5]
ACTION_MAPPER = action_mapping_dict[GAME][2]
@show ACTIONS

# SEED 
# seed_ = trunc(Int, rand() * 1000)
# seed_ = parse(Int, "1")
seed_ = parse(Int, ARGS[2])
@warn "Seed : $seed_"
Random.seed!(seed_)

# HASH 
hash = UUIDs.uuid4() |> string
folder = joinpath(folder, hash)
mkdir(folder)

### RUN CONF ###
pop_size = nthreads()
gens = 10
mut_rate = 1.1
output_mut_rate = 0.1

# Bundles Integer
fallback() = SImageND(ones(N0f8, (IMG_SIZE[1], IMG_SIZE[2])))
image2D = UTCGP.get_image2D_factory_bundles_atari()
for factory_bundle in image2D
    for (i, wrapper) in enumerate(factory_bundle)
        try
            fn = wrapper.fn(IMAGE_TYPE) # specialize
            wrapper.fallback = fallback
            # create a new wrapper in order to change the type
            factory_bundle.functions[i] =
                UTCGP.FunctionWrapper(fn, wrapper.name, wrapper.caster, wrapper.fallback)

        catch
        end
    end
end

float_bundles = UTCGP.get_float_bundles_atari()
push!(float_bundles, float_comparetuples.bundle_float_comparetuples)

# vector_float_bundles = UTCGP.get_listfloat_bundles()
# push!(vector_float_bundles, deepcopy(listnumber_fromimg.bundle_listnumber_from_img))
# POSITIONS 
positions_bundles = [deepcopy(tupleintint_2Dposition.bundle_tupleintint_2Dposition)]

# Libraries
lib_image2D = Library(image2D)
lib_float = Library(float_bundles)
# lib_vecfloat = Library(vector_float_bundles)
lib_positions = Library(positions_bundles)

lib_number_to_img = Library(deepcopy([float_bundles[7], float_bundles[8], float_bundles[9]]))
UTCGP.unpack_bundles_in_library!(lib_number_to_img)
# MetaLibrarylibfloat# ml = MetaLibrary([lib_image2D, lib_float, lib_vecfloat, lib_int])
ml = MetaLibrary([lib_image2D, lib_float, lib_positions])

offset_by = 14 # 4 inputs and 10 constants - 0.1 0.2 1. -1. 10 20 30 40 50 60 

### Model Architecture ###
model_arch = modelArchitecture(
    [IMAGE_TYPE, IMAGE_TYPE, IMAGE_TYPE, IMAGE_TYPE, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,],
    [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    # [IMAGE_TYPE, Float64, Vector{Float64}, Int], # genome
    [IMAGE_TYPE, Float64, Tuple{Int,Int}], # genome
    [Float64 for i in 1:NACTIONS], # outputs
    [2 for i in 1:NACTIONS]
)

### Node Config ###
N_nodes = 100
node_config = nodeConfig(N_nodes, 1, 3, offset_by)

### Make UT GENOME ###
shared_inputs, ut_genome = make_evolvable_utgenome(
    model_arch, ml, node_config
)
initialize_genome!(ut_genome)
correct_all_nodes!(ut_genome, model_arch, ml, shared_inputs)
UTCGP.fix_all_output_nodes!(ut_genome)

n_elite = Base.ceil(Int, pop_size * 0.2)
n_new = pop_size - n_elite
tour_size = n_elite <= 3 ? max(1, n_elite - 1) : 3
@show tour_size n_elite n_new
run_conf = RunConfGA(
    n_elite, n_new, tour_size, mut_rate, 0.1, gens
)

pop = random_pop(n_elite);

######################
# METRICS ############
######################

metrics_path = joinpath(folder, "metrics$seed_.json")
struct jsonTrackerME <: UTCGP.AbstractCallable
    tracker::UTCGP.jsonTracker
    label::String
    test_losses::Vector
end
h_params = Dict(
    "generations" => run_conf.generations,
    "mutation_rate" => run_conf.mutation_rate,
    "seed" => seed_,
    "mutation" => "ga_numbered_new_material_mutation_callback",
    "n_elite" => n_elite,
    "n_new" => n_elite,
    "n_nodes" => N_nodes,
    "tour_size" => tour_size,
)
f = open(metrics_path, "a", lock=true)
metric_tracker = UTCGP.jsonTracker(h_params, f)
atari_tracker = jsonTrackerME(metric_tracker, "Test", [])

function (jtga::jsonTrackerME)(ind_performances,
    population,
    iteration,
    run_config,
    model_architecture,
    node_config,
    meta_library,
    shared_inputs,
    population_programs,
    elite_fitnesses,
    best_programs,
    elite_idx,
    batch
)
    best_f = minimum(ind_performances)
    avg_f = mean(ind_performances)
    std_f = std(ind_performances)
    @warn "JTT $(jtga.label) Fitness : $best_f"
    s = Dict("data" => jtga.label,
        "iteration" => iteration,
        "best_fitness" => best_f,
        "avg_fitness" => avg_f,
        "std_fitness" => std_f)
    push!(jtga.test_losses, best_f)
    write(jtga.tracker.file, JSON.json(s), "\n")
    flush(jtga.tracker.file)
end

# CHECKPOINT
function save_payload(best_genome,
    best_programs, gen_tracker,
    shared_inputs, ml, run_conf, node_config,
    name::String="best_genome.pickle")
    payload = Dict()
    payload["best_genome"] = deepcopy(best_genome)
    payload["best_program"] = deepcopy(best_programs)
    payload["gen_tracker"] = deepcopy(gen_tracker)
    payload["shared_inputs"] = deepcopy(shared_inputs)
    payload["ml"] = deepcopy(ml)
    payload["run_conf"] = deepcopy(run_conf)
    payload["node_config"] = deepcopy(node_config)

    genome_path = joinpath(folder, name)
    open(genome_path, "w") do io
        @info "Writing payload to $genome_path"
        write(io, UTCGP.general_serializer(deepcopy(payload)))
    end
end
struct checkpoint <: UTCGP.AbstractCallable
    every::Int
end

function (c::checkpoint)(
    ind_performances,
    population,
    iteration,
    run_config,
    model_architecture,
    node_config,
    meta_library,
    shared_inputs,
    population_programs,
    elite_fitnesses,
    best_programs,
    elite_idx,
    batch)
    if iteration % c.every == 0
        best_individual = population[1]
        best_program = UTCGP.decode_with_output_nodes(
            best_individual,
            meta_library,
            model_architecture,
            shared_inputs,
        )
        save_payload(best_individual, best_program,
            nothing, shared_inputs,
            meta_library, run_config,
            node_config, "checkpoint_$iteration.pickle")

    end
end
checkpoit_10 = checkpoint(5)

### CUT HERE FOR DESERIALIZATION

function fit_ga_atari(
    shared_inputs::SharedInput,
    genomes::Vector{UTGenome},
    reps::Int,
    model_architecture::modelArchitecture,
    node_config::nodeConfig,
    run_config::RunConfGA,
    meta_library::MetaLibrary,
    # Callbacks before training
    pre_callbacks::UTCGP.Optional_FN,
    # Callbacks before step (before looping through data)
    population_callbacks::UTCGP.Mandatory_FN,
    mutation_callbacks::UTCGP.Mandatory_FN,
    output_mutation_callbacks::UTCGP.Mandatory_FN,
    decoding_callbacks::UTCGP.Mandatory_FN,
    # Callbacks per step (while looping through data)
    endpoint_callback::Type{<:BatchEndpoint},
    final_step_callbacks::UTCGP.Optional_FN,
    # Callbacks after step ::
    elite_selection_callbacks::UTCGP.Mandatory_FN,
    epoch_callbacks::UTCGP.Optional_FN,
    early_stop_callbacks::UTCGP.Optional_FN,
    last_callback::UTCGP.Optional_FN,
) # Tuple{UTGenome, IndividualPrograms, GenerationLossTracker}::

    local early_stop, best_programs, elite_idx, _, ind_performances, =
        UTCGP._ga_init_params(genomes[1], run_config)
    local population = UTCGP.Population(deepcopy(genomes))

    # PRE CALLBACKS
    UTCGP._make_pre_callbacks_calls(pre_callbacks)
    M_gen_loss_tracker = UTCGP.GenerationLossTracker()

    for iteration = 1:run_config.generations
        early_stop ? break : nothing
        @warn "Iteration : $iteration"
        # Population
        ga_pop_args = UTCGP.GA_POP_ARGS(
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            ind_performances,
        )
        population, time_pop =
            @unwrap_or UTCGP._make_ga_population(ga_pop_args, population_callbacks) throw(
                "Could not unwrap make_population",
            )

        # Program mutations ---
        ga_mutation_args = GA_MUTATION_ARGS(
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            shared_inputs,
        )
        population, time_mut =
            @unwrap_or UTCGP._make_ga_mutations!(ga_mutation_args, mutation_callbacks) throw(
                "Could not unwrap make_ga_mutations",
            )

        # Genotype to Phenotype mapping --- 
        population_programs, time_pop_prog = UTCGP._make_decoding(
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            shared_inputs,
            decoding_callbacks,
        )
        @show length(population_programs)

        @warn "Graphs evals"
        pop_size = length(population)
        Repetitions_loss_tracker = Matrix{Float64}(undef, pop_size, reps)
        UTCGP.reset_programs!(population_programs)
        for ith_repetition = 1:reps
            @info "Repetition n° $ith_repetition"
            UTCGP.reset_programs!(population_programs)
            M_individual_loss_tracker = UTCGP.IndividualLossTracker()
            local fitness
            endpoint_holder = PongCompetition.PythonCall.GIL.unlock() do
                fitness = endpoint_callback(
                    population_programs, model_arch, meta_library, iteration
                )
            end
            fitness_values = get_endpoint_results(fitness)
            fitness_values = reduce(vcat, fitness_values)
            UTCGP.add_pop_loss_to_ind_tracker!(M_individual_loss_tracker, fitness_values)  # appends the loss for the ith x sample to the
            ind_performances = UTCGP.resolve_ind_loss_tracker(M_individual_loss_tracker)
            Repetitions_loss_tracker[:, ith_repetition] .= ind_performances
        end
        ind_performances = mean(Repetitions_loss_tracker, dims=2)[:] # average over the 2nd dim => the repetitions

        # Resetting the population (removes node values)
        [reset_genome!(g) for g in population]

        # final step call...
        if !isnothing(final_step_callbacks)
            for final_step_callback in final_step_callbacks
                UTCGP.get_fn_from_symbol(final_step_callback)()
            end
        end

        # Elite selection callbacks
        @warn "Selection"
        # TODO added this
        println("Handing over lowest fitness of each individual over all seeds for selection")
        ga_selection_args = GA_SELECTION_ARGS(
            ind_performances,
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            population_programs,
        )
        elite_idx, time_elite = @unwrap_or UTCGP._make_ga_elite_selection(
            ga_selection_args,
            elite_selection_callbacks,
        ) throw("Could not unwrap make_ga_selection")

        elite_fitnesses = ind_performances[elite_idx]
        elite_best_fitness = minimum(skipmissing(elite_fitnesses))
        elite_best_fitness_idx = elite_fitnesses[1]
        elite_avg_fitness = mean(skipmissing(elite_fitnesses))
        elite_std_fitness = std(filter(!isnan, ind_performances))
        best_programs = population_programs[elite_idx]

        reset_programs!.(best_programs)
        # println("BEST : $(best_programs[1])")
        println("BEST : $(UTCGP.general_hasher_sha(best_programs[1]))")

        try
            histogram(ind_performances) |> println
        catch e
            @error "Could not drawn histogram"
        end

        # Subset Based on Elite IDX---
        sorted_elite = elite_idx[sortperm(ind_performances[elite_idx])]
        @show elite_idx
        @show sorted_elite
        @show ind_performances[elite_idx]
        old_pop = deepcopy(population.pop[sorted_elite])
        empty!(population.pop)
        push!(population.pop, old_pop...)
        ind_performances = ind_performances[sorted_elite]

        # EPOCH CALLBACK
        batch = []
        if !isnothing(epoch_callbacks)
            UTCGP._make_epoch_callbacks_calls(
                ind_performances,
                population,
                iteration,
                run_config,
                model_architecture,
                node_config,
                meta_library,
                shared_inputs,
                population_programs,
                elite_fitnesses,
                best_programs,
                elite_idx,
                view(batch, :),
                epoch_callbacks,)
        end

        # store iteration loss/fitness
        UTCGP.affect_fitness_to_loss_tracker!(M_gen_loss_tracker, iteration, elite_best_fitness)
        println(
            "Iteration $iteration. 
            Best fitness: $(round(elite_best_fitness, digits = 10)) at index $elite_best_fitness_idx 
            Elite mean fitness : $(round(elite_avg_fitness, digits = 10)). Std: $(round(elite_std_fitness)) at indices : $(elite_idx)",
        )

        flush(stdout)
        flush(stderr)

        if !isnothing(early_stop_callbacks) && length(early_stop_callbacks) != 0
            early_stop_args = UTCGP.GA_EARLYSTOP_ARGS(
                M_gen_loss_tracker,
                M_individual_loss_tracker,
                ind_performances,
                population,
                iteration,
                run_config,
                model_architecture,
                node_config,
                meta_library,
                shared_inputs,
                population_programs,
                elite_fitnesses,
                best_programs,
                elite_idx,
            )

            early_stop =
                UTCGP._make_ga_early_stop_callbacks_calls(early_stop_args, early_stop_callbacks) # true if any
        end

        if early_stop
            g = run_config.generations
            @warn "Early returning at iteration : $iteration from $g total iterations"
            if !isnothing(last_callback)
                last_callback(
                    ind_performances,
                    population,
                    iteration,
                    run_config,
                    model_architecture,
                    node_config,
                    meta_library,
                    population_programs,
                    elite_fitnesses,
                    best_programs,
                    elite_idx,
                )
            end
            # UTCGP.show_program(program)
            return tuple(deepcopy(population[1]), best_programs, M_gen_loss_tracker)
        end
        GC.gc(true)
    end
    @info "Returning from fit"
    return (deepcopy(population[1]), best_programs, M_gen_loss_tracker)
end

pyexec("import numpy as np; np.random.seed(0)", PongCompetition)

best_genome, best_programs, gen_tracker = fit_ga_atari(
    shared_inputs,
    pop,
    1,
    model_arch,
    node_config,
    run_conf,
    ml,
    # Callbacks before training
    nothing,
    # Callbacks before step
    (:ga_population_callback,),
    (:ga_numbered_new_material_mutation_callback,),
    (:ga_output_mutation_callback,),
    (:default_decoding_callback,),
    # Endpoints
    EnvpoolAtariEndpoint,
    # STEP CALLBACK
    nothing,
    # Callbacks after step
    (:ga_elite_selection_callback,),
    # Epoch Callback
    (atari_tracker, checkpoit_10),#nothing,
    # Final callbacks ?
    nothing,
    nothing
)

save_payload(best_genome,
    best_programs,
    gen_tracker,
    shared_inputs,
    ml, run_conf,
    node_config,
    "checkpoint_final.pickle")


