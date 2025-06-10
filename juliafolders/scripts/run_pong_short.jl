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

file = @__FILE__
home = dirname(dirname(file))
println("home $home")

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
#mkdir(folder)

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


