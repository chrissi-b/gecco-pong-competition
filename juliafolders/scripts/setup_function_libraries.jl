JULIA_CONDAPKG_OFFLINE = true

include("src/PongCompetition.jl")
using .PongCompetition

using PythonCall
NP = pyimport("numpy")

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


# GAME SETTINGS #
CROPS = Dict(
    "pong" => (15:77, 1:84)
)
GAMENAME = "Pong-v4"
GAME = "pong"
const CROP = CROPS[GAME]
ACTIONS = [0, 4, 5]


# INITIALISE EXAMPLE ENVIRONMENT #
seed = 1
env_ex = pool(GAMENAME, [seed])
env_ex[1].reset(seed=seed)
obs_, rew_, term_, trunc_, info_ = PongCompetition.step(env_ex, [0]);

x_example = obs_[1]
example = x_example[1, :, :]
example = example[CROP[1], CROP[2]]
IMAGE_TYPE = typeof(SImageND(example))
IMG_SIZE = example |> size
PROB_ACTION = false

file = @__FILE__
home = dirname(dirname(file))
println("home $home")

include(joinpath(home, "scripts", "utils.jl"))
include(joinpath(home, "scripts", "extra_fns.jl"))

action_mapping_dict = Dict(
    "pong" => [3, pong_action_mapping],
)

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

# Bundles Float
float_bundles = UTCGP.get_float_bundles_atari()
push!(float_bundles, float_comparetuples.bundle_float_comparetuples)

# POSITIONS 
positions_bundles = [deepcopy(tupleintint_2Dposition.bundle_tupleintint_2Dposition)]

# Libraries
lib_image2D = Library(image2D)
lib_float = Library(float_bundles)
lib_positions = Library(positions_bundles)

lib_number_to_img = Library(deepcopy([float_bundles[7], float_bundles[8], float_bundles[9]]))
UTCGP.unpack_bundles_in_library!(lib_number_to_img)

ml = MetaLibrary([lib_image2D, lib_float, lib_positions])

# INITIALISE GAME ENVIRONMENT #

seed = 7
env_ex = pool(GAMENAME, [seed])
env_ex[1].reset(seed=seed)
obs_, rew_, term_, trunc_, info_ = PongCompetition.step(env_ex, [0]);


""" Initialise variables """
state_view = view(obs_, [1])
previous_obs = state_view

s = size(previous_obs[1][1, CROP[1], CROP[2]])
S = Tuple{s[1],s[2]}

inputs = previous_obs[1]
ins = Any[SImageND(inputs[t, CROP[1], CROP[2]], S) for t in axes(inputs, 1)]

actions = []

reward_total = 0
points_own = 0
points_opponent = 0

actions_after_init = 0

img_counter = 1

