using PythonCall
using Pkg
using ImageCore

const PYLOCK = ReentrantLock()

function gc_()
    PythonCall.GIL.lock(PythonCall.GC.gc)
end

function pycall_lock(f::Function)
    lock(PYLOCK)
    try
        r = PythonCall.GIL.lock(f)
        return r
    finally
        unlock(PYLOCK)
    end
end


function __init__()
    Pkg.build("PythonCall")
    pyexec(
        """
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        """,
        PongCompetition
    )
end

# API
function create_env(
    name::String,
    seed::Int,
    max_episode_steps::Int,
    img_height::Int=84,
    img_width::Int=84,
    stack_num::Int=4,
)
    env = PythonCall.pyexec(@NamedTuple{env::Any}, """
       env = gym.make(id = task_id, max_episode_steps = max_episode_steps, repeat_action_probability = 0.0, frameskip = 1)
       env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30, 
        frame_skip=4, 
        terminal_on_life_loss=False,
        screen_size=(img_width, img_height), 
        grayscale_obs=True, 
        grayscale_newaxis=False, 
        scale_obs = True
        )
       env = gym.wrappers.FrameStackObservation(env, stack_size=stack_num)

       env.reset(seed = seed)
       """, PongCompetition,
        (task_id=name,
            seed=seed,
            max_episode_steps=max_episode_steps,
            img_height=img_height,
            img_width=img_width,
            stack_num=stack_num,
        ))
    env.env
end

function pool(name::String,
    seeds::Vector{Int},
    max_episode_steps::Int=27000,
    img_height::Int=84,
    img_width::Int=84,
    stack_num::Int=4,
)
    ENVS = [create_env(name, seed, max_episode_steps, img_height, img_width, stack_num) for seed in seeds]
    ENVS
end

function pool_of_pools(name::String,
    num_envs::Int,
    seeds::Vector{Int},
    max_episode_steps::Int=27000,
    img_height::Int=84,
    img_width::Int=84,
    stack_num::Int=4,)
    @info "Creating $num_envs pools"
    ENVS = [pool(name, seeds, max_episode_steps, img_height, img_width, stack_num) for i in 1:num_envs]
    ENVS
end

function reset(envs::Vector{PythonCall.Py}, seeds::Vector{Int})
    for (env, seed) in zip(envs, seeds)
        env.reset(seed=seed)
    end
end
function reset(envs::Vector{Vector{PythonCall.Py}}, seeds::Vector{Int})
    for env in envs
        reset(env, seeds)
    end
end

function step!(env::PythonCall.Py, action::Int, state, rew, term, trunc, info)
    s, r, t1, t2, i = env.step(action)
    push!(state, convert.(N0f8, pyconvert(Array, s)))
    push!(rew, pyconvert(Float64, r))
    push!(term, pyconvert(Bool, t1))
    push!(trunc, pyconvert(Bool, t2))
    push!(info, i)
end
function step(envs::Vector{PythonCall.Py}, actions::Vector{Int})
    state, rew, term, trunc, info = [], [], [], [], []
    for (env, action) in zip(envs, actions)
        step!(env, action, state, rew, term, trunc, info)
    end
    state, rew, term, trunc, info
end

function step(envs::Vector{PythonCall.Py}, actions::Vector{Int}, mask::Vector{Int})
    state, rew, term, trunc, info = [], [], [], [], []
    # @show length(envs)
    # @show length(mask)
    @assert length(mask) <= length(envs)
    @assert length(mask) == length(actions)
    for (idx, action) in zip(mask, actions)
        env = envs[idx]
        step!(env, action, state, rew, term, trunc, info)
    end
    state, rew, term, trunc, info
end

export create_env, pool, reset, step, pool_of_pools
