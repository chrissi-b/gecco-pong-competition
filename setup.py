# run file to setup julia virtual environment

import juliacall
from juliacall import Pkg as jlPkg

jlPkg.develop(path="dev/SearchNetworks.jl")
jlPkg.develop(path="dev/MAGE.jl")


