module RiskMeasures

include("general.jl")

include("essinf.jl")
export essinf

include("erm.jl")
export ERM

include("var.jl")
export VaR

include("cvar.jl")
export CVaR
    
include("evar.jl")
export EVaR


end
