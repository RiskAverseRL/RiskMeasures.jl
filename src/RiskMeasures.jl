module RiskMeasures

include("general.jl")

include("essinf.jl")
export essinf

include("erm.jl")
export ERM, softmin

include("var.jl")
export VaR

include("cvar.jl")
export CVaR

include("evar.jl")
export EVaR

include("expectile.jl")
export expectile

end
