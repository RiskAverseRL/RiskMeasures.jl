module RiskMeasures

include("general.jl")

include("essinf.jl")
export essinf, essinf

include("erm.jl")
export ERM, softmin

include("var.jl")
export VaR, VaR

include("cvar.jl")
export CVaR, CVaR

include("evar.jl")
export EVaR, EVaR

include("expectile.jl")
export expectile

end
