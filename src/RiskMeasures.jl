module RiskMeasures

include("general.jl")

include("essinf.jl")
export essinf, essinf_e

include("erm.jl")
export ERM, softmin

include("var.jl")
export VaR, VaR_e

include("cvar.jl")
export CVaR, CVaR_e

include("evar.jl")
export EVaR, EVaR_e

include("expectile.jl")
export expectile, expectile_e

end
