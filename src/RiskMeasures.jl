module RiskMeasures

using DispatchDoctor: @stable


@stable default_mode = "disable" begin
#@stable begin
    include("general.jl")
    include("essinf.jl")
    include("erm.jl")
    include("var.jl")
    include("cvar.jl")
    include("evar.jl")
    include("expectile.jl")
    include("choquet.jl")
    include("ubsr.jl")
end

export essinf, ERM, softmin, VaR, CVaR, EVaR, expectile, UBSR
export choquet_risk, closure_c, cvar_capacity
export choquet_distortion_risk, cvar_distortion

end
