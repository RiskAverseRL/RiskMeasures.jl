module RiskMeasures

using DispatchDoctor: @stable


#@stable default_mode = "disable" begin
@stable begin
    include("general.jl")
    include("essinf.jl")
    include("erm.jl")
    include("var.jl")
    include("cvar.jl")
    include("evar.jl")
    include("expectile.jl")
end

export essinf, ERM, softmin, VaR, CVaR, EVaR, expectile

end
