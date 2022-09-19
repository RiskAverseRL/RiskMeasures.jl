using BenchmarkTools
using Revise
import RiskMeasures

X :: Vector{Float64} = collect(1.0:1000.0)
p :: Vector{Float64} = collect(1000.0:-1:1.0)
p .= p ./ sum(p)
Xmin = minimum(X)

function erm(X,p,α)
    erm(X,p,α,minimum(X))
end	      

function erm(X,p,α,Xmin)
    #Xmin - one(α) / α * log(sum(p .* exp.(-α .* (X .- Xmin) ))) |> float
    a = Iterators.map((x,p) -> (@fastmath p * exp(α * (x - Xmin))), X, p) |> sum
    Xmin - one(α) / α * log(a)
end	      

(@benchmark erm($X, $p, 0.5)) |> display
(@benchmark RiskMeasures.erm($X, $p, 0.5)) |> display
(@benchmark RiskMeasures.evar($X, $p, 0.5)) |> display
(@benchmark RiskMeasures.cvar($X, $p, 0.5)) |> display
