"""
    ERM(x̃, β [, x̃min, check_inputs = true])

Compute the entropic risk measure of the random variable `x̃` with risk level `β`. 

The optional `x̃min` parameter is used as an offset in order to avoid overflows
when computing the exponential function. If not provided, the minimum value
of `x̃` is used instead. 

Assumes a maximization problem. Using `β=0` computes expectation
and `β=Inf` computes the essential infimum (smallest value with positive probability).

More details: https://en.wikipedia.org/wiki/Entropic_risk_measure
"""
function ERM end

function ERM(values::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, β::Real;
             x̃min::Real = -Inf, check_inputs = true)
    
    check_inputs && _check_pmf(values, pmf)
    β < zero(β) && _bad_risk("Risk level β must be non-negative.")

    if iszero(β)
        return dot(x̃, pmf) 
    elseif isinf(β) && β > zero(β)
        return minimum(@inbounds x̃[i] for i ∈ 1:length(x̃) if !iszero(p.p[i])) |> float
    end 
    # because entropic risk measure is translation equivariant, we can change x̃ so that
    # it is positive. That change makes it less likely that it overflows
    x̃min = isfinite(x̃min) ? x̃min : minimum(values)
    
    #@fastmath x̃min-one(β) / β*log(sum(p.p .* exp.(-β .* (x̃ .- x̃min) ))) |> float
    a = Iterators.map((x,p) -> (@fastmath p * exp(-β * (x - x̃min))), x̃, pmf) |> sum
    @fastmath x̃min - one(β) / β * log(a) 
end

ERM(x̃::DiscreteNonParametric, β::Real; kwargs...)  =
    ERM(support(x̃), probs(x̃), β; kwargs...)

"""
    softmin(x̃, p::Distribution, β; x̃min check_inputs = true)

Compute a weighted softmin function for random variable `x̃` represented as 
a vector over outcomes and distributed according to the measure `p` with risk level `β`. 
This can be seen as an approximation of the arg min function and not the min function.

The operator computes a distribution p such that
`` p_i = \frac{e^{-β x_i}}{E[e^{-β x̃}]} ``

The optional `x̃min` parameter is used as an offset in order to avoid overflows
when computing the exponential function. If not provided, the minimum value
of `x̃` is used instead. 

The value β must be positive
"""
function softmin end

function softmin(values::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, β::Real;
                 x̃min::Real) 
    
    check_inputs && _check_pmf(values, pmf)

    if β > zero(β)
        #  TODO: only needs to look at x̃ with pos prob.
        # because softmin is translation invariant add the subtract the smallest value
        # ensures that x̃ ≥ 0
        if isfinite(x̃min) 
            np = similar(p.p)
            np .= @fastmath p.p .* exp.(-β .* (x̃ .- x̃min) )
            np .= @fastmath inv(sum(np)) .* np
            mapreduce(isfinite, &, np) || error("Overflow, reduce β.")
            np
        else
            error("Input must be finite.")
        end
    else
        _bad_risk("Risk level β must be positive (>0).")
    end
end

softmin(x̃::DiscreteNonParametric, β::Real; kwargs...)  =
    softmin(support(x̃), probs(x̃), β; kwargs...)
