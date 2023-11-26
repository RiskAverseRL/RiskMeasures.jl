"""
    ERM(x̃, p::Distribution, β [, x̃min])

Compute the entropic risk measure of the random variable `x̃` represented as 
a vector over outcomes and distributed according to the measure `p` with risk level `β`. 

The optional `x̃min` parameter is used as an offset in order to avoid overflows
when computing the exponential function. If not provided, the minimum value
of `x̃` is used instead. 

Assumes a maximization problem. Using `β=0` computes expectation
and `β=Inf` computes the essential infimum (smallest value with positive probability).

More details: https://en.wikipedia.org/wiki/Entropic_risk_measure
"""
function ERM end

ERM(x̃::AbstractVector{<:Real}, β::Real) = ERM(x̃, uniform(length(x̃)), β)
ERM(x̃::AbstractVector{<:Real}, p, β) = ERM(x̃, p, β, minimum(x̃))
ERM(x̃::AbstractVector{<:Real}, p::AbstractVector{T}, β::Real, x̃min::Real) where {T<:Real} =
    ERM(x̃, Distribution{T}(p), β, x̃min)

function ERM(x̃::AbstractVector{<:Real}, p::Distribution{<:Real}, β::Real, x̃min::Real)
    length(x̃) == length(p) || _bad_distribution("Lengths of x̃ and p must match.")
    length(x̃) > 0 || _bad_risk("x̃ must have some elements")
    if iszero(β)
        mean(x̃, p) |> float
    elseif isinf(β) && β > zero(β)
        minimum(@inbounds x̃[i] for i ∈ 1:length(x̃) if !iszero(p.p[i])) |> float
    elseif zero(β) ≤ β 
        # because entropic risk measure is translation equivariant, we can change x̃ so that
        # it is positive. That change makes it less likely that it overflows
        if isfinite(x̃min)
            #@fastmath x̃min-one(β) / β*log(sum(p.p .* exp.(-β .* (x̃ .- x̃min) ))) |> float
            a = Iterators.map((x,p) -> (@fastmath p * exp(-β * (x - x̃min))), x̃, p.p) |> sum
            @fastmath x̃min - one(β) / β * log(a) |> float
        else
            NaN |> float
        end
    else
        _bad_risk("Risk level β must be non-negative.")
    end
end

"""
    softmin(x̃, p::Distribution, β [, x̃min]; check_inputs = true)

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

softmin(x̃, p, β) = softmin(x̃, p, β, minimum(x̃))
softmin(x̃::AbstractVector{<:Real}, p::AbstractVector{T}, β::Real) where {T<:Real} =
    softmin(x̃, Distribution{T}(p), β)
softmin(x̃::AbstractVector{<:Real}, β::Real) = softmin(x̃, uniform(length(x̃)), β)

function softmin(x̃::AbstractVector{<:Real}, p::Distribution{T}, β::Real, x̃min::Real) where
    {T <: Real}
    
    length(x̃) == length(p) || _bad_distribution("Lengths of x̃ and p must match.")
    length(x̃) > 0 || _bad_risk("x̃ must have some elements")

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
