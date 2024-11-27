using Distributions

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
    x̃min::Real=-Inf, check_inputs=true)

    check_inputs && _check_pmf(values, pmf)
    β < zero(β) && _bad_risk("Risk level β must be non-negative.")

    if iszero(β)
        return values' * pmf
    elseif isinf(β) && β > zero(β)
        return essinf(values, pmf; check_inputs=false).value
    end
    # because entropic risk measure is translation equivariant, we can change values
    # so that it is positive. That change makes it less likely that it overflows
    x̃min = isfinite(x̃min) ? x̃min : minimum(values)

    #@fastmath x̃min-one(β) / β*log(sum(pmf .* exp.(-β .* (values .- x̃min) ))) |> float
    a = Iterators.map((x, p) -> (@fastmath p * exp(-β * (x - x̃min))), values, pmf) |> sum
    @fastmath x̃min - one(β) / β * log(a)
end

ERM(x̃, β::Real; kwargs...) = ERM(rv2pmf(x̃)..., β; kwargs...)

"""
    softmin(x̃, β; x̃min check_inputs = true)

Compute a weighted softmin function for random variable `x̃` with risk level `β`. 
This can be seen as an approximation of the arg min function and not the min function.

The operator computes a distribution p such that
```math
p_i = \\frac{e^{-β x_i}}{\\mathbb{E}[e^{-β x̃}]}
```

The optional `x̃min` parameter is used as an offset in order to avoid overflows
when computing the exponential function. If not provided, the minimum value
of `x̃` is used instead. 

The value β must be positive
"""
function softmin end

function softmin(values::AbstractVector{<:Real}, pmf::AbstractVector{<:Real},
    β::Real; x̃min::Real=-Inf, check_inputs=true)

    check_inputs && _check_pmf(values, pmf)

    if β > zero(β)
        # TODO: only needs to look at values with pos prob.
        # because softmin is translation invariant add the subtract the smallest value
        # ensures that values ≥ 0
        if isfinite(x̃min)
            np = similar(pmf)
            np .= @fastmath pmf .* exp.(-β .* (values .- x̃min))
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

softmin(x̃, β::Real; kwargs...) = softmin(rv2pmf(x̃)..., β; kwargs...)
