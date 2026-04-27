using Distributions

# linear-time implementation of CVaR
function qCVaR!(vals::AbstractVector{<:Real}, p::AbstractVector{<:Real}, α::Real)

    T = float(eltype(vals))
    q, _ = qql!(copy(vals), copy(p), α)
    p_left =  one(T) - sum(p[i] for i in eachindex(p) if vals[i] < q; init=zero(T)) / α

    p_left ≥ 0 || error("CVaR internal error: negative remaining probability")

    pc = zeros(T, length(p))
    value = zero(T)
    
    @inbounds for i in eachindex(vals, p)
        vals[i] > q && continue
        if vals[i] < q
            pc[i]  = p[i] / α
            value += pc[i] * vals[i]
        else
            increment  = min(p[i] / α, p_left)
            pc[i]      = increment
            value     += increment * vals[i]
            p_left    -= increment
        end
    end
    return (value=value, pmf=pc)
end

"""
    CVaR(x̃, α)

Compute the conditional value at risk at level `α` for the random variable `x̃`.

    CVaR(values, pmf, α; ...) 

Compute CVaR for a discrete random variable with `values` and the probability mass
function `pmf`. 

The risk level `α` must satisfy the ``α ∈ [0,1]``. Risk aversion decreases with
an increasing `α` and `α = 1` represents the expectation,
`α = 0` computes the essential infimum (smallest value with positive probability).

Assumes a reward maximization setting and solves the dual form
```math
\\min_{q ∈ \\mathcal{Q}} q^T x̃
```
where
```math
\\mathcal{Q} = \\left\\{q ∈ Δ^n : q_i ≤ \\frac{p_i}{α}\\right\\}
```
and ``Δ^n`` is the probability simplex, and ``p`` is the distribution of ``x̃``. 

# Returns

A named tuple with CVaR `value` and the `pmf` that achieves it.

# Keyword Arguments:

- `check_inputs=true`: check that the inputs are valid.
- `fast=false`: use linear-time experimental implementation 

More details: <https://en.wikipedia.org/wiki/Expected_shortfall>
"""
function CVaR end

function CVaR(values::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real;
              check_inputs=true, fast=false)
    check_inputs && _check_α(α)
    check_inputs && _check_pmf(values, pmf)

    T = float(eltype(pmf))

    # handle special cases
    if isone(α)
        return (value=values' * pmf, pmf=Vector(pmf))
    elseif iszero(α)
        minval = essinf(values, pmf; check_inputs=false)
        minpmf = zeros(T, length(pmf))
        minpmf[minval.index] = one(T)
        return (value=T(minval.value), pmf=minpmf)
    end

    if !fast
        # Here on: α ∈ (0,1)
        pc = zeros(T, length(pmf))  # this is the new distribution
        value = zero(T)                  # CVaR value
        p_left = one(T)           # probabilities left for allocation
        α̂ = α                      # probabilities to allocate

        # Efficiency note: sorting by values is O(n*log n);
        # quickselect is O(n) and would suffice but would need be based on quantile
        sortedi = sortperm(values)
        @inbounds for i ∈ sortedi
            # update index's probability and probability left to sum to 1.0
            increment = min(pmf[i] / α̂, p_left)
            # update  return values
            pc[i] = increment
            value += increment * values[i]
            p_left -= increment
            p_left ≤ zero(p_left) && break
        end
        return (value=value, pmf=pc)
    else
        qCVaR!(copy(values), copy(pmf), α)
    end
        
end

function CVaR(x̃, α::Real; kwargs...)
    supp, pmf = rv2pmf(x̃)
    v1 = CVaR(supp, pmf, α; kwargs...)
    ỹ = DiscreteNonParametric(supp, v1.pmf)
    (value=v1.value, pmf=ỹ)
end


