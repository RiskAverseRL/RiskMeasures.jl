
"""
    CVaR(x̃, α)

Compute the conditional value at risk at level `α` for the random variable `x̃` specified
as a type of Distribution.

The risk level `α` must satisfy the ``α∈[0,1]``. Risk aversion increases with an increasing `α` and, `α=0` represents the expectation,  `α=1` computes the essential infimum (smallest value with positive probability).

Assumes a reward maximization setting and solves the dual form
``\\min_{q ∈ Q} q^T x̃``
where
``Q = {q ∈ Δ^n : q_i ≤ p_i/(1-α)}``
and ``Δ^n`` is the probability simplex. 

Returns a named tuple with `value` and the `distribution` that solves the robust
CVaR formulation such that ``\\mathbb{E}_{x̃\\sim pc}{x̃}`` equals to the CVaR value.

More details: https://en.wikipedia.org/wiki/Expected_shortfall
"""
function CVaR end


"""
    CVaR(values, pmf, α; ...) 

Compute CVaR for a discrete random variable with `values` and the probability mass
function `pmf`. See `CVaR(x̃, α)` for more details.
"""
function CVaR(values::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real;
              check_inputs = true) 

    if check_inputs
        _check_α(α)
        _check_pmf(values, pmf)
    end

    # ensure return type stability
    Tval = float(eltype(values)) # construct the return type, which must be a float
    Tdst = typeof(pmf)

    # handle special cases
    if iszero(α)
        return (value = dot(values, pmf) |> Tval :: Tval,
                distribution = distribution ? pmf : empty(pmf) :: Tdst)
    elseif isone(α)
        return essinf(xvalues, pmf; check_inputs = false, distribution = distribution)
    end

    # Here on: α ∈ (0,1)
    pc = zeros(eltype(pmf), length(pmf))  # this is the new distribution
    value = zero(Tval)                  # CVaR value
    p_left = one(eltype(pmf))           # probabilities left for allocation
    α̂ = one(α) - α                      # probabilities to allocate

    # Efficiency note: sorting by values is O(n*log n);
    # quickselect is O(n) and would suffice but would need be based on quantile
    sortedi = sort(eachindex(values, pmf); by=(i->@inbounds values[i]))
    @inbounds for i ∈ sortedi
        # update index's probability and probability left to sum to 1.0
        increment = min(pmf[i] / α̂, p_left)
        # update  return values
        pc[i] = increment
        value += increment * values[i]
        p_left -= increment
        p_left ≤ zero(p_left) && break 
    end
    return (value = value |> Tval :: Tval,
            dist = pc :: Tdst)
end

CVaR(x̃::Discrete, α::Real; kwargs...) = CVaR(support(x̃), probs(x̃), α; kwargs...)

