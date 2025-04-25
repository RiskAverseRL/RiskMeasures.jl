"""
    VaR(x̃, α) 

Compute the value at risk at risk level `α` for the random variable `x̃`.

Risk must satisfy ``α ∈ [0,1]`` and `α=0.5` computes the median and `α=1` computes the 
essential infimum (smallest value with positive probability) and `α=0` returns infinity.

Solves for
``\\inf \\{x ∈ \\mathbb{R} : \\mathbb{P}[x̃ ≤ x] > 1-α \\}``

In general, this function is neither convex nor concave in the random variable x̃.
"""
function VaR end

"""
    VaR(values, pmf, α; ...) 

Compute VaR for a discrete random variable with `values` and the probability mass
function `pmf`.  Also compute the index that achieves the value at risk. 



If `α = 0`, VaR returns the essential infimum, and if `α = 1`, VaR returns
maximum possible value, becuase VaR_1 is infinity.

Runs in ``n \\log(n)`` time where `n = length(x̃)`.


# Returns

A named tuple with VaR `value` and the `index` that achieves it. If such
an index does not exist, then returns -1.

# Keyword Arguments:

- `check_inputs=true`: check that the inputs are valid.
- `fast=false`: use linear-time experimental implementation 
"""
function VaR(values::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real;
    check_inputs=true, fast = false)

    _check_α(α)
    check_inputs && _check_pmf(values, pmf)

    T = eltype(pmf)
    # special cases
    if isone(α) # unbounded value
        return (value=typemax(T), index=-1)
    elseif iszero(α) # maximum (it is unbounded)
        return essinf(values, pmf; check_inputs=check_inputs)
    end

    if !fast
        sortedi = sortperm(values; rev = true) # sort descending
        pos = last(sortedi) # this value is used when the loop does not break
        p_accum = one(T)

        α̂ = α 
        # find the index such that the sum of the probabilities is greater than alpha
        @inbounds for i ∈ sortedi
            p_accum -= pmf[i]
            p_accum ≤ α̂ && (pos = i; break)
        end
        return (value=values[pos], index=pos)
    else 
        qql!(copy(values), copy(pmf), α)
    end
end


function VaR(x̃, α::Real; kwargs...)
    supp, pmf = rv2pmf(x̃)
    v1 = VaR(supp, pmf, α; kwargs...)
    # construct the equivalent distribution
    Tp = eltype(pmf)
    Ts = eltype(supp)
    if v1.index > 0
        vpmf = zeros(Tp, length(pmf))
        vpmf[v1.index] = one(Tp)
        ỹ = DiscreteNonParametric(supp, vpmf)
    else # happens then VaR is infinite
        ỹ = DiscreteNonParametric([typemax(Ts)], [one(Tp)])
    end

    (value=v1.value, pmf=ỹ)
end
