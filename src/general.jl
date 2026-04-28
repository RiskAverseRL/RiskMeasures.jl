using Distributions: DiscreteNonParametric, DiscreteAffineDistribution

_bad_risk(msg::AbstractString) =
    error(msg)
_bad_distribution(msg::AbstractString) =
    error(msg)

# checks a pmf for correctness
function _check_pmf(pmf::AbstractVector{<:Real})
    T = eltype(pmf)
    !isempty(pmf) || _bad_distribution("pmf must be non-empty.")
    mapreduce(isfinite, &, pmf) || _bad_distribution("Probabilities must be finite.")
    for element ∈ pmf
      element >= zero(T) || _bad_distribution("Probabilities must be non-negative.")
    end
    sum(pmf) ≈ one(T) || _bad_distribution("Probabilities must sum to 1 and not $(sum(pmf)).")
end

function _check_pmf(support::AbstractVector{<:Real}, pmf::AbstractVector{<:Real})
    length(support) == length(pmf) ||
        _bad_distribution("Lengths of support and pmf must have the same size.")
    _check_pmf(pmf)
end

function _check_α(α::Real)
    zero(α) ≤ α ≤ one(α) || _bad_risk("Risk level α must be in [0,1].")
end

# converts a random variable to a support and a probability mass function
function rv2pmf(x̃::DiscreteNonParametric)
    sp = support(x̃)
    (sp, pdf.(x̃, sp))
end

function rv2pmf(x̃::DiscreteAffineDistribution)
    # needs to handle location-scale separately because
    # the implementation of the pdf function in Distributions.LocationScale
    # leads to numerrical errors

    sp = support(x̃.ρ)
    pmf = pdf.(x̃.ρ, sp)
    sp = @. sp * x̃.σ + x̃.μ
    (sp, pmf)
end

# swaps the elements of vals between 
function swap!(vals::AbstractVector{<:Real}, p::AbstractVector{<:Real}, i::Int, j::Int)
    if i != j
        vals[i], vals[j] = vals[j], vals[i]
        p[i], p[j] = p[j], p[i]
    end
    return nothing
end

"""
    partition!(vals::AbstractVector{<:Real}, p::AbstractVector{<:Real}, pivot_ind::Int, f::Int, b::Int)

Partition the values in `vals` and `p` between `f` and `b` (inclusive) using the
Dutch Flag algorithm into less than pivot, equal to pivot, and greater than pivot. 

# Returns

A tuple (lt, gt) where `i < lt => x[i] < pivot_val` and `i > gt => x[i] > pivot_val`
and `lt ≤ i ≤ gt => x[i] = pivot_val`, where `pivot_val = vals[pivot_ind]`.
"""
function partition!(vals::AbstractVector{<:Real}, p::AbstractVector{<:Real}, pivot_ind::Int, f::Int, b::Int)
    pivot_val = vals[pivot_ind]
    lt = f
    eq = f
    gt = b
    @inbounds while eq <= gt
        if vals[eq] < pivot_val
            @inbounds swap!(vals, p, eq, lt)
            lt += 1
            eq += 1
        elseif vals[eq] > pivot_val
            @inbounds swap!(vals, p, eq, gt)
            gt -= 1
        else # vals[eq] == pivot_val
            eq += 1
        end
    end
    return (lt=lt, gt=gt)
end

"""
    qql!(vals, p, α)

Compute VaR in expected linear time without performing correctness checks. The runtime
of the algorithm is randomized but the output value is deterministic.

The input must satisfy `0 < α < 1` and `p` and `vals` must have the same length.

# Returns

A named tuple with VaR `value` as a float and the `index` that achieves it.
"""
function qql!(vals::AbstractVector{<:Real}, p::AbstractVector{<:Real}, α::Real)
    0 < α < 1 || _bad_risk("Violated: 0 < α < 1")
    length(p) == length(vals) ||
        _bad_distribution("Violated: length(p) == length(vals)")
    
    f = 1
    b = length(vals)
    @inbounds while b - f ≥ 1
        pivot_ind = rand(f:b)
        l, g = partition!(vals, p, pivot_ind, f, b)
        t = sum(view(p, f:(l-1)))
        e = sum(view(p, l:g))
        if α < t   
            b = l - 1
        elseif t + e ≤ α
            f = g + 1
            α -= t + e
        else # t + e > α ≥ t means we found it!
            f = b = g
        end
    end
    return (value=float(vals[b]), index=b)
end


# a function solely used to check if the stability checks work
@stable function test_stability(x :: Integer)
    if x < 0
        return float(x)
    else
        return x
    end
end
