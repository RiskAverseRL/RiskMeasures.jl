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
    mapreduce(x -> x ≥ zero(T), &, pmf) ||
        _bad_distribution("Probabilities must be non-negative.")
    sum(pmf) ≈ one(T) ||
        _bad_distribution("Probabilities must sum to 1 and not $(sum(pmf)).")
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
  i == j && return 
  vals[i], vals[j] = vals[j], vals[i]
  p[i], p[j] = p[j], p[i]
end

#"""
#    partition!(vals::AbstractVector{<:Real}, p::AbstractVector{<:Real}, f::Int, b::Int)
#
#Function to partition the values in `vals` and `p` between `f` and `b` (inclusive).
#The pivot is chosen from the middle. 
#
## Returns
#
#A tuple (lt::Int, gt::Int) where `lt` is the index of the start of the eq partition and `gt` is the index of end of the eq partition, the range between them includes values equal to the pivot.
#
## Examples
#
#  x = [1,2,2,3]
#  partition!(x, 2, 1, 4) # (2, 4)
#"""
function partition!(vals::AbstractVector{<:Real}, p::AbstractVector{<:Real}, pivot_val :: Real,
                    f::Int, b::Int)

    #pivot_ind = f + Int(ceil((b - f) / 2))
    #pivot_val = vals[pivot_ind]
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


## Quick Quantile (loopy version)
## NOTE: computes q+ = upper bound on the quantile function
function qql!(vals::AbstractVector{<:Real}, p::AbstractVector{<:Real}, α::Real)
    if iszero(α) # minimum
        return essinf(vals, p; check_inputs=true)
    elseif isone(α) # maximum (it is unbounded)
        return (value=typemax(eltype(p)), index=length(vals))
    end
    
    f = 1
    b = length(vals)
    g = 1
    @inbounds while b - f ≥ 1
        pivot_ind = f + Int(ceil((b - f) / 2))
        v = vals[pivot_ind]
        
        l, g = partition!(vals, p, v, f, b)
        t = sum(view(p, f:(l-1)))  # P[x < v]
        e = sum(view(p, l:g))      # P[x = v]
        if α < t  # α < P[x̃ < v] ==> α < 1 - P[x̃ ≥ v] ==> P[x̃ ≥ v] < 1 - α
            # v or greater value cannot be a quantile
            b = l-1
        # NOTE: this case is subsumed by the next one
        # Would be used for quantile, not for quatile+
        #elseif t + e < α  # P[x̃ ≤ v] < α  
        #    # v or smaller cannot be a quantile
        #    f = g + 1
        #    α -= t + e
        elseif t + e ≤ α  # P[x̃ ≤ v] ≤ α <=> P[x̃ > v] = (1 - P[x̃ ≤ v]) ≥ 1-α 
            # v or smaller cannot be a quantile+ because v + ϵ also satisfies
            # NOTE: quantile+ only case, remove for quantile
            f = g + 1
            α -= t + e
        else
            # α ≥ P[x̃ < v] and P[x̃ ≤ v] ≥ α and P[x̃ ≤ v] > α
            # => α ≥ 1 - P[x̃ ≥ v] and P[x̃ ≤ v] ≥ α and (1 - P[x̃ > v]) > α
            # => P[x̃ ≥ v] ≥ 1 - α and P[x̃ ≤ v] ≥ α and P[x̃ > v] < 1-α
            # hence v is a quantile+, since it is a quantile and cannot be greater
            f = b = g
        end 
    end
    return (value=vals[f], index=f)
end
