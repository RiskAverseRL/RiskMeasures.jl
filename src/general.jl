using Distributions: DiscreteNonParametric


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
