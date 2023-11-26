_bad_risk(msg::AbstractString) =
    error(msg)
_bad_distribution(msg::AbstractString) =
    error(msg)

# checks a pmf for correctness
function _check_pmf(p::AbstractVector{<:Real})
    !isempty(p) || _bad_distribution("pmf must be non-empty.")
    mapreduce(isfinite, &, p) || _bad_distribution("Probabilities must be finite.")
    mapreduce(x -> x≥zero(T), &, p) ||
        _bad_distribution("Probabilities must be non-negative.")
    sum(p) ≈ one(T) || _bad_distribution("Probabilities must sum to 1.")
end

function _check_pmf(x̃::AbstractVector{<:Real}, p::AbstractVector{<:Real})
    length(x̃) == length(p) || _bad_distribution("Lengths of x̃ and p must match.")
    _check_pmf(p)
end

function _check_α(α::Real)
    zero(α) ≤ α ≤ one(α) || _bad_risk("Risk level α must be in [0,1].")
end
