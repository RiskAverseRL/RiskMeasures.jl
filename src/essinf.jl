"""
    essinf(x̃; distribution = false)

Compute the essential infimum of the random variable `x̃`, which is the minimum
value with positive probability
"""
function essinf end

"""
    essinf_e(values, pmf; ...)

Compute the value for a random variable with `values` and the probability mass
function `pmf`.

See `essinf_e(x̃)` for more details.
"""
function essinf_e(values::AbstractVector{Tval}, pmf::AbstractVector{<:Real};
                  check_inputs = true) :: @NamedTuple{value::Tval, index::Int} where
    {Tval <: Real}
    
    check_inputs && _check_pmf(values, pmf)
    
    minval = typemax(Tval)
    minindex = -1

    @inbounds for i ∈ eachindex(values, pmf)
        (!iszero(pmf[i]) && values[i] < minval) &&
            (minval = float(values[i]); minindex = i)
    end

    (value = minval, index = minindex)
end


essinf(x̃; kwargs...) =
    essinf_d(rv2pmf(x̃)...; kwargs...).value
