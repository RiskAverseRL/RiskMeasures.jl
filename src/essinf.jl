"""
    essinf(x̃)

Compute the essential infimum of the random variable `x̃`, which is the minimum
value with positive probability
"""
function essinf end

"""
    essinf(values, pmf; ...)

Compute the value for a random variable with `values` and the probability mass
function `pmf`.
"""
function essinf(values::AbstractVector{Tval}, pmf::AbstractVector{<:Real};
                check_inputs = true) ::
                    @NamedTuple{value::Tval, index::Int} where {Tval <: Real}
    
    check_inputs && _check_pmf(values, pmf)

    Tout = float(Tval)
    
    minval :: Tout = typemax(Tout)
    minindex :: Int = -1

    @inbounds for i ∈ eachindex(values, pmf)
        if !iszero(pmf[i]) && values[i] < minval 
            minval = values[i]
            minindex = i
        end
    end

    (value = minval :: Tout, index = minindex)
end


essinf(x̃; kwargs...) =
    essinf(rv2pmf(x̃)...; kwargs...).value
