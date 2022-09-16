import Base

"""
Represents a discrete distribution over a finite set of outcomes 
with non-negative values and sums to 1. The parameter `p` is copied 
if it is not a vector and reused otherwise. 

The default constructor ingests (does not copy) the parameter.
"""
struct Distribution{T<:Real} 
    p :: Vector{T}

    function Distribution{T}(p::Vector{T}) where {T<:Real} 
        !isempty(p) || _bad_distribution("distribution must be non-empty.")
        all(p .≥ zero(T)) || _bad_distribution("Probabilities must be non-negative.")
        sum(p) ≈ one(T) || _bad_distribution("Probabilities must sum to 1.")
        new(p)
    end
end


function Distribution{T}(p::AbstractVector{T}) where {T<:Real} 
    Distribution{T}(Vector{T}(p))
end

"""
    uniform(n)

Construct a uniform distribution of length `n`.
"""
function uniform(n::Int, T::Type) 
    length(n) > 0 || _bad_distribution("Must use positive n")
    Distribution{T}(ones(T, n) / n)
end

uniform(n::Int) = uniform(n, Float64)
    
function Base.length(p::Distribution{T}) where {T}
    return length(p.p)
end
