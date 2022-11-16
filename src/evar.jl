import Optim: optimize, Brent

"""
    evar(X, p::Distribution, α; β_max = 100.)

Compute the EVaR risk measure of the random variable `X` represented as a vector over
outcomes and distributed according to the measure `p` with risk level `α`. The risk
level `α` should be in [0,1].

When `α = 0`, the function computes the expected value, and when `α = 1`, then the 
function computes the essential infimum (a minimum value with positive probability).

The function solves 
``\\sup_{β > 0} \\operatorname{erm}_β (X) - β^{-1} \\log (1/(1-α))``.
The optimization over the value of `β` is upper bounded by `β_max`. Large
values of `β_max` may cause the computation to overflow.

The function implicitly assumes that all elements have non-zero (if diminishingly small)
probability. 

Returns the evar and the value β that achieves the supremum above.

See:
Ahmadi-Javid, A. “Entropic Value-at-Risk: A New Coherent Risk Measure.” Journal of
Optimization Theory and Applications 155(3), 2012.
"""
function evar(X::AbstractVector{<:Real}, p::Distribution{T}, α::Real;
              β_max = 100.) where {T <: Real}
    length(X) == length(p) || _bad_distribution("Lengths of X and p must match.")
    length(X) > 0 || _bad_risk("X must have some elements")

    # TODO: This method minimizes the quasi-convex representation. Perhaps we should
    # be working with the convex one!
    
    if iszero(α)
        (evar = mean(X, p.p) |> float, β=0., p = p)
    elseif isone(α)
        v,np = essinf(X,p)
        (evar = v, β = β_max, p = np)
    elseif zero(α) ≤ α ≤ one(α)
        Xmin = minimum(X)
        # the function is minimized
        f(β) = -erm(X, p, β, Xmin) - (log(1-α) / β)
        sol = optimize(f, 0., β_max, Brent())
        sol.converged || error("Failed to find optimal β (unknown reason).")
        isfinite(sol.minimum) ||
            error("Overflow, computed an invalid solution. Reduce β_max.")
        β = float(sol.minimizer)
        # compute the robust representation solution from Donsker Varadhan
        (evar = -float(sol.minimum),
         β = β,
         p = Distribution{T}(softmin(X, p, β, Xmin))) 
    else
        _bad_risk("Parameter β must be in [0,1]")
    end
end

evar(X::AbstractVector{<:Real}, p::AbstractVector{T}, α::Real; kwargs...) where {T<:Real} =
    evar(X, Distribution{T}(p), α; kwargs...)

evar(X::AbstractVector{<:Real}, α::Real; kwargs...) =
    evar(X, uniform(length(X)), α; kwargs...) 
