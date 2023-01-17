import Optim: optimize, Brent, BFGS

"""
    evar(X, p::Distribution, α; [βmin, βmax, reciprocal = false])

Compute the EVaR risk measure of the random variable `X` represented as a vector over
outcomes and distributed according to the measure `p` with risk level `α`. The risk
level `α` should be in [0,1].

When `α = 0`, the function computes the expected value, and when `α = 1`, then the 
function computes the essential infimum (a minimum value with positive probability).

The function solves 
``\\max_{β ∈ [βmin, βmax]} \\operatorname{erm}_β (X) - β^{-1} \\log (1/(1-α))``.
Large values of `βmax` may cause the computation to overflow.

If `reciprocal = false`, then the quasi-concave problem above is solved directly.
If `reciprocal = true`, then the optimization is reformulated in terms of `λ = 1/β`
to get a concave function that can be solved (probably) more efficiently

The function implicitly assumes that all elements of the probability space have non-zero
probability. 

Returns the evar and the value β that attains the maximum above.

See:
Ahmadi-Javid, A. “Entropic Value-at-Risk: A New Coherent Risk Measure.” Journal of
Optimization Theory and Applications 155(3), 2012.
"""
function evar(X::AbstractVector{<:Real}, p::Distribution{T}, α::Real;
              βmin = 1e-5, βmax = 100., reciprocal = false) where {T <: Real}
    length(X) == length(p) || _bad_distribution("Lengths of X and p must match.")
    length(X) > 0 || _bad_risk("X must have some elements")

    if iszero(α)
        (evar = mean(X, p.p) |> float, β=0., p = p)
    elseif isone(α)
        v,np = essinf(X,p)
        (evar = v, β = βmax, p = np)
    elseif zero(α) ≤ α ≤ one(α)
        Xmin = minimum(X)
        if !(reciprocal)
            # the function is minimized
            logconst = log(1-α)
            f(β) = -erm(X, p, β, Xmin) - (logconst / β)
            sol = optimize(f, βmin, βmax, Brent())
            sol.converged || error("Failed to find optimal β (unknown reason).")
            isfinite(sol.minimum) ||
                error("Overflow, computed an invalid solution. Reduce βmax.")
            β = float(sol.minimizer)
            # compute the robust representation solution from Donsker Varadhan
            return (evar = -float(sol.minimum), β = β,
                    p = Distribution{T}(softmin(X, p, β, Xmin))) 
        else
            g(λ) = - (erm(X, p, 1/λ, Xmin) + λ*log(1-α))
            # this is the derivative, but not sure if it is useful:
            # df(λ) = - (λ * erm(X, p, 1/λ; Xmin) + softmintrue(X, p, 1/λ; Xmin))
            sol = optimize(g, 1.0 / βmax, 1.0 / βmin, Brent())
            sol.converged || error("Failed to find optimal λ (unknown reason).")
            isfinite(sol.minimum) ||
                error("Overflow, computed an invalid solution. Reduce βmax.")
            β = 1. / float(sol.minimizer)
            # compute the robust representation solution from Donsker Varadhan
            return (evar = -float(sol.minimum), β = β,
                    p = Distribution{T}(softmin(X, p, β, Xmin))) 
        end
    else
        _bad_risk("Parameter β must be in [0,1]")
    end
end

evar(X::AbstractVector{<:Real}, p::AbstractVector{T}, α::Real; kwargs...) where {T<:Real} =
    evar(X, Distribution{T}(p), α; kwargs...)

evar(X::AbstractVector{<:Real}, α::Real; kwargs...) =
    evar(X, uniform(length(X)), α; kwargs...) 
