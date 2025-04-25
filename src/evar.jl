import Optim: optimize, Brent, BFGS
using Distributions

"""
    EVaR(x̃, α; βmin = 1e-5, βmax = 100, reciprocal = false, distribution = false)

Compute the EVaR risk measure of the random variable `x̃` with risk level `α` in [0,1].

    EVaR(values, pmf, α; ...)

Compute EVaR for a discrete random variable with `values` and the probability mass
function `pmf`.

When `α = 1`, the function computes the expected value, and when `α = 0`, then the 
function computes the essential infimum (a minimum value with positive probability).

The function solves
```math
\\max_{β ∈ [βmin, βmax]} \\operatorname{ERM}_β (x̃) - β^{-1} \\log (1/(α)).
```

Large values of `βmax` may cause the computation to overflow.

If `reciprocal = false`, then the quasi-concave problem above is solved directly.
If `reciprocal = true`, then the optimization is reformulated in terms of `λ = 1/β`
to get a concave function that can be solved (probably) more efficiently

The function implicitly assumes that all elements of the probability space have non-zero
probability. 

Returns the EVaR and the value β that attains the maximum above.

See:
Ahmadi-Javid, A. “Entropic Value-at-Risk: A New Coherent Risk Measure.” Journal of
Optimization Theory and Applications 155(3), 2012.
"""
function EVaR end


function EVaR(values::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real;
    βmin=1e-5, βmax=100.0, reciprocal=false, check_inputs=true)

    check_inputs && _check_α(α)
    check_inputs && _check_pmf(values, pmf)

    T = eltype(pmf) |> float

    tol = 1e2 # tolerance for suboptimality

    if isone(α)
        return (value=values' * pmf, β=0.0, pmf=Vector{T}(pmf))
    elseif iszero(α)
        minval = essinf(values, pmf; check_inputs=false)
        minpmf = zeros(T, length(pmf))
        minpmf[minval.index] = one(T)
        return (value=minval.value, β=Inf, pmf=minpmf)
    end

    xmin = minimum(values)
    if !(reciprocal)
        # the function is minimized
        logconst = log(α)
        f(β) = -ERM(values, pmf, β; x̃min=xmin, check_inputs=false) -
               (logconst / β)
        sol = optimize(f, βmin, βmax, Brent())
        sol.abs_tol < tol || error("Failed to find optimal β (unknown reason).")
        isfinite(sol.minimum) ||
            error("Overflow, computed an invalid solution. Reduce βmax.")
        β = float(sol.minimizer)
        # compute the robust representation solution from Donsker Varadhan
        return (value=-float(sol.minimum), β=β,
            pmf=softmin(values, pmf, β; x̃min=xmin, check_inputs=false))
    else
        g(λ) = -(ERM(values, pmf, 1 / λ; x̃min=xmin, check_inputs=false) +
                 λ * log(α))
        # this is the derivative, but it does not appear to be useful
        # df(λ) = - (λ * ERM(values, pmf, 1/λ; xmin) + softmin(, p, 1/λ; xmin))
        sol = optimize(g, inv(βmax), inv(βmin), Brent())
        sol.abs_tol < tol || error("Failed to find optimal λ (unknown reason).")
        isfinite(sol.minimum) ||
            error("Overflow, computed an invalid solution. Reduce βmax.")
        β = 1.0 / float(sol.minimizer)
        # compute the robust representation solution from Donsker Varadhan
        return (value=-float(sol.minimum), β=β,
            pmf=softmin(values, pmf, β; x̃min=xmin, check_inputs=false))
    end
end



function EVaR(x̃, α::Real; kwargs...)
    supp, pmf = rv2pmf(x̃)
    v1 = EVaR(supp, pmf, α; kwargs...)
    ỹ = DiscreteNonParametric(supp, v1.pmf)
    (value=v1.value, pmf=ỹ, β=v1.β)
end
