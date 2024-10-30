import Optim: optimize, Brent, BFGS
using Distributions
# TODO: check if the following is true
"""
    expectile(x̃, α)

Compute the expectile risk measure of the random variable `x̃` with risk level `α` in (0,1).
When `α = 1/2`, the function computes the expected value. Notice the rannge for \alpha is non-inclusive.

The function solves
```math
\\min_{x ∈ Real} α E((X - x)^2_+) - (1-α) E((X - x)^2_-)
```
"""
function expectile end


"""
    expectile_e(values, pmf, α; ...)

Compute expectile for a discrete random variable with `values` and the probability mass
function `pmf`. See `expectile(x̃, α)` for more details.
"""
function expectile_e(values::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real; check_inputs=true)
    check_inputs && zero(α) < α < one(α) || _bad_risk("Risk level α must be in (0,1).")
    check_inputs && _check_pmf(values, pmf)

    if abs(α - 0.5) <= 1e-10
        return (value=values' * pmf, pmf=Vector(pmf))
    end

    xmin, xmax = extrema(values)
    # the function is minimized
    f(x) = α * (max.(values .- x, 0) .^ 2)' * pmf + (1 - α) * (max.(x .- values, 0) .^ 2)' * pmf
    sol = optimize(f, xmin, xmax, Brent())
    sol.converged || error("Failed to find optimal x (unknown reason).")
    isfinite(sol.minimum) ||
        error("Overflow, computed an invalid solution. Check α.")
    x = float(sol.minimizer)
    return (value=x, pmf=pmf)
end


expectile(x̃, α::Real; kwargs...) = expectile_e(rv2pmf(x̃)..., α; kwargs...).value

function expectile_e(x̃, α::Real; kwargs...)
    supp, pmf = rv2pmf(x̃)
    v1 = expectile_e(supp, pmf, α; kwargs...)
    ỹ = DiscreteNonParametric(supp, v1.pmf)
    (value=v1.value, solution=ỹ)
end
