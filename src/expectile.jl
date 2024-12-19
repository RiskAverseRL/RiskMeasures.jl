import Optim: optimize, Brent, BFGS
using Distributions

"""
    expectile(x̃, α)

Compute the expectile risk measure of the random variable `x̃` with risk level
`α ∈ (0,1)`. When `α = 1/2`, the function computes the expected value.
Expectile is only coherent when `α ∈ (0,1/2]`.

Notice the range for `α` does not include `0` or `1`.

The function solves
```math
\\argmin_{x ∈ Real} α \\mathbb{E}((x̃ - x)^2_+) - (1-α) \\mathbb{E}((x̃ - x)^2_-)
```
"""
function expectile end


"""
    expectile(values, pmf, α; ...)

Compute expectile for a discrete random variable with `values` and the probability mass
function `pmf`. See `expectile(x̃, α)` for more details.

**Note**: The expectile is only coherent when `α ≤ 0.5`. 
"""
function expectile(values::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real; check_inputs=true)
    check_inputs && (zero(α) < α < one(α) || _bad_risk("Risk level α must be in (0,1)."))
    check_inputs && _check_pmf(values, pmf)

    if abs(α - 0.5) <= 1e-10
        return (value=values' * pmf, pmf=pmf)
    end

    xmin, xmax = extrema(values)
    # the function is minimized
    f(x) = α * (max.(values .- x, 0) .^ 2)' * pmf + (1 - α) * (max.(-1 * (values .- x), 0) .^ 2)' * pmf
    sol = optimize(f, xmin, xmax, Brent())
    sol.converged || error("Failed to find optimal x (unknown reason).")
    isfinite(sol.minimum) || error("Overflow, computed an invalid solution. Check α.")
    x = float(sol.minimizer)
    return (value=x, pmf=pmf)
end

function expectile(x̃, α::Real; kwargs...)
    supp, pmf = rv2pmf(x̃)
    v1 = expectile(supp, pmf, α; kwargs...)
    ỹ = DiscreteNonParametric(supp, v1.pmf)
    (value=v1.value, pmf=ỹ)
end
