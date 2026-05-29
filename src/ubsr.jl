using Distributions

"""
    UBSR(x̃, u, λ; ...)

Compute the Utility Based Shortfall RiskMeasure (UBSR) for a discrete random variable `x̃`,
for a monotone function `u`, and risk-threshold `λ`.

    UBSR(x, p, u, λ; ...)

Compute the Utility Based Shortfall RiskMeasure (UBSR) for a discrete random variable with
values `x`, and pmf `p`, for a monotone function `u`, and risk-threshold `λ`.

```math
\\operatorname{UBSR}(x, p, u, λ) =
\\sup \\{z ∈ ℝ \\mid \\mathbb{E}[u(\\tilde{x} - z)] ≥ -λ \\} =
\\sup \\{z ∈ ℝ \\mid -g(z) \\le λ \\}
```
where
```math
g(z) := \\mathbb{E}[u(\\tilde{x} - z)]
```
The UBSR can be seen as the right-continuous inverse of g

If `u` is NOT monotone (non-decreasing), the `UBSR` return is undefined and may
terminate with an error.

# Keyword Arguments:
- `zmin=-1e6`, `zmax=1e6`: Lower and upper bounds for the bisection search.
- `tol=1e-7`: Tolerance for convergence of the bisection method.

# Returns
- A named tuple with the computed UBSR `value`.
"""
function UBSR end

function UBSR(x::AbstractVector{<:Real}, p::AbstractVector{<:Real}, u::Function, λ::Real;
              zmin::Float64=-1e6, zmax::Float64=1e6, tol::Float64=1e-6, check_inputs=true)
    check_inputs && _check_pmf(x, p)
    zmin < zmax || error("zmin < zmax is violated")
    
    g(z) = p' * u.(x .- z) # is non-increasing


    g(zmin) < -λ && (@warn "zmin is too high: E[u(x - z_min)] < λ." ; return (value=-Inf,))
    g(zmax) ≥ -λ && (@warn "zmax is too low: E[u(x - z_max)] > λ."; return (value=Inf,))

    g(zmin) < g(zmax) && error("Function g is not monotone: $(g(zmin)) < $(g(zmax)).")
    
    # Bisection Method
    while (zmax - zmin) > tol
        zmid = (zmin + zmax) / 2
        umid = g(zmid)

        if umid ≥ -λ
            zmin = zmid
        else
            zmax = zmid
        end
    end
    (value=zmax,)  # return the maximum because we want an upper bound
end

function UBSR(x̃, u, λ; kwargs...)
    supp, pmf = rv2pmf(x̃)
    UBSR(supp, pmf, u, λ; kwargs...)
end


