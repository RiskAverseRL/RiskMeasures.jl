using Distributions

"""
    UBSR(x, p, u, λ; z_min=-1e6, z_max=1e6, tol=1e-7)

Compute the Utility Based Shortfall RiskMeasure (UBSR) for a 
discrete random variable with reward vector `x`, probability vector `p`,
for a monotone function `u`, and risk-threshold `λ`.

```math
\\operatorname{UBSR(x, p, u, λ) =
\\sup \\{z ∈ ℝ \\mid \\mathbb{E}[u(x - z)] ≥ λ \\} = \\sup \\{z ∈ ℝ \\mid g(z) ≥ λ \\}
```

If `u` is NOT monotone (non-descreasing), the function return is undefined.

# Keyword Arguments:
- `z_min`, `z_max`: Bounds for the bisection search.
- `tol`: Tolerance for convergence of the bisection method.

# Returns
- A named tuple with the computed UBSR `value`.
"""
function UBSR end

function UBSR(x::AbstractVector{<:Real}, p::AbstractVector{<:Real}, u::Function, λ::Real;
              z_min=-1e6, z_max=1e6, tol=1e-6, check_inputs=true)

    check_inputs && _check_pmf(p)
    
    ExU(z) = p' * u.(x .- z) # is non-increasing
    
    f_min = ExU(z_min) 
    f_max = ExU(z_max) 
    
    f_min < λ && error("z_min is too high: E[u(x - z_min)] < λ.")
    f_max > λ && error("z_max is too low: E[u(x - z_max)] > λ.")

    f_min > f_max && error("Function u is not monotone.")
    
    # Bisection Method
    low = z_min
    high = z_max
    while (high - low) > tol
        z_mid = (low + high) / 2
        u_mid = ExU(z_mid)
        if u_mid ≥ λ
            # u(x-z) is monotone in z => ExU ≥ λ
            low = z_mid
        elseif u_mid 
        else
            high = z_mid
        end
    end

    (value=(low + high) / 2,)
end

function UBSR(x̃, u, λ; kwargs...)
    supp, pmf = rv2pmf(x̃)
    v1 = UBSR(supp, pmf, u, λ; kwargs...)
    (value=v1.value,)
end


