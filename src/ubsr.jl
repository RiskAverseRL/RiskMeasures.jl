using Distributions

"""
    UBSR(x, p, u, λ; z_min=-1e6, z_max=1e6, tol=1e-7)

Computes the Utility Based Shortfall RiskMeasure (UBSR) for a 
discrete random variable with reward vector `x` and probability vector `p`.

UBSR(x, p, u, λ) = sup{z ∈ ℝ : E[u(x - z)] ≥ λ}

We assume u is non-constant and non-decreasing, 
so the supremum is well-defined and can be found via bisection search.

# Arguments
- `x`: Reward vector.
- `p`: Probability vector.
- `u`: Utility function. Non-constant and non-decreasing.
- `λ`: Risk threshold.
# Keyword Arguments:
- `z_min`, `z_max`: Bounds for the bisection search.
- `tol`: Tolerance for convergence of the bisection method.
# Returns
- A named tuple with the computed UBSR `value`.
"""
function UBSR end

function UBSR(x::AbstractVector{<:Real}, p::AbstractVector{<:Real}, u::Function, λ::Real; z_min=-1e6, z_max=1e6, tol=1e-6,
              check_inputs=true)
  check_inputs && _check_pmf(p)

  ExU(z) = sum(p .* u.(x .- z))

  f_min = ExU(z_min) - λ
  f_max = ExU(z_max) - λ
  if f_min < 0
    @warn("z_min is too high; E[u(x - z_min)] < λ. Increase the search range.")
  end
  if f_max > 0
    @warn "z_max might be too low; E[u(x - z_max)] > λ. The supremum might be larger."
  end

  # Bisection Method
  low = z_min
  high = z_max
  while (high - low) > tol
    z_mid = (low + high) / 2
    if ExU(z_mid) >= λ
      # Since u(x-z) is non-increasing in z, if ExU >= λ, 
      # we can try a larger z to find the supremum.
      low = z_mid
    else
      high = z_mid
    end
  end

  return (value=(low + high) / 2,)
end

function UBSR(x̃, u, λ; kwargs...)
  supp, pmf = rv2pmf(x̃)
  v1 = UBSR(supp, pmf, u, λ; kwargs...)
  (value=v1.value,)
end


