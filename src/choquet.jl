"""
    choquet_risk(x, pmf, c, α)

Compute the risk measure for a given choquet capacity function `c` and random variable `x` with probabilities `pmf`.
The choquet capacity function `c` is parametrized by a level `α`.
"""
function choquet_risk(x::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, c::Function, α::Float64)
  indices = sortperm(x)
  ξ = zeros(Float64, length(x))
  for i in 1:length(x)
      ci = c(indices[1:i], pmf, α)
      c1m1 = c(indices[1:i-1], pmf, α)
      @assert 0.0 <= ci <= 1.0
      @assert 0.0 <= c1m1 <= 1.0
      ξ[indices[i]] = ci - c1m1
  end
  ξ' * x
end


"""
    closure_c(ρ)

Given a risk measure function *ρ*, return a *closure* that computes the submodular function
`c(S) = -ρ(-1_S)` where `1_S` is the indicator vector of an index set `S`.
"""
function closure_c(ρ::Function)
  function (S::AbstractVector{<:Integer}, pmf::AbstractVector{<:Real}, alpha::Float64) # this is the submodular function
    #length(S) == 0 && return 0 # By definition c(∅) = 0
    one_tilde = zeros(Float64, length(pmf))
    one_tilde[S] .= 1 # wow, julia thanks for this neat notation
    -ρ(-one_tilde, pmf, alpha)
  end
end
