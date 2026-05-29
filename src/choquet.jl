"""

    choquet_risk(x̃, c, α)

Compute the risk measure for a given choquet capacity function `c` and
random variable `x̃`.


    choquet_risk(x, pmf, c, α)

Compute the risk measure for a given choquet capacity function `c` and
random variable `x` with probabilities `pmf`.

The choquet risk measure solves
```math
\\operatorname{choquet}(x, p, c, α) =
\\min \\{ x^T q \\mid
   q \\in \\Delta_n, q(\\mathcal{U}) \\le c(\\mathcal{U}, p, \\alpha), \\forall \\mathcal{U} \\}
```

The choquet capacity function `c` that returns a non-negative value
and is parametrized by the random variable `S`, a probability mass function `pmf`, and
level `α ∈ [0,1]`.

The runtime of this function can be quadratic depending on the evaluation of the
capacity function.
"""
function choquet_risk(x::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, c::Function,
                      α::Real; check_inputs = true)
    _check_α(α)
    check_inputs && _check_pmf(x, pmf)

    indices = sortperm(x)
    T = float(eltype(pmf))
    ξ = zeros(T, length(x))

    c_prev = zero(T)
    for i in 1:length(x)
        c_curr = c(view(indices, 1:i), pmf, α)
        ξ[indices[i]] = c_curr - c_prev
        c_prev = c_curr
    end
    (value = ξ'*x, pmf = ξ)
end

function choquet_risk(x̃, c, α; kwargs...)
    supp, pmf = rv2pmf(x̃)
    choquet_risk(supp, pmf, c, α; kwargs...)
end


"""
    cvar_distortion(t, α)

Compute the choquet capacity function equivalent to CVaR at level `α`, to be used with `choquet_distortion_risk`.
"""
function cvar_distortion(t::Real, α::Real)
    t ≥ zero(t) || error("t must be non-negative")
    
    if zero(α) < α ≤ one(α)
        min(t / α, one(t))
    elseif α == zero(α) && t > zero(t)
        one(t)
    elseif α == zero(α) && t == zero(t)
        zero(t)
    else
        error("α must be in [0,1]")
    end
end

"""
    cvar_capacity(S, pmf, α)

Compute the choquet capacity function equivalent to CVaR at level `α`, to be used
with `choquet_risk`. Here `S` is the list of indices into the `pmf`.
"""
cvar_capacity(S::AbstractVector{<:Integer}, pmf::AbstractVector{<:Real}, α::Real) = 
    cvar_distortion(sum(pmf[i] for i in S), α)

"""
    choquet_distortion_risk(x̃, g, α)

Compute the choquet risk measure for a law-invariant capacity `c(A) = g(P[A])`,
where `g : [0,1] × R → [0,1]` is a distortion function with `g(0, α) = 0` and `g(1, α) = 1`.

    choquet_distortion_risk(x, pmf, g, α)

Compute the choquet risk measure for a law-invariant capacity `c(A) = g(P[A])`,
where `g : [0,1] × R → [0,1]` is a distortion function with `g(0, α) = 0` and `g(1, α) = 1`.

The choquet distortion risk measure solves
```math
\\operatorname{choquet}(x, p, c, α) =
\\min \\{ x^T q \\mid
   q \\in \\Delta_n, q(\\mathcal{U}) \\le g(p(\\mathcal{U}), \\alpha), \\forall \\mathcal{U} \\}
```

More efficient than `choquet_risk` for law-invariant measures: `g` is evaluated
on scalars rather than index sets, and cumulative probabilities are computed once.
"""
function choquet_distortion_risk(x::AbstractVector{<:Real}, pmf::AbstractVector{<:Real},
                                 g::Function, α::Real; check_inputs = true)
    _check_α(α)
    check_inputs && _check_pmf(x, pmf)

    indices = sortperm(x)
    T = float(eltype(pmf))
    ξ = zeros(T, length(x))
    
    g_prev = zero(T)
    F = zero(T)
    for i in 1:length(x)
        F += pmf[indices[i]]
        g_curr = T(g(F, α))
        ξ[indices[i]] = g_curr - g_prev
        g_prev = g_curr
    end
    (value = ξ'*x, pmf = ξ)
end


function choquet_distortion_risk(x̃, c, α; kwargs...)
    supp, pmf = rv2pmf(x̃)
    choquet_distortion_risk(supp, pmf, c, α; kwargs...)
end



"""
    closure_c(ρ)

Given a risk function `ρ(x, pmf) -> Real`, return a closure that computes the submodular function
`c(S, P) = -ρ(-1_S, P)` where `1_S` is the indicator vector of an index set `S`. When `ρ` is
coherent and comonotonic, then `choquet_risk` recovers the same risk.

```@example
using RiskMeasures

x = [1,3,-5,3]
p = [0.3,0.2,0.3,0.2]
α = 0.4

ρ(x, p, α) = CVaR(x, p, α).value
choquet_risk(x, p, RiskMeasures.closure_c(ρ), α)
CVaR(x, p, α)
```
"""
function closure_c(ρ::Function)
    function (S::AbstractVector{<:Integer}, pmf::AbstractVector{<:Real}, α::Real)
        T = float(eltype(pmf))
        one_tilde = zeros(T, length(pmf))
        one_tilde[S] .= one(T)
        -ρ(-one_tilde, pmf, α)
    end
end
