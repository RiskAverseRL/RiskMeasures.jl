"""
    choquet_risk(x, pmf, c, α)

Compute the risk measure for a given choquet capacity function `c` and
random variable `x` with probabilities `pmf`.

The choquet capacity function `c` that returns a non-negative value
and is parametrized by a level `α ∈ [0,1]`.

The runtime of this function can be quadratic depending on the evaluation of the
capacity function.
"""
function choquet_risk(x::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, c::Function,
                      α::Real; check_inputs = true)

    check_inputs && _check_α(α)
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
    ξ' * x
end


"""
    closure_c(ρ)

Given a risk measure function `ρ`, return a closure that computes the submodular function
`c(S) = -ρ(-1_S)` where `1_S` is the indicator vector of an index set `S`.
"""
function closure_c(ρ::Function)
    function (S::AbstractVector{<:Integer}, pmf::AbstractVector{<:Real}, alpha::Real)
        T = float(eltype(pmf))
        one_tilde = zeros(T, length(pmf))
        one_tilde[S] .= one(T)
        -ρ(-one_tilde, pmf, alpha)
    end
end


"""
    cvar_capacity(S, pmf, α)

choquet capacity function equivalent to CVaR at level `α`.
Returns `min(sum(pmf[S]) / α, 1)`, which is the distorted probability `g(P(S))`
with distortion `g(t) = min(t/α, 1)`.
"""
function cvar_capacity(S::AbstractVector{<:Integer}, pmf::AbstractVector{<:Real}, α::Real)
    T = float(eltype(pmf))
    isempty(S) && return zero(T)
    min(sum(pmf[i] for i in S) / α, one(T))
end


"""
    choquet_distortion_risk(x, pmf, g, α)

Compute the choquet risk measure for a law-invariant capacity `c(A) = g(P[A])`,
where `g : [0,1] → [0,1]` is a distortion function with `g(0) = 0` and `g(1) = 1`.

More efficient than `choquet_risk` for law-invariant measures: `g` is evaluated
on scalars rather than index sets, and cumulative probabilities are computed once.
"""
function choquet_distortion_risk(x::AbstractVector{<:Real}, pmf::AbstractVector{<:Real},
                                 g::Function, α::Real; check_inputs = true)
    check_inputs && _check_α(α)
    check_inputs && _check_pmf(x, pmf)

    indices = sortperm(x)
    T = float(eltype(pmf))

    g_prev = zero(T)
    value  = zero(T)
    F      = zero(T)
    for i in 1:length(x)
        F     += pmf[indices[i]]
        g_curr = T(g(F, α))
        value += (g_curr - g_prev) * x[indices[i]]
        g_prev = g_curr
    end
    value
end


"""
    cvar_distortion(t, α)

Distortion function equivalent to CVaR at level `α`: `min(t / α, 1)`.
"""
cvar_distortion(t::Real, α::Real) = min(t / α, one(t))
