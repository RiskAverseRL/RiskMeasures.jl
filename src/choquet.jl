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
and is parametrized by the list of indices `S`, a probability mass function `pmf`, and
level `α ∈ [0,1]`.

The runtime of this function can be quadratic depending on the evaluation of the
capacity function.

# Returns

A named tuple with risk `value` and the `pmf` that achieves it.

# Examples

```jldoctest
julia> choquet_risk([1, 2, 3, 4, 5], [0.2, 0.2, 0.2, 0.2, 0.2], cvar_capacity, 0.4).value
1.5
```
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
    v1 = choquet_risk(supp, pmf, c, α; kwargs...)
    ỹ = DiscreteNonParametric(supp, v1.pmf)
    (value = v1.value, pmf = ỹ)
end


"""
    cvar_distortion(t, α)

Compute the choquet capacity function equivalent to CVaR at level `α`, to be used with `choquet_distortion_risk`.

# Examples

```jldoctest
julia> cvar_distortion(0.2, 0.4)
0.5
```
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

# Examples

```jldoctest
julia> cvar_capacity([1], [0.2, 0.3, 0.5], 0.4)
0.5
```
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

This function is more efficient than `choquet_risk` for law-invariant measures: `g` is evaluated
on scalars rather than index sets, and cumulative probabilities are computed once.


# Returns

A named tuple with risk `value` and the `pmf` that achieves it.

# Examples

```jldoctest
julia> choquet_distortion_risk([1, 2, 3, 4, 5], [0.2, 0.2, 0.2, 0.2, 0.2], cvar_distortion, 0.4).value
1.5
```
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


function choquet_distortion_risk(x̃, g, α; kwargs...)
    supp, pmf = rv2pmf(x̃)
    v1 = choquet_distortion_risk(supp, pmf, g, α; kwargs...)
    ỹ = DiscreteNonParametric(supp, v1.pmf)
    (value = v1.value, pmf = ỹ)
end



"""
    closure_c(ρ)

Given a risk function `ρ(values, pmf, α) -> Real`, return a closure that computes the submodular
function `c(S, pmf, α) = -ρ(-1_S, pmf, α)` where `1_S` is the indicator vector of an index set `S`.
When `ρ` is coherent and comonotonic, then `choquet_risk` recovers the same risk.

# Returns

A set function that can be used with `choquet_risk`

# Examples

```jldoctest
julia> ρ(values, pmf, α) = CVaR(values, pmf, α).value;

julia> c = closure_c(ρ);

julia> choquet_risk([5, 2, 3, 4, 1], [0.2, 0.2, 0.2, 0.2, 0.2], c, 0.4).value
1.5
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


"""
     choquet_ews(x, p, (m, c); [check_inputs = true])

Compute the risk measure for a random variable `x and an EWS function
with parameters `p`, `m`, `c`. This algorithm can evaluate certain
comotonic coherent risk measures in linear time.

If linear time computation is not a concern, it is better to use a standard
implementations such as `choquet_risk` or `choquet_distortion_risk`.

# Returns

A named tuple with risk `value` and the `pmf` that achieves it.

# See Also

See `choquet_ews_cvar` and `choquet_ews_tvar` for examples of ews functions

# Examples

```jldoctest
julia> choquet_ews([1,2,3,4,5],[0.2,0.2,0.2,0.2,0.2], choquet_ews_cvar(0.4)).valuey
1.5
```

```jldoctest
julia> round(choquet_ews([1,2,3,4,5],[0.2,0.2,0.2,0.2,0.2], choquet_ews_tvar(0.4)).value,digits=4)
1.1231
```
"""
function choquet_ews(x::AbstractVector{<:Real}, p::AbstractVector{<:Real},
                     ews::Tuple{<:Real,<:Real}; check_inputs = true)
    (m, c) = ews
    if check_inputs
        _check_pmf(x, p)
        zero(c) ≤ c ≤ one(c) || error("Input violates c ≥ 0")
        zero(m) < m  || error("Input violates m > 0")
        c + m ≥ one(m) || error("Input violates c + m ≥ 1")
    end

    T = float(eltype(x))
    α = (1 - c) / m
    v :: T = α < one(α) ? qql!(copy(x), copy(p), α)[1] : typemax(T)
    kmin :: Int = findmin(x)[2]
    zero(T) < p[kmin] || error("The function requires that p[argmin x] > 0")

    q = zeros(T, length(p))
    if x[kmin] ≈ v
        q[kmin] = one(T)
        return (value=v::T, pmf=q)
    end

    sumlt = @inbounds sum(p[i] for i ∈ eachindex(p,x) if x[i] < v || i == kmin, init=zero(T))
    θ :: T = 1 - min(c + m*sumlt, 1)
    value :: T = zero(T)
    @inbounds for i ∈ eachindex(p,x)
        if i == kmin
            q[i] = m * p[i] + c
        elseif x[i] < v
            q[i] = m * p[i]
        elseif x[i] == v
            q[i] = min(θ, m * p[i])
            θ -= q[i]
        end
        value += q[i] * x[i]
    end	
    return (value=value, pmf = q)
end

function choquet_ews_cvar(α :: Real)
    if α == zero(α)
        (0.0, 1.0) 
    else
        (Float64(1.0 / α), 0.0) # m, c
    end
end

function choquet_ews_tvar(α :: Real)
    if α == zero(α)
        (0.0, 1.0) 
    else
        (1.0, min(1, sqrt(0.5 * log(1/Float64(α))))) # m, c
    end
end
