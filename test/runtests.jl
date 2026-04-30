using Preferences

set_preferences!("RiskMeasures", "dispatch_doctor_mode" => "error")

using RiskMeasures
using Test
using DispatchDoctor

using Statistics: median, mean
using LinearAlgebra: ones
using Distributions
import Random

function putter()
    values = [-20.0, -4.7, 1.6, 2.8, 5.3, 10.0]
    pmf = [0.5, 0.05, 0.1, 0.05, 0.1, 0.2]
    α = 0.5

    a = EVaR(values, pmf, α, reciprocal=true)
    b = EVaR(values, pmf, α, reciprocal=false)
end


function compute_VaR(x::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real; kwargs...)
    v      = VaR(x, pmf, α; fast=false, kwargs...)
    v_fast = VaR(x, pmf, α; fast=true,  kwargs...)
    @test v.value ≈ v_fast.value
    @test v.index < 1 ? v_fast.index == v.index == -1 : true
    @test v.index < 1 || (x[v.index] == v.value == v_fast.value)
    @test v.index < 1 || (x[v.index] == x[v_fast.index])
    return v
end

function compute_VaR(x̃, α; kwargs...)
    v      = VaR(x̃, α; fast=false, kwargs...)
    v_fast = VaR(x̃, α; fast=true,  kwargs...)
    @test v.value ≈ v_fast.value
    @test v.value ≈ mean(v.pmf) 
    @test mean(v.pmf) ≈ mean(v_fast.pmf)
    return v
end

function compute_CVaR(x::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real; kwargs...)
    c      = CVaR(x, pmf, α; fast=false, kwargs...)
    c_fast = CVaR(x, pmf, α; fast=true,  kwargs...)
    @test c.value ≈ c_fast.value
    @test c.pmf ≈ c_fast.pmf atol=0.01
        c_choquet = choquet_risk(x, pmf, cvar_capacity, α)
        @test c_choquet ≈ c.value atol=1e-10
        c_distortion = choquet_distortion_risk(x, pmf, cvar_distortion, α)
        @test c_distortion ≈ c.value atol=1e-10
    return c
end

function compute_CVaR(x̃, α; kwargs...)
    c      = CVaR(x̃, α; fast=false, kwargs...)
    c_fast = CVaR(x̃, α; fast=true,  kwargs...)
    @test c.value ≈ c_fast.value
    @test mean(c.pmf)   ≈ mean(c_fast.pmf) 
    return c
end

function compute_EVaR(x̃, α; kwargs...)
    e       = EVaR(x̃, α; reciprocal=false, kwargs...)
    e_recip = EVaR(x̃, α; reciprocal=true,  kwargs...)

    #if !isapprox(e.β, e_recip.β, atol=0.0001)
    #    println(x̃, "\t", α)
    #end
    @test e.value ≈ e_recip.value
    #@test e.β ≈ e_recip.β atol=0.0001 #unstable when EVaR == essinf
    @test e.pmf.p ≈ e_recip.pmf.p atol=0.01
    @test mean(e.pmf) ≈ e.value atol=0.02
    @test mean(e_recip.pmf) ≈ e_recip.value atol=0.02
    return e
end

function compute_expectile(x̃, α; kwargs...)
    expectile(x̃, α; kwargs...)
end

@testset "ERM" begin
    X = [2.0, 5.0, 6.0, 9.0, 3.0, 1.0]
    p = [0.1, 0.1, 0.2, 0.5, 0.1, 0.0]
    x̃ = DiscreteNonParametric(X, p)

    @test ERM(x̃, 0.0) ≈ ERM(x̃, 1e-5) atol=1e-3
    @test ERM(x̃, 0.0) ≈ sum(X .* p)
    @test ERM(x̃, 0) ≥ ERM(x̃, 1) ≥ ERM(x̃, 2.0)
    @test ERM(x̃, Inf) ≈ 2.0

    # test translation invariance
    @test ERM(x̃ + 100, 2.0) ≈ ERM(x̃, 2.0) + 100.0
    @test ERM(x̃ - 100, 2.0) ≈ ERM(x̃, 2.0) - 100.0

    @test ERM(x̃ + 300.0, 3.0) ≈ ERM(x̃, 3.0) + 300.0
    @test ERM(x̃ - 300.0, 3.0) ≈ ERM(x̃, 3.0) - 300.0
end

@testset "ERM bounds" begin
    X = [2.0, 5.0, 6.0, 9.0, 3.0, 1.0]
    p = [0.1, 0.1, 0.2, 0.5, 0.1, 0.0]
    x̃ = DiscreteNonParametric(X, p)

    @test_throws ErrorException ERM(x̃, -1.0)
end


@testset "VaR" begin
    p = [0.1, 0.2, 0.3, 0.1, 0.3, 0.0]
    X = [4, 5, 1, 2, -1, -2]
    x̃ = DiscreteNonParametric(X, p)

    @test compute_VaR(x̃, 0).value    ≈ -1.0
    @test compute_VaR(x̃, 0.01).value ≈ -1.0
    @test compute_VaR(x̃, 1).value    ≈ Inf
    @test compute_VaR(x̃, 0.5).value  ≈ 1.0
    @test compute_VaR(x̃, 0.701).value  ≈ 4.0


    @test compute_VaR(X, p, 0).value    ≈ -1.0
    @test compute_VaR(X, p, 0.01).value ≈ -1.0
    @test compute_VaR(X, p, 1).value    ≈ Inf
    @test compute_VaR(X, p, 0.5).value  ≈ 1.0
    @test compute_VaR(X, p, 0.701).value  ≈ 4.0

    p = [0.1, 0.2, 0.3, 0.1, 0.3]
    X = [4.0, 5.0, 1.0, 2.0, -1.0]
    x̃ = DiscreteNonParametric(X, p)

    @test compute_VaR(x̃, 0).value    ≈ -1.0
    @test compute_VaR(x̃, 0.01).value ≈ -1.0
    @test compute_VaR(x̃, 1).value    ≈ Inf
    @test compute_VaR(x̃, 0.5).value  ≈ 1.0
    @test compute_VaR(x̃, 0.701).value  ≈ 4.0

    @test compute_VaR(X, p, 0).value    ≈ -1.0
    @test compute_VaR(X, p, 0.01).value ≈ -1.0
    @test compute_VaR(X, p, 1).value    ≈ Inf
    @test compute_VaR(X, p, 0.5).value  ≈ 1.0
    @test compute_VaR(X, p, 0.701).value  ≈ 4.0

    # Bernoulli distribution
    X = [0.0, 1.0]
    p = [0.5, 0.5]
    x̃ = DiscreteNonParametric(X, p)
    @test compute_VaR(x̃, 0.5).value  ≈ 1.0
    @test compute_VaR(x̃, 0.51).value ≈ 1.0
    @test compute_VaR(x̃, 0.3).value  ≈ 0.0
    @test compute_VaR(x̃, 0.7).value  ≈ 1.0
end

@testset "VaR/CVaR/EVaR bounds" begin
    X = [2.0, 5.0, 6.0, 9.0, 3.0, 1.0]
    p = [0.1, 0.1, 0.2, 0.5, 0.1, 0.0]
    x̃ = DiscreteNonParametric(X, p)

    @test_throws ErrorException VaR(x̃, -1)
    @test_throws ErrorException VaR(x̃, -1, fast=true)
    @test_throws ErrorException VaR(x̃, 2)
    @test_throws ErrorException VaR(x̃, 2, fast=true)

    @test_throws ErrorException CVaR(x̃, -1)
    @test_throws ErrorException CVaR(x̃, 2)

    @test_throws ErrorException EVaR(x̃, -1)
    @test_throws ErrorException EVaR(x̃, 2)

    @test_throws ErrorException expectile(x̃, -1)
    @test_throws ErrorException expectile(x̃, 2)
end

@testset "VaR median" begin
    # must find the median when it is unique
    X = [2.0, 5.0, 6.0, 9.0, 3.0, 1.0, -1.0]
    p = ones(length(X)) / length(X)
    x̃ = DiscreteNonParametric(X, p)

    @test compute_VaR(x̃, 0.5).value   ≈ median(X)
    @test -compute_VaR(-x̃, 0.5).value ≈ median(X)

    # must be a bound on the median when it is not
    X = [2.0, 5.0, 6.0, 9.0, 3.0, 1.0]
    p = ones(length(X)) / length(X)
    x̃ = DiscreteNonParametric(X, p)

    @test -compute_VaR(-x̃, 0.50001).value ≤ median(X)
    @test  compute_VaR( x̃, 0.50001).value ≥ median(X)

    # just test a larger version
    X = collect(-30:30)
    p = ones(length(X)) / length(X)
    x̃ = DiscreteNonParametric(X, p)

    @test  compute_VaR( x̃, 0.500001).value  ≈ median(X)
    @test -compute_VaR(-x̃, 0.5000001).value ≈ median(X)
end

@testset "CVaR" begin
    p = [0.1, 0.2, 0.3, 0.1, 0.3, 0.0]
    X = [4, 5, 1, 2, -1, -2]
    x̃ = DiscreteNonParametric(X, p)

    @test compute_CVaR(x̃, 0).value    ≈ -1.0
    @test compute_CVaR(x̃, 0.01).value ≈ -1.0
    @test compute_CVaR(x̃, 1).value    ≈ 1.6
    @test compute_CVaR(x̃, 0.5).value  ≈ -0.2
    @test compute_CVaR(x̃, 0.6).value  ≈ 0.0

    p = [0.1, 0.2, 0.3, 0.1, 0.3]
    X = [4.0, 5.0, 1.0, 2.0, -1.0]
    x̃ = DiscreteNonParametric(X, p)

    @test compute_CVaR(x̃, 0).value    ≈ -1.0
    @test compute_CVaR(x̃, 0.00).value ≈ -1.0
    @test compute_CVaR(x̃, 1).value    ≈ 1.6
    @test compute_CVaR(x̃, 0.5).value  ≈ -0.2
    @test compute_CVaR(x̃, 0.6).value  ≈ 0
end

@testset "Risk measures order" begin
    p = [0.1, 0.2, 0.3, 0.1, 0.3, 0.0]
    X = [0, 5, 6, 2, -1, -10]
    x̃ = DiscreteNonParametric(X, p)

    for α ∈ range(0.0, 1.0, 10)
        @test all(p .≥ 0.0)

        evar = compute_EVaR(x̃, α; βmax=60)
        cvar = compute_CVaR(x̃, α)
        var  = compute_VaR(x̃, α)

        @test minimum(X) ≤ evar.value
        @test evar.value ≤ cvar.value
        @test cvar.value ≤ var.value
        @test cvar.value ≤ mean(x̃)
    end
end

@testset "Translation equivariance" begin
    p = [0.05, 0.1, 0.1, 0.05, 0.2, 0.5]
    X = [-4.7, 5.3, 1.6, 2.8, 10, -20]
    x̃ = DiscreteNonParametric(X, p)

    for α ∈ range(0.0, 1.0, 7)
        for c ∈ range(-10.0, 10.0, 6)
            @test compute_VaR(x̃ + c, α).value - c  ≈ compute_VaR(x̃, α).value
            @test compute_CVaR(x̃ + c, α).value - c ≈ compute_CVaR(x̃, α).value
            @test compute_EVaR(x̃ + c, α).value - c ≈ compute_EVaR(x̃, α).value
            if zero(α) < α < one(α) * 0.5
                @test compute_expectile(x̃ + c, α).value ≈
                               compute_expectile(x̃, α).value + c atol=1e-5 
            end
        end
    end
end

@testset "Positive homogeneity" begin
    p = [0.05, 0.1, 0.1, 0.05, 0.2, 0.5]
    X = [-4.7, 5.3, 1.6, 2.8, 10, -20]
    x̃ = DiscreteNonParametric(X, p)

    for α ∈ range(0.0, 1.0, 5)
        for c ∈ range(0.1, 10.0, 6)
            @test compute_VaR(x̃ * c, α).value ≈
                    c * compute_VaR(x̃, α).value atol=1e-5 rtol=0.01
            @test compute_CVaR(x̃ * c, α).value ≈
                    c * compute_CVaR(x̃, α).value atol=1e-5 rtol=0.01
            @test compute_EVaR(x̃ * c, α).value ≈
                    c * compute_EVaR(x̃, α).value atol=1e-5 rtol=0.01
            @test compute_expectile(x̃ * c, α, check_inputs=false).value ≈
                    c * compute_expectile(x̃, α, check_inputs=false).value atol=1e-5 rtol=0.01
        end
    end
end

@testset "TI and PH" begin

    Random.seed!(1981)
    x̃ = DiscreteNonParametric(rand(Normal(), 10), rand(Dirichlet(ones(10))))

    for α ∈ range(0.0, 1.0, 5)
        for c ∈ range(-10.0, 10.0, 6)
            for a ∈ range(0.1, 10.0, 6)
                @test ≈(compute_VaR(x̃ * a + c, α).value,
                        a * compute_VaR(x̃, α).value  + c, atol=1e-5, rtol=0.001)
                @test ≈(compute_CVaR(x̃ * a + c, α).value,
                        a * compute_CVaR(x̃, α).value + c, atol=1e-5, rtol=0.001)
                @test ≈(compute_EVaR(x̃ * a + c, α).value,
                        a * compute_EVaR(x̃, α).value + c, atol=1e-5, rtol=0.001)
            end
        end
    end
end

@testset "EVaR reciprocal" begin
    p = [0.05, 0.1, 0.1, 0.05, 0.2, 0.5]
    X = [-4.7, 5.3, 1.6, 2.8, 10, -20]
    x̃ = DiscreteNonParametric(X, p)

    for α ∈ range(0, 1, length=10)
        compute_EVaR(x̃, α)
    end
end

@testset "VaR/EVaR/CVaR distribution matches" begin
    X = collect(1.0:20.0)
    p = collect(20.0:-1:1.0)
    p .= p ./ sum(p)
    x̃ = DiscreteNonParametric(X, p)

    for α ∈ 0:0.2:1
        c = compute_CVaR(x̃, α)
        @test c.value ≈ mean(c.pmf)
        e = compute_EVaR(x̃, α)
        @test e.value ≈ mean(e.pmf)
        v = compute_VaR(x̃, α)
        @test v.value ≈ mean(v.pmf)
    end
end

@testset "CVaR fast pmf" begin
    values = [3.0, 1.0, 2.0]
    pmf    = [0.3, 0.4, 0.3]
    for α ∈ range(0.0, 1.0, 6)
        c = compute_CVaR(values, pmf, α)
        @test values' * c.pmf ≈ c.value
    end
end

@testset "VaR fast index" begin
    values = [3.0, 1.0, 2.0]
    pmf    = [0.3, 0.4, 0.3]
    for α ∈ range(0.0, 1.0, 6)
        compute_VaR(values, pmf, α)
    end
end

@testset "Expectile" begin
    X = collect(1.0:20.0)
    p = collect(20.0:-1:1.0)
    p .= p ./ sum(p)
    x̃ = DiscreteNonParametric(X, p)

    @test compute_expectile(x̃, 0.5).value ≈ X' * p
    @test_throws ErrorException expectile(x̃, -1.0)
    @test_throws ErrorException expectile(x̃, 1.0)
    @test_throws ErrorException expectile(x̃, 0.0)
    # Test monotonocity and subaddativity
    Y = rand(Float64, length(X))
    Z = rand(Float64, length(X))
    for i ∈ eachindex(X)
        Y[i] < X[i] && (Y[i] = X[i])
    end
    ỹ = DiscreteNonParametric(Y, p)
    z̃ = DiscreteNonParametric(Z, p)
    σ̃ = DiscreteNonParametric(X + Z, p)
    for α ∈ 0.1:0.01:0.9
        @test compute_expectile(x̃, α).value ≥ compute_expectile(ỹ, α).value
        if α <= 0.5
            @test compute_expectile(σ̃, α).value + 1e-10 ≥
                compute_expectile(x̃, α).value + compute_expectile(z̃, α).value
        else
            @test compute_expectile(σ̃, α).value - 1e-10 ≤
                compute_expectile(x̃, α).value + compute_expectile(z̃, α).value
        end
    end
end

@testset "Large CVaR/VaR Test" begin
    Random.seed!(1981)
    n = 1e2 |> Int
    x = rand(Float64, n) .* 100
    pmf_uniform = ones(n) ./ n
    pmf_sparse = zeros(Float64, n)
    inds = unique(rand(1:n, Int(ceil(log(n)))))
    pmf_sparse[inds] .= 1 / length(inds)

    for α ∈ 0.1:0.05:0.9
        compute_VaR(x, pmf_uniform, α)
        compute_CVaR(x, pmf_uniform, α)
        compute_VaR(x, pmf_sparse, α)
        compute_CVaR(x, pmf_sparse, α)
    end
end

@testset "Check type stability" begin
    @test_throws DispatchDoctor.TypeInstabilityError RiskMeasures.test_stability(1)
end

# general quality assurance
using Aqua
Aqua.test_all(RiskMeasures)
