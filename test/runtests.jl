using Preferences

set_preferences!("RiskMeasures", "dispatch_doctor_mode" => "error")

using RiskMeasures
using Test
using DispatchDoctor

using Statistics: median, mean
using LinearAlgebra: ones
using Distributions
import Random

## === Disable warnings
import Logging
Logging.disable_logging(Logging.Warn)


function compute_VaR(x::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real; kwargs...)
    v = VaR(x, pmf, α; fast=false, kwargs...)
    v_fast = VaR(x, pmf, α; fast=true, kwargs...)
    v_ubsr = UBSR(x, pmf, z -> (z ≥ 0 ? 0 : -1), α)
    @test v.value ≈ v_fast.value
    @test v.index < 1 || x[v.index] == v.value
    @test v.index < 1 ||x[v_fast.index] == v_fast.value
    @test v.index < 1 ? v_fast.index == v.index == -1 : true
    @test v_ubsr.value ≈ v.value atol = 0.01
    return v
end

function compute_VaR(x̃, α; kwargs...)
    v = VaR(x̃, α; fast=false, kwargs...)
    v_fast = VaR(x̃, α; fast=true, kwargs...)
    v_ubsr = UBSR(x̃, z -> (z ≥ 0 ? 0 : -1), α)
    @test v.value ≈ v_fast.value
    @test v.value ≈ mean(v.pmf)
    @test mean(v.pmf) ≈ mean(v_fast.pmf)
    @test v_ubsr.value ≈ v.value atol = 0.01
    return v
end

function compute_CVaR(x::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real; kwargs...)
    c = CVaR(x, pmf, α; fast=false, kwargs...)
    c_fast = CVaR(x, pmf, α; fast=true, kwargs...)
    c_choquet = choquet_risk(x, pmf, cvar_capacity, α)
    c_distortion = choquet_distortion_risk(x, pmf, cvar_distortion, α)
    @test c.value ≈ c_fast.value
    @test c.pmf ≈ c_fast.pmf atol = 0.01
    @test c_choquet.value ≈ c.value atol = 1e-10
    @test c_distortion.value ≈ c.value atol = 1e-10
    return c
end

function compute_CVaR(x̃, α; kwargs...)
    c = CVaR(x̃, α; fast=false, kwargs...)
    c_fast = CVaR(x̃, α; fast=true, kwargs...)
    c_choquet = choquet_risk(x̃, cvar_capacity, α)
    c_distortion = choquet_distortion_risk(x̃, cvar_distortion, α)
    @test c.value ≈ c_fast.value
    @test mean(c.pmf) ≈ mean(c_fast.pmf)
    @test c_choquet.value ≈ c.value atol = 1e-10
    @test c_distortion.value ≈ c.value atol = 1e-10
    return c
end

function compute_EVaR(x̃, α; kwargs...)
    e = EVaR(x̃, α; reciprocal=false, kwargs...)
    e_recip = EVaR(x̃, α; reciprocal=true, kwargs...)

    @test e.value ≈ e_recip.value
    @test e.pmf.p ≈ e_recip.pmf.p atol = 0.01
    @test mean(e.pmf) ≈ e.value atol = 0.02
    @test mean(e_recip.pmf) ≈ e_recip.value atol = 0.02
    return e
end

function compute_expectile(x̃, α; kwargs...)
    v = expectile(x̃, α; kwargs...)
    # TODO : add the comparison with
    return v
end



@testset "Mixture model test" begin
    X = [1, 5, 6, 7, 20]
    P = [0.1, 0.1, 0.2, 0.5, 0.1]
    x̃ = DiscreteNonParametric(X, P)

    m̃ = MixtureModel([x̃, x̃], [0.3, 0.7])
    
    β = 0.1

    @test compute_VaR(x̃, 0.1).value ≈ compute_VaR(m̃, 0.1).value
    @test compute_CVaR(x̃, 0.1).value ≈ compute_CVaR(m̃, 0.1).value
    @test compute_EVaR(x̃, 0.1).value ≈ compute_EVaR(m̃, 0.1).value
    @test ERM(x̃, 0.1) ≈ ERM(m̃, 0.1)
    @test compute_expectile(x̃, 0.1).value ≈ compute_expectile(m̃, 0.1).value
    @test UBSR(x̃, z -> (z ≥ 0 ? 0 : -1), 0.1).value ≈
        UBSR(m̃, z -> (z ≥ 0 ? 0 : -1), 0.1).value
    @test UBSR(x̃, z -> (-exp(-β * z)), 0.1).value ≈
        UBSR(m̃, z -> (-exp(-β * z)), 0.1).value
    @test choquet_risk(x̃, cvar_capacity, 0.5).value ≈
        choquet_risk(m̃, cvar_capacity, 0.5).value
    @test choquet_distortion_risk(x̃, cvar_distortion, 0.5).value ≈
        choquet_distortion_risk(m̃, cvar_distortion, 0.5).value
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
    # Test monotonocity and sub/super-addativity
    X = rand(Float64, length(X))
    Y = rand(Float64, length(X))
    Z = rand(Float64, length(X))
    for i ∈ eachindex(X) # make sure that Y ≥ X
        Y[i] < X[i] && (Y[i] = X[i])
    end
    x̃ = DiscreteNonParametric(X, p)
    ỹ = DiscreteNonParametric(Y, p)
    z̃ = DiscreteNonParametric(Z, p)
    σ̃ = DiscreteNonParametric(X + Z, p)
    for α ∈ 0.1:0.01:0.9
        # check monotonocity
        @test compute_expectile(ỹ, α).value ≥ compute_expectile(x̃, α).value
        if α <= 0.5
            # check sub-additivity
            @test compute_expectile(σ̃, α).value + 1e-10 ≥
                compute_expectile(x̃, α).value + compute_expectile(z̃, α).value
        else
            # check super-additivity
            @test compute_expectile(σ̃, α).value - 1e-10 ≤
                compute_expectile(x̃, α).value + compute_expectile(z̃, α).value
        end
    end
end

@testset "Large CVaR/VaR Test" begin
    Random.seed!(1981)
    n = 1000 |> Int
    x = rand(Float64, n) .* 100
    pmf_uniform = ones(n) ./ n
    pmf_sparse = zeros(Float64, n)
    inds = unique(rand(1:n, Int(ceil(log(n)))))
    pmf_sparse[inds] .= 1 / length(inds)

    for α ∈ 0.01:0.05:0.99
      risk_level = α + 1e-5
      compute_VaR(x, pmf_uniform, risk_level)
      compute_CVaR(x, pmf_uniform, risk_level)
      compute_VaR(x, pmf_sparse, risk_level)
      compute_CVaR(x, pmf_sparse, risk_level)
    end
end

@testset "UBSR Test" begin
    function test_UBSR(x, p)
      u = (z) -> z
      v = UBSR(x, p, u, 0)
      @test v.value ≈ sum(x .* p) atol=1e-5
      dnp = DiscreteNonParametric(x, p)
      v2 = UBSR(dnp, u, 0)
      @test v2.value ≈ sum(x .* p) atol=1e-5
      # Special cases
      # ERM
      β = 0.5
      u = z -> (-exp(-β * z))
      v = UBSR(x, p, u, 1)
      @test v.value ≈ ERM(x, p, β) atol=1e-5
      # VaR
      α = 0.9001
      u = (z) -> (z > 0 ? 0 : -1)
      v = UBSR(x, p, u, α)
      @test v.value ≈ compute_VaR(x, p, α).value atol=1e-5
      # Expectile
      u = (z) -> (α * max(z, 0) - (1-α) * max(-z, 0))
      v = UBSR(x, p, u, 0)
      @test v.value ≈ compute_expectile(dnp, α).value atol=1e-5
    end
    x = [1.0, 2.0, 3.0]
    p = [0.2, 0.5, 0.3]
    test_UBSR(x, p)
    x = collect(1.0:20.0)
    p = collect(20.0:-1:1.0)
    p .= p ./ sum(p)
    test_UBSR(x, p)
    x = collect(-10.0:10.0)
    p = collect(1.0:21.0)
    p .= p ./ sum(p)
    test_UBSR(x, p)
end

@testset "Choquet risk capacity" begin
    x = [1,3,-5,3]
    p = [0.3,0.2,0.3,0.2]
    α = 0.4

    ρ(x, p, α) = CVaR(x, p, α).value
    v1 = choquet_risk(x, p, RiskMeasures.closure_c(ρ), α)
    v2 = CVaR(x, p, α)

    @test v1.value ≈ v2.value
    @test v1.pmf ≈ v2.pmf
end


@testset "Check type stability" begin
    @test_throws DispatchDoctor.TypeInstabilityError RiskMeasures.test_stability(1)
end

# general quality assurance
using Aqua
Aqua.test_all(RiskMeasures)
