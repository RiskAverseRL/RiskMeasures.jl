using RiskMeasures
using Test

using Statistics: median, mean
using LinearAlgebra: ones
using Distributions
import Random

@testset "ERM" begin
    X = [2.0, 5.0, 6.0, 9.0, 3.0, 1.0]
    p = [0.1, 0.1, 0.2, 0.5, 0.1, 0.0]
    x̃ = DiscreteNonParametric(X, p)

    @test isapprox(ERM(x̃, 0.0), ERM(x̃, 1e-5); atol=1e-3)
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

    @test VaR(x̃, 0).value ≈ -1.0
    @test VaR(x̃, 0, has_duplicates=false).value ≈ -1.0
    @test VaR(x̃, 0.01).value ≈ -1.0
    @test VaR(x̃, 0.01, has_duplicates=false).value ≈ -1.0
    @test VaR(x̃, 1).value ≈ Inf
    @test VaR(x̃, 1, has_duplicates=false).value ≈ Inf
    @test VaR(x̃, 0.5).value ≈ 1.0
    @test VaR(x̃, 0.5, has_duplicates=false).value ≈ 1.0
    @test VaR(x̃, 0.7).value ≈ 2.0
    @test VaR(x̃, 0.7, has_duplicates=false).value ≈ 2.0

    p = [0.1, 0.2, 0.3, 0.1, 0.3]
    X = [4.0, 5.0, 1.0, 2.0, -1.0]
    x̃ = DiscreteNonParametric(X, p)

    @test VaR(x̃, 0).value ≈ -1.0
    @test VaR(x̃, 0, has_duplicates=false).value ≈ -1.0
    @test VaR(x̃, 0.01).value ≈ -1.0
    @test VaR(x̃, 0.01, has_duplicates=false).value ≈ -1.0
    @test VaR(x̃, 1).value ≈ Inf
    @test VaR(x̃, 1, has_duplicates=false).value ≈ Inf
    @test VaR(x̃, 0.5).value ≈ 1.0
    @test VaR(x̃, 0.5, has_duplicates=false).value ≈ 1.0
    @test VaR(x̃, 0.7).value ≈ 2.0
    @test VaR(x̃, 0.7, has_duplicates=false).value ≈ 2.0

    # Bernoulli distribution
    X = [0.0, 1.0]
    p = [0.5, 0.5]
    x̃ = DiscreteNonParametric(X, p)
    @test VaR(x̃, 0.5).value ≈ 0.0
    @test VaR(x̃, 0.5, has_duplicates=false).value ≈ 0.0
    @test VaR(x̃, 0.51).value ≈ 1.0
    @test VaR(x̃, 0.51, has_duplicates=false).value ≈ 1.0
    @test VaR(x̃, 0.3).value ≈ 0.0
    @test VaR(x̃, 0.3, has_duplicates=false).value ≈ 0.0
    @test VaR(x̃, 0.7).value ≈ 1.0
    @test VaR(x̃, 0.7, has_duplicates=false).value ≈ 1.0
end

@testset "VaR/CVaR/EVaR bounds" begin
    X = [2.0, 5.0, 6.0, 9.0, 3.0, 1.0]
    p = [0.1, 0.1, 0.2, 0.5, 0.1, 0.0]
    x̃ = DiscreteNonParametric(X, p)

    @test_throws ErrorException VaR(x̃, -1)
    @test_throws ErrorException VaR(x̃, -1, has_duplicates=false)
    @test_throws ErrorException VaR(x̃, 2)
    @test_throws ErrorException VaR(x̃, 2, has_duplicates=false)

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

    @test VaR(x̃, 0.5).value ≈ median(X)
    @test VaR(x̃, 0.5, has_duplicates=false).value ≈ median(X)
    @test -VaR(-x̃, 0.5).value ≈ median(X)
    @test -VaR(-x̃, 0.5, has_duplicates=false).value ≈ median(X)

    # must be a bound on the median when it is not
    X = [2.0, 5.0, 6.0, 9.0, 3.0, 1.0]
    p = ones(length(X)) / length(X)
    x̃ = DiscreteNonParametric(X, p)

    @test -VaR(-x̃, 0.5).value ≥ median(X)
    @test -VaR(-x̃, 0.5, has_duplicates=false).value >= median(X)
    @test VaR(x̃, 0.5).value ≤ median(X)
    @test VaR(x̃, 0.5, has_duplicates=false).value <= median(X)

    # just test a larger version
    X = collect(-30:30)
    p = ones(length(X)) / length(X)
    x̃ = DiscreteNonParametric(X, p)

    @test VaR(x̃, 0.5).value ≈ median(X)
    @test VaR(x̃, 0.5, has_duplicates=false).value ≈ median(X)
    @test -VaR(-x̃, 0.5).value ≈ median(X)
    @test -VaR(-x̃, 0.5, has_duplicates=false).value ≈ median(X)
end

@testset "CVaR" begin
    p = [0.1, 0.2, 0.3, 0.1, 0.3, 0.0]
    X = [4, 5, 1, 2, -1, -2]
    x̃ = DiscreteNonParametric(X, p)

    @test CVaR(x̃, 0).value ≈ -1.0
    @test CVaR(x̃, 0.01).value ≈ -1.0
    @test CVaR(x̃, 1).value ≈ 1.6
    @test CVaR(x̃, 0.5).value ≈ -0.2
    @test CVaR(x̃, 0.6).value ≈ 0.0

    p = [0.1, 0.2, 0.3, 0.1, 0.3]
    X = [4.0, 5.0, 1.0, 2.0, -1.0]
    x̃ = DiscreteNonParametric(X, p)

    @test CVaR(x̃, 0).value ≈ -1.0
    @test CVaR(x̃, 0.00).value ≈ -1.0
    @test CVaR(x̃, 1).value ≈ 1.6
    @test CVaR(x̃, 0.5).value ≈ -0.2
    @test CVaR(x̃, 0.6).value ≈ 0
end

@testset "Risk measures order" begin
    p = [0.1, 0.2, 0.3, 0.1, 0.3, 0.0]
    X = [0, 5, 6, 2, -1, -10]
    x̃ = DiscreteNonParametric(X, p)

    for α ∈ range(0.0, 1.0, 10)
        @test all(p .≥ 0.0)

        @test minimum(X) ≤ EVaR(x̃, α; βmax=60).value

        @test EVaR(x̃, α; βmax=60).value ≤ CVaR(x̃, α).value
        @test CVaR(x̃, α).value ≤ VaR(x̃, α).value
        @test CVaR(x̃, α).value ≤ VaR(x̃, α, has_duplicates=false).value
        @test CVaR(x̃, α).value ≤ mean(x̃)
    end
end

@testset "Translation equivariance" begin
    p = [0.05, 0.1, 0.1, 0.05, 0.2, 0.5]
    X = [-4.7, 5.3, 1.6, 2.8, 10, -20]
    x̃ = DiscreteNonParametric(X, p)

    for α ∈ range(0.0, 1.0, 7)
        for c ∈ range(-10.0, 10.0, 6)
            @test VaR(x̃ + c, α).value - c ≈ VaR(x̃, α).value
            @test VaR(x̃ + c, α, has_duplicates=false).value - c ≈ VaR(x̃, α, has_duplicates=false).value
            @test CVaR(x̃ + c, α).value - c ≈ CVaR(x̃, α).value
            @test EVaR(x̃ + c, α).value - c ≈ EVaR(x̃, α).value
            if zero(α) < α < one(α) * 0.5
                @test ≈(expectile(x̃ + c, α).value, expectile(x̃, α).value + c, atol=1e-5, rtol=0.01)
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
            @test ≈(VaR(x̃ * c, α).value, c * VaR(x̃, α).value, atol=1e-5, rtol=0.01)
            @test ≈(VaR(x̃ * c, α, has_duplicates=false).value, c * VaR(x̃, α, has_duplicates=false).value, atol=1e-5, rtol=0.01)
            @test ≈(CVaR(x̃ * c, α).value, c * CVaR(x̃, α).value, atol=1e-5, rtol=0.01)
            @test ≈(EVaR(x̃ * c, α).value, c * EVaR(x̃, α).value, atol=1e-5, rtol=0.01)
            @test ≈(expectile(x̃ * c, α, check_inputs=false).value, c * expectile(x̃, α, check_inputs=false).value, atol=1e-5, rtol=0.01)
        end
    end
end

@testset "TI and PH" begin

    Random.seed!(1981)
    x̃ = DiscreteNonParametric(rand(Normal(), 10), rand(Dirichlet(ones(10))))

    for α ∈ range(0.0, 1.0, 5)
        for c ∈ range(-10.0, 10.0, 6)
            for a ∈ range(0.1, 10.0, 6)
                @test ≈(VaR(x̃ * a + c, α).value, a * VaR(x̃, α).value + c, atol=1e-5, rtol=0.001)
                @test ≈(VaR(x̃ * a + c, α, has_duplicates=false).value, a * VaR(x̃, α, has_duplicates=false).value + c, atol=1e-5, rtol=0.001)
                @test ≈(CVaR(x̃ * a + c, α).value, a * CVaR(x̃, α).value + c, atol=1e-5, rtol=0.001)
                @test ≈(EVaR(x̃ * a + c, α).value, a * EVaR(x̃, α).value + c, atol=1e-5, rtol=0.001)
            end
        end
    end
end

@testset "EVaR reciprocal" begin
    p = [0.05, 0.1, 0.1, 0.05, 0.2, 0.5]
    X = [-4.7, 5.3, 1.6, 2.8, 10, -20]
    x̃ = DiscreteNonParametric(X, p)

    for α ∈ range(0, 1, length=10)
        e1 = EVaR(x̃, α; reciprocal=false)
        e2 = EVaR(x̃, α; reciprocal=true)
        @test e1.value ≈ e2.value
        @test e1.β ≈ e2.β
    end
end

@testset "VaR/EVaR/CVaR distribution matches" begin
    X = collect(1.0:20.0)
    p = collect(20.0:-1:1.0)
    p .= p ./ sum(p)
    x̃ = DiscreteNonParametric(X, p)

    for α ∈ 0:0.2:1
        c = CVaR(x̃, α)
        @test c.value ≈ mean(c.pmf)
        e = EVaR(x̃, α)
        @test e.value ≈ mean(e.pmf)
        v = VaR(x̃, α)
        @test v.value ≈ mean(v.pmf)
        v = VaR(x̃, α, has_duplicates=false)
        @test v.value ≈ mean(v.pmf)
    end
end

@testset "Expectile" begin
    X = collect(1.0:20.0)
    p = collect(20.0:-1:1.0)
    p .= p ./ sum(p)
    x̃ = DiscreteNonParametric(X, p)

    @test expectile(x̃, 0.5).value ≈ X' * p
    @test_throws ErrorException expectile(x̃, -1.0)
    @test_throws ErrorException expectile(x̃, 1.0)
    @test_throws ErrorException expectile(x̃, 0.0)
    # Test monotonocity and subaddativity
    Y = rand(Float64, length(X))
    Z = rand(Float64, length(X))
    for i ∈ eachindex(X)
        Y[i] < X[i] && (Y[i] = X[i])
    end
    ỹ = DiscreteNonParametric(Y, p)
    z̃ = DiscreteNonParametric(Z, p)
    σ̃ = DiscreteNonParametric(X + Z, p)
    for α ∈ 0.1:0.01:0.9
        @test expectile(x̃, α).value ≥ expectile(ỹ, α).value
        if α <= 0.5
            @test expectile(σ̃, α).value + 1e-10 ≥ expectile(x̃, α).value + expectile(z̃, α).value
        else
            @test expectile(σ̃, α).value - 1e-10 ≤ expectile(x̃, α).value + expectile(z̃, α).value
        end
    end
end

