using RiskMeasures
using Test

using Statistics: median
using LinearAlgebra: ones

@testset "ERM" begin
    X = [2. , 5. , 6. , 9. , 3., 1.]
    p = [.1 , .1 , .2 , .5 , .1, 0.]
    
    @test isapprox(erm(X,p,0.), erm(X,p,1e-5); atol=1e-3)
    @test erm(X,p,0.) ≈ sum(X .* p)
    @test erm(X,p,0) ≥ erm(X,p,1) ≥ erm(X,p,2.)
    @test erm(X,p,Inf) ≈ 2.

    # test translation invariance
    @test erm(X .+ 100., p, 2.) ≈ erm(X, p, 2.) + 100.
    @test erm(X .- 100., p, 2.) ≈ erm(X, p, 2.) - 100.
    
    @test erm(X .+ 300., p, 3.) ≈ erm(X, p, 3.) + 300.
    @test erm(X .- 300., p, 3.) ≈ erm(X, p, 3.) - 300.
end

@testset "ERM bounds" begin
    X = [2. , 5. , 6. , 9. , 3., 1.]
    p = [.1 , .1 , .2 , .5 , .1, 0.]
    p2 = [.1 , .1 , .2 , .5 , .1]

    @test_throws ErrorException erm(X, p, -1.)
    @test_throws ErrorException erm(X, p2, 2.)
end


@testset "VaR" begin
    p = [0.1, 0.2, 0.3, 0.1, 0.3, 0.0]
    X = [4, 5, 1, 2, -1, -2];

    @test var(X,p,1).var ≈ -1.0
    @test var(X, p, 0.99).var ≈ -1.0
    @test var(X, p, 0.).var ≈ 5.0
    @test var(X, p, 0.5).var ≈ 1.0
    @test var(X, p, 0.4).var ≈ 2.0

    p = [0.1, 0.2, 0.3, 0.1, 0.3]
    X = [4., 5., 1., 2., -1.]

    @test var(X, p, 1).var ≈ -1.0
    @test var(X, p, 0.99).var ≈ -1.0
    @test var(X, p, 0.0).var ≈ 5.0
    @test var(X, p, 0.5).var ≈ 1.0
    @test var(X, p, 0.4).var ≈ 2.0
end

@testset "VaR bounds" begin
    X = [2. , 5. , 6. , 9. , 3., 1.]
    p = [.1 , .1 , .2 , .5 , .1, 0.]
    p2 = [.1 , .1 , .2 , .5 , .1]

    @test_throws ErrorException var(X, p, -1) 
    @test_throws ErrorException var(X, p, 2) 
    @test_throws ErrorException var(X, p2, 0.5)
end

@testset "VaR median" begin
    # must find the median when it is unique
    X = [2. , 5. , 6. , 9. , 3., 1., -1.]
    p = ones(length(X)) / length(X)

    @test var(X, p, 0.5).var ≈ median(X)
    @test -var(-X, p, 0.5).var ≈ median(X)

    # must be a bound on the media when it is not
    X = [2. , 5. , 6. , 9. , 3., 1.]
    p = ones(length(X)) / length(X)

    @test var(X, p, 0.5).var ≥ median(X)
    @test -var(-X, p, 0.5).var ≤ median(X)
    
    # just test a larger version
    X = collect(-30:30)
    p = ones(length(X)) / length(X)

    @test var(X, p, 0.5).var ≈ median(X)
    @test -var(-X, p, 0.5).var ≈ median(X)
end

@testset "CVaR" begin
    p = [0.1, 0.2, 0.3, 0.1, 0.3, 0.0]
    X = [4, 5, 1, 2, -1, -2];

    @test cvar(X, p, 1).cvar ≈ -1.
    @test cvar(X, p, 0.99).cvar ≈ -1.0
    @test cvar(X, p, 0.0).cvar ≈ 1.6
    @test cvar(X, p, 0.5).cvar ≈ -0.2
    @test cvar(X, p, 0.4).cvar ≈ 0.0
    
    p = [0.1, 0.2, 0.3, 0.1, 0.3]
    X = [4., 5., 1., 2., -1.]

    @test cvar(X, p, 1.).cvar ≈ -1.0
    @test cvar(X, p, 1.00).cvar ≈ -1.0
    @test cvar(X, p, 0.0).cvar ≈ 1.6
    @test cvar(X, p, 0.5).cvar ≈ -0.2
    @test cvar(X, p, 0.4).cvar ≈ 0
end

@testset "Risk measures order" begin
    p = [0.1, 0.2, 0.3, 0.1, 0.3, 0.0]
    X = [0, 5, 6, 2, -1, -10];

    for α ∈ range(0.,1.,10)
        @test all(p .≥ 0.)
        @test minimum(X) ≤ evar(X, p, α; βmax = 60).evar 
        @test evar(X, p, α; βmax = 60).evar ≤ cvar(X, p, α).cvar
        @test cvar(X, p, α).cvar ≤ var(X, p, α).var
        @test cvar(X, p, α).cvar ≤ mean(X, p)
        @test var(X, p, α).var ≤ maximum(X)
    end
end

@testset "Translation equivariance" begin
    p = [0.05, 0.1, 0.1, 0.05, 0.2, 0.5]
    X = [-4.7, 5.3, 1.6, 2.8, 10, -20];
    e = ones(length(X))
    for α ∈ range(0., 1., 5)
        for c ∈ range(-10., 10., 6)
            @test var(X .+ c .* e, p, α).var - c ≈ var(X, p, α).var
            @test cvar(X .+ c .* e, p, α).cvar - c ≈ cvar(X, p, α).cvar
            @test evar(X .+ c .* e, p, α).evar - c ≈ evar(X, p, α).evar
        end
    end
end

@testset "EVaR reciprocal" begin
    p = [0.05, 0.1, 0.1, 0.05, 0.2, 0.5]
    X = [-4.7, 5.3, 1.6, 2.8, 10, -20];

    for α ∈ range(0,1,length=10)
        e1 = evar(X, p, α; reciprocal = false)
        e2 = evar(X, p, α; reciprocal = true)
        @test e1.evar ≈ e2.evar
        @test e1.β ≈ e2.β
    end
end

@testset "CVaR bounds" begin
    X = [2. , 5. , 6. , 9. , 3., 1.]
    p = [.1 , .1 , .2 , .5 , .1, 0.]
    p2 = [.1 , .1 , .2 , .5 , .1]

    @test_throws ErrorException cvar(X, p, -1) 
    @test_throws ErrorException cvar(X, p, 2) 
    @test_throws ErrorException cvar(X, p2, 0.5)
end

@testset "CVaR distribution matches" begin
    X = collect(1.0:20.0)
    p = collect(20.0:-1:1.0)
    p .= p ./ sum(p)
    for α ∈ [0., 0.2, 0.4, 0.6, 0.8, 1.0]
        c = cvar(X, p, α)
        @test c.cvar ≈ sum(c.p.p .* X)
    end
end

@testset "EVaR distribution matches" begin
    X = collect(1.0:20.0)
    p = collect(20.0:-1:1.0)
    p .= p ./ sum(p)

    for α ∈ [0., 0.2, 0.4, 0.6, 0.8, 1.0]
        α = 0.3
        c = evar(X, p, α)
        @test c.evar ≈ sum(c.p.p .* X)
    end
end
