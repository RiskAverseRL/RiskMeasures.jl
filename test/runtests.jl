using RiskMeasures
using Test


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

    @test_throws ErrorException erm(X,p,-1.)
    @test_throws ErrorException erm(X,p2,2.)
end


@testset "VaR" begin
    p = [0.1, 0.2, 0.3, 0.1, 0.3, 0.0]
    X = [4, 5, 1, 2, -1, -2];

    @test var(X,p,1).var ≈ -1.0
    @test var(X, p, 0.99).var ≈ -1.0
    @test var(X, p, 0.).var ≈ 5.0
    @test var(X, p, 0.5).var ≈ 1.0
    @test var(X, p, 0.4).var ≈ 1.0

    p = [0.1, 0.2, 0.3, 0.1, 0.3]
    X = [4., 5., 1., 2., -1.]

    @test var(X, p, 1).var ≈ -1.0
    @test var(X, p, 0.99).var ≈ -1.0
    @test var(X, p, 0.0).var ≈ 5.0
    @test var(X, p, 0.5).var ≈ 1.0
    @test var(X, p, 0.4).var ≈ 1.0
end

@testset "VaR bounds" begin
    X = [2. , 5. , 6. , 9. , 3., 1.]
    p = [.1 , .1 , .2 , .5 , .1, 0.]
    p2 = [.1 , .1 , .2 , .5 , .1]

    @test_throws ErrorException var(X, p, -1) 
    @test_throws ErrorException var(X, p, 2) 
    @test_throws ErrorException var(X, p2, 0.5)
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

@testset "CVaR bounds" begin
    X = [2. , 5. , 6. , 9. , 3., 1.]
    p = [.1 , .1 , .2 , .5 , .1, 0.]
    p2 = [.1 , .1 , .2 , .5 , .1]

    @test_throws ErrorException cvar(X, p, -1) 
    @test_throws ErrorException cvar(X, p, 2) 
    @test_throws ErrorException cvar(X, p2, 0.5)
end
