RiskMeasures
============

Julia library for computing risk measures for random variables. The random variable represents profits or rewards that are to be maximized. The computed risk value is also better when greater.

The following risk measures are currently supported

- VaR: Value at risk
- CVaR: Conditional value at risk
- ERM: Entropic risk measure
- EVaR: Entropic value at risk
- expectile: A coherent elicitable risk measure
- UBSR: Utility-based shortfall risk for a given utility
- Choquet: Choquet and distortion risk measures for a choquet capacity and distortion functions

When supported, the risk measure returns also the optimal distribution 

## General assumptions

- Random variables represent *rewards* (greater value is preferred)
- Risk measures become less risk-averse with the increasing value of the risk parameter `α` or `β` 

## Supported distributions

- General discrete distributions (`DiscreteNonParametric`)
- Transformed discrete distributions ('DiscreteAffineDistribution')
- Mixtures of random variables ('MixtureModel')


## Examples

### Using arrays

```Julia
using RiskMeasures

x = [1, 5, 6, 7, 20]
p = [0.1, 0.1, 0.2, 0.5, 0.1]

VaR(x, p, 0.1)   # value at risk
CVaR(x, p, 0.1)  # conditional value at risk
EVaR(x, p, 0.1)  # entropic value at risk
ERM(x, p, 0.1)   # entropic risk measure
expectile(x, p, 0.1)   # expectile risk measure
UBSR(x, p, z -> (z ≥ 0 ? 0 : -1), 0.1)  # utility-based shortfall risk that equals to VaR
β = 0.1; UBSR(x, p, z -> (-exp(-β * z)), 0.1)  # utility-based shortfall risk that equals to ERM
choquet_risk(x, p, cvar_capacity, 0.5) # choquet risk measure
choquet_distortion_risk(x, p, cvar_distortion, 0.5) # law-invariant choquet (distortion) risk measure
```

### Using random variables

```Julia
using RiskMeasures
using Distributions

X = [1, 5, 6, 7, 20]
P = [0.1, 0.1, 0.2, 0.5, 0.1]
x̃ = DiscreteNonParametric(X, P)

VaR(x̃, 0.1)   # value at risk
CVaR(x̃, 0.1)  # conditional value at risk
EVaR(x̃, 0.1)  # entropic value at risk
ERM(x̃, 0.1)   # entropic risk measure
expectile(x̃, 0.1)   # expectile risk measure
UBSR(x̃, z -> (z ≥ 0 ? 0 : -1), 0.1)  # utility-based shortfall risk that equals to VaR
β = 0.1; UBSR(x̃, z -> (-exp(-β * z)), 0.1)  # utility-based shortfall risk that equals to ERM
choquet_risk(x̃, cvar_capacity, 0.5) # choquet risk measure
choquet_distortion_risk(x̃, cvar_distortion, 0.5) # law-invariant choquet (distortion) risk measure
```

### Using transformed random variables

We can also compute risk measures of transformed random variables

```Julia
using RiskMeasures
using Distributions

x̃ = DiscreteNonParametric([1, 5, 6, 7, 20], [0.1, 0.1, 0.2, 0.5, 0.1])
VaR(5*x̃ + 10, 0.1)   # value at risk
CVaR(x̃ - 10, 0.1)    # conditional value at risk
```

### Using mixture models

```Julia
using RiskMeasures
using Distributions

x̃ = DiscreteNonParametric([-5, -3, 2, 7, 33], [0., 0.2, 0.1, 0.5, 0.2])
ỹ = DiscreteNonParametric([1, 5, 6, 7, 20], [0.1, 0.1, 0.2, 0.5, 0.1])
m̃ = MixtureModel([x̃, ỹ], [0.3, 0.7])

VaR(m̃, 0.4)
CVaR(m̃, 0.4)
```

Please see the documentation and unit tests for more examples of how this package can be used to compute the risk. 

Most risk measure functions return tuples that include additional statistics and values. For example, this includes the distribution that attains the risk value and the optimal ERM-`β` in EVaR. 


# Functions

## Value at Risk

```@docs
VaR
```

## Conditional Value at Risk

```@docs
CVaR
```

## Entropic Value at Risk


```@docs
EVaR
```

## Entropic Risk Measure

```@docs
ERM
```

```@docs
softmin
```

## Expectile

```@docs
expectile
```

## Essential Infimum

```@docs
essinf
```

## Choquet Risk

```@docs
choquet_risk
```

```@docs
choquet_distortion_risk
```

```@docs
cvar_capacity
```

```@docs
cvar_distortion
```

```@docs
closure_c
```

## UBSR

```@docs
UBSR
```


# Notable Internal Functions

```@docs
RiskMeasures.partition!
```

```@docs
RiskMeasures.qql!
```
