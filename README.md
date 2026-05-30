RiskMeasures
============

[![Build Status](https://github.com/RiskAverseRL/RiskMeasures/workflows/CI/badge.svg)](https://github.com/RiskAverseRL/RiskMeasures/actions)

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://riskaverserl.github.io/RiskMeasures.jl/dev/)


Julia library for computing risk measures for random variables. The random variable represents *profits* or *rewards* that are to be maximized. Also the computed risk value is preferable when it is greater.


The following risk measures are currently supported

- VaR: Value at risk
- CVaR: Conditional value at risk
- ERM: Entropic risk measure
- EVaR: Entropic value at risk
- expectile: A coherent elicitable risk measure
- UBSR: Utility-based shortfall risk for a given utility
- Choquet: Choquet and distortion risk measures for a choquet capacity and distortion functions

All risk measures, except ERM, are non-decreasing in risk level alpha. The ERM is non-increasing in level beta.

The package currently only supports random variables with discrete probability distributions, but support for continuous probability distributions is planned for the future. 


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

Please see the unit tests for examples of how this package can be used to compute the risk. 


## Future development plans:

- Additional common risk measures: Mean semi-deviation, 
- Analytical computation for special distributions, like Normal and others
- Add an optional integration with Mosek's exponential cones to support computation of EVaR. 
- Phi-divergence risk measures for any phi-divergence function

## See Also

- [Another RiskMeasures.jl](https://github.com/rubsc/RiskMeasures.jl)
- [ActuaryUtilities.jl](https://github.com/JuliaActuary/ActuaryUtilities.jl)
