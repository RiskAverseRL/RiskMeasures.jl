RiskMeasures
============

[![Build Status](https://github.com/RiskAverseRL/RiskMeasures/workflows/CI/badge.svg)](https://github.com/RiskAverseRL/RiskMeasures/actions)

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://riskaverserl.github.io/RiskMeasures.jl/dev/)


**Warning**: This is package is under development and the computed values should be treated with caution. 

Julia library for computing risk measures for random variables. The random variable represents *profits* or *rewards* that are to be maximized. Also the computed risk value is preferable when it is greater.


The following risk measures are currently supported

- VaR: Value at risk
- CVaR: Conditional value at risk
- ERM: Entropic risk measure
- EVaR: Entropic value at risk
- expectile: Expectile

All risk measures, except ERM, are non-decreasing in risk level alpha. The ERM is non-increasing in level beta.

The package currently only supports random variables with descrete probability distributions, but support for continuous probabilty distributions is planned for the future. 


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
expectile(x̃, 0.1)   # entropic risk measure
```

### Using random variables

```Julia
using RiskMeasures
using Distributions

x̃ = DiscreteNonParametric([1, 5, 6, 7, 20], [0.1, 0.1, 0.2, 0.5, 0.1])

VaR(x̃, 0.1)   # value at risk
CVaR(x̃, 0.1)  # conditional value at risk
EVaR(x̃, 0.1)  # entropic value at risk
ERM(x̃, 0.1)   # entropic risk measure
expectile(x̃, 0.1)   # entropic risk measure
```

### Using transformed random variables

We can also compute risk measures of transformed random variables

```Julia
VaR(5*x̃ + 10, 0.1)   # value at risk
CVaR(x̃ - 10, 0.1)  # conditional value at risk
```

Please see the unit tests for examples of how this package can be used to compute the risk. 

## Future development plans:

- Analytical computation for special distributions, like Normal and others
- Add an optional intergration with Mosek's exponential cones to support computation of EVaR. 
- Coquet capacity risk measures
- General risk measure construction from utility functions, such as CE, OCE, utility shortfall risk measures. 
- Phi-divergence risk mesures for any phi-divergence function

## See Also

- [MarketRisk.jl](https://github.com/mpkuperman/MarketRisk.jl)
