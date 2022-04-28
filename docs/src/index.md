RiskMeasures
============

Julia library for computing risk measures for random variables. The random variable represents profits or rewards that are to be maximized. The computed risk value is also better when greater.


The following risk measures are currently supported

- VaR: Value at risk
- CVaR: Conditional value at risk
- ERM: Entropic risk measure
- EVaR: Entropic value at risk

The focus is currently on random variables with categorical (discrete) probability distributions, but continuous probabilty distributions may be supported in the future too. 

In general, the smaller value of the risk parameter indicates that the risk measure is less risk-averse or that it is closer to the expectation operator. 

**Warning**: This is package is in development and the computed values should be treated with caution. 

## Examples

```Julia
using RiskMeasures
X = [1, 5, 6, 7, 20]
p = [0.1, 0.1, 0.2, 0.5, 0.1]

cvar(X, p, 0.1)
```

## See Also

- [MarketRisk.jl](https://github.com/mpkuperman/MarketRisk.jl)


# Functions

## Value at Risk

```@docs
var
```

## Conditional Value at Risk

```@docs
cvar
```

## Entropic Risk Measure

```@docs
erm
```

## Entropic Value at Risk


```@docs
evar
```
