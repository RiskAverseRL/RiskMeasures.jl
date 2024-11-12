RiskMeasures
============

Julia library for computing risk measures for random variables. The random variable represents profits or rewards that are to be maximized. The computed risk value is also better when greater.

The following risk measures are currently supported

- VaR: Value at risk
- CVaR: Conditional value at risk
- ERM: Entropic risk measure
- EVaR: Entropic value at risk

When supported, the risk measure returns also the optimal distribution 

## General assumptions

- Random variables represent *rewards* (greater value is preferred)
- Risk measures become less risk-averse with the increasing value of the risk parameter `α` or `β` 

## Supported distributions

- General discrete distributions (`DiscreteNonParametric`)

**Warning**: This is package is in development and the computed values should be treated with caution. 

## Examples

```Julia
using RiskMeasures
using Distributions

x̃ = DiscreteNonParametric([1, 5, 6, 7, 20], [0.1, 0.1, 0.2, 0.5, 0.1])
 
VaR(x̃, 0.1)   # value at risk
CVaR(x̃, 0.1)  # conditional value at risk
EVaR(x̃, 0.1)  # entropic value at risk
ERM(x̃, 0.1)   # entropic risk measure
```

We can also compute risk measures of transformed random variables

```Julia
VaR(5*x̃ + 10, 0.1)   # value at risk
CVaR(x̃ - 10, 0.1)  # conditional value at risk
```

Extended methods `VaR_e`, `CVaR_e`, and `EVaR_e` also return additional statistics and values, such as the distribution that attains the risk value and the optimal `β` in EVaR.

## See Also

- [MarketRisk.jl](https://github.com/mpkuperman/MarketRisk.jl)


# Functions

## Value at Risk

```@docs
VaR
```

```@docs
VaR_e
```


## Conditional Value at Risk

```@docs
CVaR
```

```@docs
CVaR_e
```

## Entropic Value at Risk


```@docs
EVaR
```

```@docs
EVaR_e
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

```@docs
expectile_e
```

## Essential Infimum

```@docs
essinf
```

```@docs
essinf_e
```
