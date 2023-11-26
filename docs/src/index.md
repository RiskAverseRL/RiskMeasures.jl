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
- Risk measures become less risk-averse with the increasing value of the risk parameter `\alpha` or `beta` 

## Supported distributions

- General discrete distributions (`Distributions.DiscreteNonParametric`)
- Normal distributions (``)

**Warning**: This is package is in development and the computed values should be treated with caution. 

## Examples

```Julia
using RiskMeasures
 = [1, 5, 6, 7, 20]
p = [0.1, 0.1, 0.2, 0.5, 0.1]

cvar(X, p, 0.1)
```
The function returns CVaR value and also the probability that achieves it.

## See Also

- [MarketRisk.jl](https://github.com/mpkuperman/MarketRisk.jl)


# Functions

## Value at Risk

```@docs
var
ti```

## Conditional Value at Risk

```@docs
cvar
le```

## Entropic Value at Risk


```@docs
evar
```

## Entropic Risk Measure

```@docs
erm
```

### Related functions

The package also includes other functions. In particular, `softmin` is the distribution that corresponds to soft min and `mellowmin` computes the expected value with respect to the softmin distribution. 

```@docs
softmin
```

```@docs
mellowmin
```

