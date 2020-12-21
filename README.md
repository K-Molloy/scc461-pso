# scc461-pso
## Generic Particle Swarm Optimiser
### Kieran Molloy | Lancaster University 2020
Highly Configurable Particle Swarm Optimiser implemented in pure Rust

## Introduction

#### Easily mimimise a cost function across multiple threads
* Multiple optimisations can be run in parallel with __Independant Swarms__
* or Swarms can periodically share best-known locations in the search space with __Collaborative Swarms__

#### Motion dynamics are loosly based on [Matlab's Particle Swarm Algorithm](https://www.mathworks.com/help/gads/particle-swarm-optimization-algorithm.html)

#### Optimizer takes a closure with the following header as the cost function:
```Rust
move |x: &[f64]| -> f64 {}
```
* The input slice-array represents a point in the N dimensional optimization space
* The returned cost is used to navigate the search space to locate a minimum
* currently only supports f64, but future updates may allow more generic cost-functions

## Installation


## Examples


