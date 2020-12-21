extern crate rand;

mod optimiser;
pub use optimiser::{JobConfig, PSOConfig, ParamDist, SwarmConfig, SwarmConfigDistribution, PSO};

// use std::f64::consts::PI;

// use std::time::{Duration, Instant};

#[macro_use]


extern crate cpython;
extern crate ps_optim;

use cpython::{Python, PyResult};
use ps_optim::{PSO, PSOConfig, JobConfig, SwarmConfigDistribution, ParamDist};

fn optimiser() {
    let num_variables = 5;

    // Define a cost function to minimize:

    //data captured by closure
    let mins = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let coeffs = vec![0.25; num_variables];

    //cost function closure must take a &[f64] and return f64
    let cost_function = move |point: &[f64]| -> f64 {
        let mut sum = 0.0;
        for i in 0..num_variables {
            sum += (point[i] - mins[i]).powi(2) * coeffs[i];
        }
        sum
    };


    // Create a PSO Configuration:
    let pso_config = PSOConfig::new(
        8,          // 8 swarms (each spawned on their own thread)
        128,        // 128 particles per swarm
        10,         // console is updated by each thread every 10 itterations
        true        // optimizer is verbose (will provide more detailed information to console)
    );


    // Create a PSO:
    let pso = PSO::new(pso_config);


    // Create a Job Configuration:
    let job_config = JobConfig::new(
        num_variables, 
        vec![[-15.0, 15.0]; num_variables],  // [upper, lower] bounds for each variable
        vec![1.125; num_variables],          // max velocity for each variable
        100,                                 // run for 100 itterations
        0.0000001,                           // exit cost (swarms will stop when a cost of 0.0000001 is reached)
    );


    // Create a Swarm Configuration Distribution:

    //the optimizer will sample a new swarm configuration for each swarm
    //this is usefull for automatically creating a range of swarm behaviors
    //in this case we are using new_independant, so all 8 optimizations will run seperately in parallel
    let swarm_config = SwarmConfigDistribution::new_independant(
        ParamDist::Fixed(1.45),                 // local: fixed value of 1.45
        ParamDist::Range([1.65, 0.25]),         // tribal: random value: 1.65 +/- 25% 
        ParamDist::Fixed(0.4),                  // momentum: fixed value of 0.4
        ParamDist::Range([1.25, 0.05]),         // momentum growth factor: random value: 1.25 +/- 5%
        ParamDist::Fixed(0.0125),               // wall bounce factor: fixed value of 0.0125
    );


    // Minimize cost function:

    //use minimize_distributed to accept SwarmConfigDistribution
    let min = pso.minimize_distributed(job_config, swarm_config, cost_function);

    println!("Minimum of: {}, With value: {:?}", min.0, min.1);

    Ok(min)

}

py_module_initialiser!(libps_optim, initlibps_optim, PyInit_ps_optim, |py, m|{
    try!(m.add(py, "__doc__", "This module is implemented in rust! :)"));
    try!(m.add(py, "optimiser", py_fn!(py, optimiser())))
})