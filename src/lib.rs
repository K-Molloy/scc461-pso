
use std::io;
extern crate rand;


// PSO Call
mod optimiser;
pub use optimiser::{JobConfig, PSOConfig, ParamDist, SwarmConfig, SwarmConfigDistribution, PSO};

// CPython Call
extern crate cpython;
use cpython::{PyResult, Python, py_module_initializer, py_fn};

// Python Connector - Module Initialiser
py_module_initializer!(ps_optim, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    m.add(py, "get_result", py_fn!(py, get_result(val: &str)))?;
    m.add(py, "solve", py_fn!(py, solve()))?;
    Ok(())
});

// PSO Solver
fn solve(_py: Python) -> PyResult<(f64, Vec<f64>)> {
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
    let min = pso.minimise_distributed(job_config, swarm_config, cost_function);

    println!("Minimum of: {}, With value: {:?}", min.0, min.1);

    Ok(min)

}

// Test Python Function
fn get_result(_py: Python, val: &str) -> PyResult<String> {
    Ok("Rust says: ".to_owned() + val)
}