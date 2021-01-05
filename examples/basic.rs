// Basic Example
// Kieran Molloy 2020
// Lancaster University

// Single Swarm - Single Thread

// Default Independant swarm config
// Search Space Bounds and Max Velocities still must be defined in JobConfig
// Fastest for higher single clock speed CPU cores (i.e Intel i7 10th gen)

//extern crate ps_optim;
use ps_optim::{PSO, PSOConfig, JobConfig};

fn main() {
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
        1,          // 1 swarm used in optimization
        256,        // 256 particles are spawned
        10,         // console is updated every 10 itterations
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
        0.0000001,                           // exit cost (optimization will stop when a cost of 0.0000001 is reached)
    );


    // Minimize cost function:

    //use minimize_independant to optimize with the default independant-swarm configuration
    //the next example will show how to use collaborative-swarms
    let min = pso.minimize_independant(job_config, cost_function);

    println!("Minimum of: {}, With value: {:?}", min.0, min.1);
}