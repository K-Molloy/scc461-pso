// Multi-Thread Example
// Kieran Molloy 2020
// Lancaster University

// Using 4 collaborate swarms with fewer particles in each

// A custom swam configuration is defined with motion coeffcienets and a record-sharing period
// Fastest for lower single clock speed CPU cores (i.e AMD Ryzen)

extern crate ps_optim;
use ps_optim::{PSO, PSOConfig, JobConfig, SwarmConfig};

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
        4,          // 4 swarms (each spawned on their own thread)
        64,         // 64 particles per swarm
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
        0.0000001,                           // exit cost (swarms will stop when a cost of 0.0000001 is reached)
    );


    // Create a custom Swarm Configuration:

    //collaborative swarms will share information with eachother about best known locations in the search space
    let swarm_config = SwarmConfig::new_collaborative(
        1.45,       // local weigth:    how much particles care about their best known location
        1.6,        // tribal weight:   how much particles care about their swarms best known location
        1.25,       // global weight:   how much particles care about the overall best known location
        0.4,        // inertial coefficient:    component of a particles velocity that contributes to its next velocity
        1.25,       // inertial growth factor:  how much inertia grows and shrinks throughout optimization
        0.125,      // wall bounce factor:      component of velocity that is saved when particle goes out of bounds
        10,         // tribal-global collab period:   swarms share best known location every 10 itterations
    );


    // Minimize cost function:

    //use minimize to optimize with a custom SwarmConfig
    let min = pso.minimize(job_config, swarm_config, cost_function);

    println!("Minimum of: {}, With value: {:?}", min.0, min.1);
}