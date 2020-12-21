// Sample Integration Tests
// Kieran Molloy 2020
// Lancaster University

extern crate rand;

extern crate ps_optim;
use ps_optim::{JobConfig, PSOConfig, ParamDist, SwarmConfig, SwarmConfigDistribution, PSO};

use std::f64::consts::PI;


#[test]
#[should_panic(expected = "optimizer did not converge!")]
fn test_exp_sin_panic() {
    let num_variables = 5;

    let opt_config = PSOConfig::new(8, 256, 1000, true);
    let opt = PSO::new(opt_config);

    let job_config = JobConfig::new(
        num_variables,
        vec![[-2.0 * PI, 2.0 * PI]; num_variables],
        vec![0.25; num_variables],
        10000,
        0.000000001,
    );

    //let swarm_config = SwarmConfig::new_collaborative(1.6, 1.7, 1.8, 0.5, 1.0125, 0.125, 25);
    let swarm_config = SwarmConfig::new_collaborative(0.65, 0.9, 1.7, 0.5, 1.125, 0.125, 32);

    let min = opt.minimize(job_config, swarm_config, move |pt: &[f64]| -> f64 {
        let mut sum = 0.0;
        let mut sin_sum = 0.0;
        for i in 0..num_variables {
            sum += pt[i].abs();
            sin_sum += pt[i].powi(2).sin();
        }
        sum * (-1.0 * sin_sum).exp()
    });

    assert!((min.0 - 0.0).abs() < 0.001, "optimizer did not converge!");
}

#[test]
fn test_exp_sin_pass() {
    let num_variables = 5;

    let opt_config = PSOConfig::new(8, 256, 1000, true);
    let opt = PSO::new(opt_config);

    let job_config = JobConfig::new(
        num_variables,
        vec![[-2.0 * PI, 2.0 * PI]; num_variables],
        vec![0.25; num_variables],
        10000,
        0.000000001,
    );

    let swarm_config = SwarmConfig::new_collaborative(1.6, 1.7, 1.8, 0.5, 1.0125, 0.125, 25);
    //let swarm_config = SwarmConfig::new_collaborative(0.65, 0.9, 1.7, 0.5, 1.125, 0.125, 32);

    let min = opt.minimize(job_config, swarm_config, move |pt: &[f64]| -> f64 {
        let mut sum = 0.0;
        let mut sin_sum = 0.0;
        for i in 0..num_variables {
            sum += pt[i].abs();
            sin_sum += pt[i].powi(2).sin();
        }
        sum * (-1.0 * sin_sum).exp()
    });

    assert!((min.0 - 0.0).abs() < 0.001, "optimizer did not converge!");
}

#[test]
fn test_opt_quad_collab() {
    let num_variables = 10;

    let opt_config = PSOConfig::new(8, 128, 0, true);
    let opt = PSO::new(opt_config);

    let job_config = JobConfig::new(
        num_variables,
        vec![[-10.0, 10.0]; num_variables],
        vec![1.125; num_variables],
        1000,
        0.00000000001,
    );

    let swarm_config = SwarmConfig::new_collaborative(0.35, 0.8, 1.9, 0.5, 1.125, 0.125, 15);

    let centers = vec![2.5; num_variables];
    let coeffs = vec![5.0; num_variables];

    // let now = Instant::now();

    let min = opt.minimize(job_config, swarm_config, move |pt: &[f64]| -> f64 {
        let mut sum = 0.0;
        for i in 0..num_variables {
            sum += (pt[i] - centers[i]).powi(2) * coeffs[i];
        }
        sum
    });
}

#[test]
fn test_opt_quad_indep() {
    let num_variables = 4;

    let opt_config = PSOConfig::new(5, 128, 100, true);
    let opt = PSO::new(opt_config);

    let job_config = JobConfig::new(
        num_variables,
        vec![[-10.0, 10.0], [-5.0, 5.0], [2.5, 20.0], [-20.0, 2.5]],
        vec![1.5; num_variables],
        100,
        0.00001,
    );

    let swarm_config_dist = SwarmConfigDistribution::new_independant(
        ParamDist::Fixed(1.45),
        ParamDist::Range([1.65, 0.125]),
        ParamDist::Range([0.4, 0.125]),
        ParamDist::Fixed(1.125),
        ParamDist::Range([0.125, 0.1]),
    );

    let centers = vec![2.5; num_variables];
    let coeffs = vec![5.0; num_variables];

    let min = opt.minimize_distributed(job_config, swarm_config_dist, move |pt: &[f64]| -> f64 {
        let mut sum = 0.0;
        for i in 0..num_variables {
            sum += (pt[i] - centers[i]).powi(2) * coeffs[i];
        }
        sum
    });
}