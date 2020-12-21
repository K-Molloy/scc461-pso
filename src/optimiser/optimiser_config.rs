use std::fmt;

use super::Bound;

const MAX_THREADS: usize = 32;
const MAX_VARIABLES: usize = 128;

#[derive(Clone, Debug)]
pub struct PSOConfig {
    pub num_threads: usize,
    pub nppt: usize,
    pub uce: Option<usize>,
    pub is_verbose: bool
}

impl PSOConfig {
    pub fn new(num_swarms: usize, num_particles_per_thread: usize, update_console_every: usize, verbose: bool) -> Self {
        assert!(num_swarms <= MAX_THREADS, "Number of Swarms must not exceed {}!", MAX_THREADS);
        assert!(num_swarms > 0, "There must be at least one Swarm");

        let uce = match update_console_every == 0 {
            true => None,
            false => Some(update_console_every),
        };

        PSOConfig{
            num_threads: num_swarms, 
            nppt: num_particles_per_thread, 
            uce: uce,
            is_verbose: verbose,
        }
    }
}

impl fmt::Display for PSOConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} swarm-threads of {} particles each \n", self.num_threads, self.nppt)
    }
}

#[derive(Clone, Debug)]
pub struct JobConfig{
    pub num_variables: usize,
    pub pos_bounds: Bound,
    pub vel_bounds: Bound,
    pub max_itterations: usize,
    pub exit_cost: f64,
}

impl JobConfig {
    pub fn new(
        num_variables: usize, 
        variable_bounds: Vec<[f64;2]>, 
        max_velocities: Vec<f64>, 
        max_itterations: usize,
        exit_cost: f64,
    ) -> Self {
        assert!(num_variables <= MAX_VARIABLES, "Number of variables must not exceed {}!", MAX_VARIABLES);
        assert_eq!(num_variables, variable_bounds.len(), "Variable Bounds Vector must have one entry for each variable!");
        assert_eq!(num_variables, max_velocities.len(), "Max Velociteis Vector must have one entry for each variable!");

        let pos_bounds = Bound::from_upper_lower(variable_bounds);
        let vel_bounds = Bound::from_max(max_velocities);

        JobConfig {num_variables, pos_bounds, vel_bounds, max_itterations, exit_cost}
    }

    //add constructor with auto max_vel
}


impl fmt::Display for JobConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Optimiser will search a {} variable space bounded by {} \n
            and velocities bounded by {} \n
            Optimiser will run for {} itterations, unless the minimum cost drops below {} \n",
            self.num_variables, 
            self.pos_bounds, 
            self.vel_bounds,
            self.max_itterations,
            self.exit_cost,
        )
    }
}