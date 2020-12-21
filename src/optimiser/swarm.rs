mod particle;
mod swarm_config;
mod swarm_elements;

pub use swarm_config::{ParamDist, SwarmConfig, SwarmConfigDistribution};
use swarm_elements::SpaceRanger;
pub use swarm_elements::{Bound, Record};

use super::JobConfig;
use particle::Particle;
use rand::prelude::*;
use std::sync::{Arc, RwLock};

fn stochasticity(iteration: usize, max_itterations: usize) -> f64 {
    let progress = (iteration as f64) / (max_itterations as f64);
    assert!(progress >= 0.0 && progress <= 1.0);

    let arctan = 2.0 * (1.0 - (std::f64::consts::PI * (progress - 0.5)).atan()) + 0.25;
    let guass = (-1.0 * ((progress - std::f64::consts::FRAC_1_SQRT_2) / 0.025).powi(2)).exp();

    arctan + guass
}

pub struct SwarmColaborative {
    name: String,
    motion_coeffs: [f64; 4],
    _igf: f64,
    wbf: f64,
    tgse: usize,
    uce: Option<usize>,
    particles: Vec<Particle>,
    num_particles: usize,
    tribal_record: Record,
    global_record: Record,
}

impl SwarmColaborative {
    pub fn new(
        name: String,
        config: &SwarmConfig,
        update_console_every: Option<usize>,
        num_particles: usize,
    ) -> Self {
        SwarmColaborative {
            name: name,
            motion_coeffs: config.motion_coeffs(),
            _igf: config.inertial_growth_factor(),
            wbf: config.wall_bounce_factor(),
            tgse: config.tribal_global_share_every(),
            uce: update_console_every,
            particles: Vec::with_capacity(num_particles),
            num_particles: num_particles,
            tribal_record: Record::blank(1),
            global_record: Record::blank(1),
        }
    }

    pub fn run<F>(
        &mut self,
        job_config: JobConfig,
        get_cost: Arc<F>,
        global_record_lock: &mut Arc<RwLock<Record>>,
    ) where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = thread_rng();

        self.tribal_record = Record::blank(job_config.num_variables);
        self.global_record = Record::blank(job_config.num_variables);

        let mut no_new_record_count = 4;
        let mut try_globalise_agian = false;

        self.initialise_particles(
            job_config.num_variables,
            &job_config.pos_bounds,
            &job_config.vel_bounds,
            &mut rng,
        );

        let space_ranger = SpaceRanger::new(
            job_config.num_variables,
            &job_config.pos_bounds,
            &job_config.vel_bounds,
            self.wbf,
        );

        for i in 0..job_config.max_itterations {
            let stoch = stochasticity(i, job_config.max_itterations);

            self.update_particles(
                &space_ranger,
                &get_cost,
                &mut no_new_record_count,
                &mut rng,
                stoch,
            );

            self.globalise(i, global_record_lock, &mut try_globalise_agian);

            self.update_console(i, &no_new_record_count);

            if self.global_record.get_cost() < job_config.exit_cost {
                self.globalise(i, global_record_lock, &mut true);
                return;
            }
        }
    }

    fn initialise_particles(
        &mut self,
        num_variables: usize,
        pos_bounds: &Bound,
        vel_bounds: &Bound,
        rng: &mut ThreadRng,
    ) {
        for _ in 0..self.num_particles {
            let mut pos = pos_bounds.sample_vec(rng);
            let mut vel = vel_bounds.sample_vec(rng);

            self.particles
                .push(Particle::new(&mut pos, &mut vel, num_variables));
        }
    }

    fn update_particles<F>(
        &mut self,
        space_ranger: &SpaceRanger,
        get_cost: &Arc<F>,
        no_new_record_count: &mut i128,
        rng: &mut ThreadRng,
        stoch: f64,
    ) where
        F: Fn(&[f64]) -> f64,
    {
        let mut best_cost = self.tribal_record.get_cost();
        let mut best_cost_index = 0;

        let tribal_best_pos = self.tribal_record.get_location();
        let global_best_pos = self.global_record.get_location();

        for (i, p) in self.particles.iter_mut().enumerate() {
            p.update_colab(
                &self.motion_coeffs,
                rng,
                stoch,
                &tribal_best_pos,
                &global_best_pos,
            );

            space_ranger.constrain(p.pos_vel());

            if let Some(new_cost) = p.test_cost(get_cost) {
                if new_cost < best_cost {
                    best_cost = new_cost;
                    best_cost_index = i;
                }
            }
        }

        let top_record = self.particles[best_cost_index].get_record();

        if top_record < self.tribal_record {
            self.tribal_record = top_record;
            *no_new_record_count -= 1;

            if *no_new_record_count < 0 {
                *no_new_record_count = 0;
            } else if *no_new_record_count < 2 {
                //self.motion_coeffs[3] *= self._igf;
            } else if *no_new_record_count > 7 {
                //self.motion_coeffs[3] /= self._igf;
            }
        } else {
            *no_new_record_count += 1;
        }
    }

    fn globalise(
        &mut self,
        itteration: usize,
        global_record_lock: &mut Arc<RwLock<Record>>,
        try_agian: &mut bool,
    ) {
        if (itteration + 1) % self.tgse == 0 || *try_agian {
            match global_record_lock.try_write() {
                Ok(mut gr_write_ref) => {
                    if self.tribal_record < *gr_write_ref {
                        self.global_record = self.tribal_record.clone();
                        *gr_write_ref = self.global_record.clone();
                    } else {
                        self.global_record = (*gr_write_ref).clone();
                    }
                    *try_agian = false;
                }
                Err(_) => *try_agian = true,
            }
        }
    }

    fn update_console(&self, itteration: usize, no_new_record_count: &i128) {
        if let Some(update_period) = self.uce {
            if (itteration + 1) % update_period == 0 {
                println!(
                    "{} iter {}: \t c = {}, w = {} \n \t Tribal {} \n \t Global {}",
                    self.name,
                    itteration + 1,
                    *no_new_record_count,
                    self.motion_coeffs[3],
                    self.tribal_record,
                    self.global_record
                );
            }
        }
    }
}

pub struct SwarmIndependant {
    name: String,
    motion_coeffs: [f64; 4],
    _igf: f64,
    wbf: f64,
    uce: Option<usize>,
    particles: Vec<Particle>,
    num_particles: usize,
    record: Record,
}

impl SwarmIndependant {
    pub fn new(
        name: String,
        config: &SwarmConfig,
        update_console_every: Option<usize>,
        num_particles: usize,
    ) -> Self {
        SwarmIndependant {
            name: name,
            motion_coeffs: config.motion_coeffs(),
            _igf: config.inertial_growth_factor(),
            wbf: config.wall_bounce_factor(),
            uce: update_console_every,
            particles: Vec::with_capacity(num_particles),
            num_particles: num_particles,
            record: Record::blank(1),
        }
    }

    pub fn run<F>(&mut self, job_config: JobConfig, get_cost: Arc<F>) -> Record
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = thread_rng();

        self.record = Record::blank(job_config.num_variables);

        let mut no_new_record_count = 4;

        self.initialise_particles(
            job_config.num_variables,
            &job_config.pos_bounds,
            &job_config.vel_bounds,
            &mut rng,
        );

        let space_ranger = SpaceRanger::new(
            job_config.num_variables,
            &job_config.pos_bounds,
            &job_config.vel_bounds,
            self.wbf,
        );

        for i in 0..job_config.max_itterations {
            let stoch = stochasticity(i, job_config.max_itterations);

            self.update_particles(
                &space_ranger,
                &get_cost,
                &mut no_new_record_count,
                &mut rng,
                stoch,
            );

            self.update_console(i, &no_new_record_count);

            if self.record.get_cost() < job_config.exit_cost {
                return self.record.clone();
            }
        }

        self.record.clone()
    }

    fn initialise_particles(
        &mut self,
        num_variables: usize,
        pos_bounds: &Bound,
        vel_bounds: &Bound,
        rng: &mut ThreadRng,
    ) {
        for _ in 0..self.num_particles {
            let mut pos = pos_bounds.sample_vec(rng);
            let mut vel = vel_bounds.sample_vec(rng);

            self.particles
                .push(Particle::new(&mut pos, &mut vel, num_variables));
        }
    }

    fn update_particles<F>(
        &mut self,
        space_ranger: &SpaceRanger,
        get_cost: &Arc<F>,
        no_new_record_count: &mut i128,
        rng: &mut ThreadRng,
        stoch: f64,
    ) where
        F: Fn(&[f64]) -> f64,
    {
        let mut best_cost = self.record.get_cost();
        let mut best_cost_index = 0;

        let best_pos = self.record.get_location();

        for (i, p) in self.particles.iter_mut().enumerate() {
            p.update_indep(&self.motion_coeffs, rng, stoch, &best_pos);

            space_ranger.constrain(p.pos_vel());

            if let Some(new_cost) = p.test_cost(get_cost) {
                if new_cost < best_cost {
                    best_cost = new_cost;
                    best_cost_index = i;
                }
            }
        }

        let top_record = self.particles[best_cost_index].get_record();

        if top_record < self.record {
            self.record = top_record;
            *no_new_record_count -= 1;

            if *no_new_record_count < 0 {
                *no_new_record_count = 0;
            } else if *no_new_record_count < 2 {
                //self.motion_coeffs[3] *= self._igf;
            } else if *no_new_record_count > 7 {
                //self.motion_coeffs[3] /= self._igf;
            }
        } else {
            *no_new_record_count += 1;
        }
    }

    fn update_console(&self, itteration: usize, no_new_record_count: &i128) {
        if let Some(update_period) = self.uce {
            if (itteration + 1) % update_period == 0 {
                println!(
                    "{} iter {}: \t c = {}, w = {} \n \t {} ",
                    self.name,
                    itteration + 1,
                    *no_new_record_count,
                    self.motion_coeffs[3],
                    self.record,
                );
            }
        }
    }
}