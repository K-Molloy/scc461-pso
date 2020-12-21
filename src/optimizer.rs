mod optimizer_config;
mod swarm;

pub use optimizer_config::{JobConfig, PSOConfig};
pub use swarm::{ParamDist, SwarmConfig, SwarmConfigDistribution};

use rand::prelude::thread_rng;
use std::sync::{Arc, RwLock};
use std::thread;
use swarm::{Bound, Record, SwarmColaborative, SwarmIndependant};

pub struct PSO {
    config: PSOConfig,
}

impl PSO {
    pub fn new(config: PSOConfig) -> Self {
        if config.is_verbose {
            println!("PSO initialized with {}", config)
        }

        PSO { config }
    }

    pub fn minimize_collaborative<F>(&self, job_config: JobConfig, cost_func: F) -> (f64, Vec<f64>)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send + 'static,
    {
        self.minimize(job_config, SwarmConfig::default_collab(), cost_func)
    }

    pub fn minimize_independant<F>(&self, job_config: JobConfig, cost_func: F) -> (f64, Vec<f64>)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send + 'static,
    {
        self.minimize(job_config, SwarmConfig::default_independant(), cost_func)
    }

    pub fn minimize<F>(
        &self,
        job_config: JobConfig,
        swarm_config: SwarmConfig,
        cost_func: F,
    ) -> (f64, Vec<f64>)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send + 'static,
    {
        let swarm_configs = (0..self.config.num_threads)
            .into_iter()
            .map(|_| swarm_config.clone())
            .collect();
        self.minimize_specific(job_config, swarm_configs, cost_func)
    }

    pub fn minimize_distributed<F>(
        &self,
        job_config: JobConfig,
        swarm_config_dist: SwarmConfigDistribution,
        cost_func: F,
    ) -> (f64, Vec<f64>)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send + 'static,
    {
        let mut rng = thread_rng();
        let swarm_configs = (0..self.config.num_threads)
            .into_iter()
            .map(|_| swarm_config_dist.sample_configuration(&mut rng))
            .collect();
        self.minimize_specific(job_config, swarm_configs, cost_func)
    }

    pub fn minimize_specific<F>(
        &self,
        job_config: JobConfig,
        swarm_configs: Vec<SwarmConfig>,
        cost_func: F,
    ) -> (f64, Vec<f64>)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send + 'static,
    {
        let is_collaborative_job = self.verify_swarm_configs(&swarm_configs);
        let get_cost = Arc::new(cost_func);

        if self.config.is_verbose {
            println!("Job Starting... \n {}", job_config)
        }

        return match is_collaborative_job {
            true => self.run_collaborative_job(job_config, swarm_configs, get_cost),
            false => self.run_independant_job(job_config, swarm_configs, get_cost),
        };
    }

    fn run_collaborative_job<F>(
        &self,
        job_config: JobConfig,
        swarm_configs: Vec<SwarmConfig>,
        get_cost: Arc<F>,
    ) -> (f64, Vec<f64>)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send + 'static,
    {
        let global_record = Arc::new(RwLock::new(Record::blank(job_config.num_variables)));

        let swarm_threads: Vec<Result<std::thread::JoinHandle<()>, std::io::Error>> =
            (0..self.config.num_threads)
                .into_iter()
                .map(|i| {
                    let mut swarm = SwarmColaborative::new(
                        format!("Swarm_{}", i),
                        &swarm_configs[i],
                        self.config.uce,
                        self.config.nppt,
                    );
                    let mut global_record_ref = global_record.clone();
                    let get_cost_ref = get_cost.clone();
                    let job_config_clone = job_config.clone();

                    thread::Builder::new()
                        .name(format!("swarm_thread_{}", i))
                        .spawn(move || {
                            swarm.run(job_config_clone, get_cost_ref, &mut global_record_ref);
                        })
                })
                .collect();

        for st in swarm_threads {
            match st {
                Ok(handle) => handle.join().expect("Thread Panicked during Job!"),
                Err(msg) => println!("Unable to join thread: {}", msg),
            }
        }

        let minimum = global_record.read().unwrap();
        if self.config.is_verbose {
            println!("Minimum {}", minimum)
        }

        (*minimum).to_tuple()
    }

    fn run_independant_job<F>(
        &self,
        job_config: JobConfig,
        swarm_configs: Vec<SwarmConfig>,
        get_cost: Arc<F>,
    ) -> (f64, Vec<f64>)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send + 'static,
    {
        let mut minimum = Record::blank(job_config.num_variables);

        let swarm_threads: Vec<Result<std::thread::JoinHandle<Record>, std::io::Error>> =
            (0..self.config.num_threads)
                .into_iter()
                .map(|i| {
                    let mut swarm = SwarmIndependant::new(
                        format!("Swarm_{}", i),
                        &swarm_configs[i],
                        self.config.uce,
                        self.config.nppt,
                    );
                    let get_cost_ref = get_cost.clone();
                    let job_config_clone = job_config.clone();

                    thread::Builder::new()
                        .name(format!("swarm_thread_{}", i))
                        .spawn(move || -> Record { swarm.run(job_config_clone, get_cost_ref) })
                })
                .collect();

        for st in swarm_threads {
            match st {
                Ok(handle) => {
                    let thread_record = handle.join().expect("Thread Panicked during Operation!");
                    minimum.blind_accumulate(&thread_record);
                }
                Err(msg) => println!("Unable to join thread: {}", msg),
            }
        }

        if self.config.is_verbose {
            println!("Minimum {}", minimum)
        }

        minimum.to_tuple()
    }

    fn verify_swarm_configs(&self, swarm_configs: &Vec<SwarmConfig>) -> bool {
        assert_eq!(
            self.config.num_threads,
            swarm_configs.len(),
            "One Swarm Configuration must be provided for each swarm-thread!"
        );

        if self.config.num_threads == 1 {
            assert!(
                !swarm_configs[0].is_collaborative(),
                "Single-Swarm Optimization may not use a Collaborative Swarm Configuration!"
            );
            if self.config.is_verbose {
                println!("Swarm Configuration: \n \t {}", swarm_configs[0])
            }
            return false;
        } else {
            if self.config.is_verbose {
                println!("Swarm Configurations: \n \t 0: {}", swarm_configs[0])
            }
            let is_collaborative_opt = swarm_configs[0].is_collaborative();
            for i in 1..self.config.num_threads {
                assert_eq!(is_collaborative_opt, swarm_configs[i].is_collaborative(),
                    "All swarm configurations must have the same Collaborative or Independant status!"
                );
                if self.config.is_verbose {
                    println!("\t {}: {}", i, swarm_configs[i])
                }
            }
            return is_collaborative_opt;
        }
    }
}