use rand::prelude::ThreadRng;
use rand::Rng;
use std::fmt;

#[derive(Clone, Debug)]
pub struct SwarmConfig{
    motion_coeffs: [f64; 4],
    igf: f64,
    wbf: f64,
    tgse: Option<usize>,
}

impl SwarmConfig {
    pub fn new_collaborative (
        local: f64, tribal: f64, global: f64, 
        inertial: f64, inertial_growth_factor: f64, 
        wall_bounce_factor: f64,
        tribal_global_share_every: usize,
    ) -> Self {

        let l = correct_sine(local);
        let t = correct_sine(tribal);
        let g = correct_sine(global);
        let i = correct_sine(inertial);

        let igf = correct_sine(inertial_growth_factor);
        let wbf = correct_sine(wall_bounce_factor);

        SwarmConfig {
            motion_coeffs: [l, t, g, i],
            igf: igf,
            wbf: wbf,
            tgse: Some(tribal_global_share_every),
        }
    }

    pub fn new_independant (
        local: f64, tribal: f64,
        inertial: f64, inertial_growth_factor: f64,
        wall_bounce_factor: f64,
    ) -> Self {

        let l = correct_sine(local);
        let t = correct_sine(tribal);
        let i = correct_sine(inertial);

        let igf = correct_sine(inertial_growth_factor);
        let wbf = correct_sine(wall_bounce_factor);

        SwarmConfig {
            motion_coeffs: [l, t, 0.0, i],
            igf: igf,
            wbf: wbf,
            tgse: None
        }
    }

    pub fn default_collab() -> Self {
        SwarmConfig {
            motion_coeffs: [1.45, 1.65, 1.55, 0.4],
            igf: 1.125,
            wbf: 0.125,
            tgse: Some(8),
        }
    }

    pub fn default_independant() -> Self {
        SwarmConfig {
            motion_coeffs: [1.45, 1.65, 0.0, 0.4],
            igf: 1.125,
            wbf: 0.125,
            tgse: None,
        }
    }

    pub fn motion_coeffs(&self) -> [f64;4] {
        self.motion_coeffs
    }

    pub fn inertial_growth_factor(&self) -> f64 {
        self.igf
    }

    pub fn wall_bounce_factor(&self) -> f64 {
        self.wbf * -1.0
    }

    pub fn tribal_global_share_every(&self) -> usize {
        match self.tgse {
            Some(tgse) => tgse,
            None => panic!("Independant swarm configurations do not have a tribal-global sharing period!")
        }
    }

    pub fn is_collaborative(&self) -> bool {
        match self.tgse {
            Some(_) => true,
            None => false,
        }
    }
}

impl fmt::Display for SwarmConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.tgse {
            Some(tribal_global_share_every) => {
                write!(f, "Collaborative Swarm [shares record with global collective every {} itterations]: \n", tribal_global_share_every)?;
            },
            None => {
                write!(f, "Independant Swarm: \n")?;
            }
        }

        write!(f, "\t \t Motion coefficients: {:?}, inertial growth factor: {}, wall bounce factor: {}", 
            self.motion_coeffs, self.igf, self.wbf,
        )

    }
}

#[derive(Clone, Debug)]
pub struct SwarmConfigDistribution {
    l: ParamDist,
    t: ParamDist,
    g: ParamDist,
    i: ParamDist,
    igf: ParamDist,
    wbf: ParamDist,
    tgse: Option<usize>
}

impl SwarmConfigDistribution {
    pub fn new_collaborative(
        local: ParamDist, tribal: ParamDist, global: ParamDist,
        inertial: ParamDist, inertial_growth_factor: ParamDist,
        wall_bounce_factor: ParamDist,
        tribal_global_share_every: usize,
    ) -> Self {

        let l = correct_sine_dist(local);
        let t = correct_sine_dist(tribal);
        let g = correct_sine_dist(global);
        let i = correct_sine_dist(inertial);

        let ifg = correct_sine_dist(inertial_growth_factor);
        let wbf = correct_sine_dist(wall_bounce_factor);

        SwarmConfigDistribution{
            l: l,
            t: t,
            g: g,
            i: i,
            igf: ifg,
            wbf: wbf,
            tgse: Some(tribal_global_share_every),
        }
    }

    pub fn new_independant(
        local: ParamDist, tribal: ParamDist,
        inertial: ParamDist, inertial_growth_factor: ParamDist,
        wall_bounce_factor: ParamDist,
    ) -> Self {
        let l = correct_sine_dist(local);
        let t = correct_sine_dist(tribal);
        let i = correct_sine_dist(inertial);

        let ifg = correct_sine_dist(inertial_growth_factor);
        let wbf = correct_sine_dist(wall_bounce_factor);

        SwarmConfigDistribution{
            l: l,
            t: t,
            g: ParamDist::Fixed(0.0),
            i: i,
            igf: ifg,
            wbf: wbf,
            tgse: None,
        }
    }

    pub fn sample_configuration(&self, rng: &mut ThreadRng) -> SwarmConfig {

        SwarmConfig {
            motion_coeffs: [self.l.sample(rng), self.t.sample(rng), self.g.sample(rng), self.i.sample(rng)],
            igf: self.igf.sample(rng),
            wbf: self.wbf.sample(rng),
            tgse: self.tgse,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ParamDist {
    Fixed(f64),
    Range([f64; 2]),
}

impl ParamDist {
    pub fn sample(&self, rng: &mut ThreadRng) -> f64 {
        match self {
            Self::Fixed(value) => *value,
            Self::Range(mean_var) => {
                let offset = rng.gen_range(-1.0 * mean_var[1], mean_var[1]);
                mean_var[0] * (1.0 + offset)
            }
        }
    }
}

fn correct_sine_dist(param: ParamDist) -> ParamDist {
    match param {
        ParamDist::Fixed(value) => ParamDist::Fixed(correct_sine(value)),
        ParamDist::Range(mean_var) => {
            assert!(mean_var[1] < 1.0 && mean_var[1] > 0.0, 
                "Parameter Distribution Variance must be between 0 and 1"
            );
            let mean = correct_sine(mean_var[0]);
            ParamDist::Range([mean, mean_var[1]])
        }
    }
}

fn correct_sine(param: f64) -> f64 {
    match param > 0.0 {
        true => param,
        false => -1.0 * param,
    }
}