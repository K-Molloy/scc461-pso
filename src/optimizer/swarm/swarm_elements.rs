use rand::prelude::ThreadRng;
use rand::distributions::{Distribution, Uniform};
use std::cmp::Ordering;
use std::fmt;


#[derive(Clone, Debug)]
pub struct Bound {
    dists: Vec<Uniform<f64>>,
    ranges: Vec<[f64; 2]>,
}

impl Bound {
    pub fn from_upper_lower(lims: Vec<[f64;2]>) -> Self {
        let dists = lims.iter().map(|x| {
            assert!(x[0] < x[1], "All lower bounds must be less than than upper bounds in lims vector!");
            Uniform::new(x[0], x[1])
        }).collect();

        Bound {
            dists: dists,
            ranges: lims,
        }
    }

    pub fn from_max(maxs: Vec<f64>) -> Self {
        let mut dists = Vec::new();
        let mut ranges = Vec::new();

        for x in maxs {
            assert!(x > 0.0, "Max values must be posative numbers in max -values vector!");
            dists.push(Uniform::new(-1.0 * x, x));
            ranges.push([-1.0 * x, x]);
        }

        Bound {dists, ranges}
    }

    pub fn sample_vec(&self, rng: &mut ThreadRng) -> Vec<f64> {
        self.dists.iter().map(|dist| dist.sample(rng)).collect()
    }

    pub fn range_vec(&self) -> Vec<[f64; 2]> {
        self.ranges.clone()
    }
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
       
        let num_vars = self.ranges.len() - 1;

        write!(f, "[")?;
        for r in 0..num_vars {
            let range = self.ranges[r];
            write!(f, "({} to {}), ", range[0], range[1])?;
        }

        write!(f, "({} to {})]", self.ranges[num_vars][0], self.ranges[num_vars][1])
    }
}

#[derive(Clone, Debug)]
pub struct Record {
    cost: f64,
    location: Vec<f64>
}

impl Record {
    pub fn blank(num_vars: usize) -> Self {
        Record {
            cost: std::f64::MAX,
            location: vec![0.0; num_vars],
        }
    }

    pub fn new(cost: f64, location: &[f64]) -> Self {
        Record{cost:cost, location: location.to_vec()}
    }

    pub fn blind_accumulate(&mut self, other_rec: &Self) {
        if other_rec.cost < self.cost {
            self.cost = other_rec.cost;
            self.location = other_rec.location.clone();
        }
    }

    pub fn loses_to(&self, other_cost: f64) -> bool {
        other_cost < self.cost
    }

    pub fn to_tuple(&self) -> (f64, Vec<f64>) {
        (self.cost, self.location.clone())
    }

    pub fn pos(&self, i: usize) -> f64 {
        self.location[i]
    }

    pub fn get_location(&self) -> Vec<f64> {
        self.location.clone()
    }

    pub fn get_cost(&self) -> f64 {
        self.cost
    }
}

impl PartialOrd for Record {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}

impl PartialEq for Record {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl fmt::Display for Record {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Record of: {:.20} @ \t [", self.cost)?;

        let num_vars = self.location.len()-1;

        for i in 0..num_vars {
            write!(f, "{:.5}, ", self.location[i])?;
        }

        write!(f, "{:.5}]", self.location[num_vars])
    }
}

pub struct SpaceRanger {
    num_variables: usize,
    pos_ranges: Vec<[f64; 2]>,
    vel_ranges: Vec<[f64; 2]>,
    wbf: f64,
}

impl SpaceRanger {
    pub fn new(num_variables: usize, pos_bounds: &Bound, vel_bounds: &Bound, wbf: f64) -> Self {
        SpaceRanger {
            num_variables: num_variables,
            pos_ranges: pos_bounds.range_vec(),
            vel_ranges: vel_bounds.range_vec(),
            wbf: wbf,
        }
    }

    pub fn constrain(&self, pos_vel: &mut[f64]) {
        for i in 0..self.num_variables {
            let iv = i + self.num_variables;

            if pos_vel[i] < self.pos_ranges[i][0] {
                pos_vel[i] = self.pos_ranges[i][0];
                pos_vel[iv] = pos_vel[iv] * self.wbf;
            } else if pos_vel[i] > self.pos_ranges[i][1] {
                pos_vel[i] = self.pos_ranges[i][1];
                pos_vel[iv] = pos_vel[iv] * self.wbf;
            }

            if pos_vel[iv] < self.vel_ranges[i][0] {
                pos_vel[iv] = self.vel_ranges[i][0];
            } else if pos_vel[iv] > self.vel_ranges[i][1] {
                pos_vel[iv] = self.vel_ranges[i][1];
            }

        }
    }
}