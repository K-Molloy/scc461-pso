use super::Record;
use rand::prelude::ThreadRng;
use rand::Rng;
use std::sync::Arc;

pub struct Particle {
    pos_vel: Vec<f64>,
    rec: Record,
    num_vars: usize,
}

impl Particle {
    pub fn new(pos: &mut Vec<f64>, vel: &mut Vec<f64>, num_vars: usize) -> Self {
        pos.append(vel);
        Particle {
            pos_vel: pos.clone(),
            rec: Record::blank(num_vars),
            num_vars: num_vars,
        }
    }

    pub fn update_colab(
        &mut self,
        motion_coeffs: &[f64; 4],
        rng: &mut ThreadRng,
        stoch: f64,
        tribal_best: &Vec<f64>,
        global_best: &Vec<f64>,
    ) {
        let rands: Vec<Vec<f64>> = (0..3)
            .map(|_| {
                (0..self.num_vars)
                    .map(|_| rng.gen_range(0.2, stoch))
                    .collect()
            })
            .collect();

        for pi in 0..self.num_vars {
            let vi = pi + self.num_vars;
            self.pos_vel[pi] += self.pos_vel[vi];

            self.pos_vel[vi] = motion_coeffs[3] * self.pos_vel[vi]
                + motion_coeffs[0] * rands[0][pi] * (self.rec.pos(pi) - self.pos_vel[pi])
                + motion_coeffs[1] * rands[1][pi] * (tribal_best[pi] - self.pos_vel[pi])
                + motion_coeffs[2] * rands[2][pi] * (global_best[pi] - self.pos_vel[pi]);
        }
    }

    pub fn update_indep(
        &mut self,
        motion_coeffs: &[f64; 4],
        rng: &mut ThreadRng,
        stoch: f64,
        best: &Vec<f64>,
    ) {
        let rands: Vec<Vec<f64>> = (0..2)
            .map(|_| {
                (0..self.num_vars)
                    .map(|_| rng.gen_range(0.2, stoch))
                    .collect()
            })
            .collect();

        for pi in 0..self.num_vars {
            let vi = pi + self.num_vars;
            self.pos_vel[pi] += self.pos_vel[vi];

            self.pos_vel[vi] = motion_coeffs[3] * self.pos_vel[vi]
                + motion_coeffs[0] * rands[0][pi] * (self.rec.pos(pi) - self.pos_vel[pi])
                + motion_coeffs[1] * rands[1][pi] * (best[pi] - self.pos_vel[pi]);
        }
    }

    pub fn pos_vel(&mut self) -> &mut [f64] {
        &mut self.pos_vel[..]
    }

    fn pos_slice(&self) -> &[f64] {
        &self.pos_vel[0..self.num_vars]
    }

    pub fn test_cost<F>(&mut self, get_cost: &Arc<F>) -> Option<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let cost = get_cost(self.pos_slice());

        return match self.rec.loses_to(cost) {
            true => {
                self.rec = Record::new(cost, self.pos_slice());
                Some(cost)
            }
            false => None,
        };
    }

    pub fn get_record(&self) -> Record {
        self.rec.clone()
    }
}