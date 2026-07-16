#![allow(dead_code)]

use crate::training::config::LrScheduleConfig;

pub struct LrScheduler {
    base_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    max_steps: usize,
    schedule_type: String,
    current_step: usize,
}

impl LrScheduler {
    pub fn new(base_lr: f64, config: &LrScheduleConfig) -> Self {
        let min_lr = base_lr * config.min_lr_ratio;
        Self {
            base_lr,
            min_lr,
            warmup_steps: config.warmup_steps,
            max_steps: config.max_steps,
            schedule_type: config.schedule_type.clone(),
            current_step: 0,
        }
    }

    pub fn get_lr(&self) -> f64 {
        let step = self.current_step as f64;

        if step < self.warmup_steps as f64 {
            // Linear warmup
            self.base_lr * (step / self.warmup_steps as f64)
        } else if step >= self.max_steps as f64 {
            self.min_lr
        } else {
            match self.schedule_type.as_str() {
                "cosine" => self.cosine_decay(step),
                "linear" => self.linear_decay(step),
                "constant" => self.base_lr,
                _ => self.cosine_decay(step),
            }
        }
    }

    fn cosine_decay(&self, step: f64) -> f64 {
        let progress =
            (step - self.warmup_steps as f64) / (self.max_steps - self.warmup_steps) as f64;
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        self.min_lr + (self.base_lr - self.min_lr) * cosine
    }

    fn linear_decay(&self, step: f64) -> f64 {
        let progress =
            (step - self.warmup_steps as f64) / (self.max_steps - self.warmup_steps) as f64;
        self.base_lr * (1.0 - progress) + self.min_lr * progress
    }

    pub fn step(&mut self) {
        self.current_step += 1;
    }

    pub fn current_step(&self) -> usize {
        self.current_step
    }

    pub fn set_step(&mut self, step: usize) {
        self.current_step = step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_schedule() {
        let config = LrScheduleConfig {
            schedule_type: "cosine".to_string(),
            warmup_steps: 100,
            max_steps: 1000,
            min_lr_ratio: 0.1,
        };
        let mut sched = LrScheduler::new(1e-4, &config);

        assert_eq!(sched.get_lr(), 0.0); // step 0

        sched.set_step(50);
        let lr = sched.get_lr();
        assert!(lr > 0.0 && lr < 1e-4); // warmup

        sched.set_step(100);
        assert!((sched.get_lr() - 1e-4).abs() < 1e-6); // end of warmup

        sched.set_step(1000);
        assert!((sched.get_lr() - 1e-5).abs() < 1e-6); // min lr
    }
}
