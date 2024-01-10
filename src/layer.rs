use crate::neuron::{ActivationFunction, Neuron};

pub struct Layer {
    neuron_list: Vec<Neuron>,
}

impl Layer {
    pub fn new(
        layer_width: usize,
        input_width: usize,
        activation_function: ActivationFunction,
    ) -> Layer {
        let mut neuron_list = Vec::new();

        for _ in 0..layer_width {
            neuron_list.push(Neuron::new(input_width, activation_function));
        }

        Layer { neuron_list }
    }

    fn compute_n_to_n(&self, input_params: Vec<f64>) -> Vec<f64> {
        let mut result_vec = Vec::new();

        for neuron in self.neuron_list.iter() {
            result_vec.push(neuron.compute(input_params.clone()));
        }

        result_vec
    }

    fn compute_n_to_1(&self, input_params: Vec<f64>) -> f64 {
        let mut result_vec = Vec::new();

        for neuron in self.neuron_list.iter() {
            result_vec.push(neuron.compute(input_params.clone()));
        }

        result_vec.into_iter().fold(0_f64, |acc, item| acc + item)
    }
}
