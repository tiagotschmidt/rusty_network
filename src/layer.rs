use crate::neuron::{ActivationFunction, Neuron};

pub struct Layer {
    neuron_list: Vec<Neuron>,
}

impl Layer {
    pub fn new(
        layer_width: usize,
        input_width: usize,
        learning_rate: f64,
        activation_function: ActivationFunction,
        activation_function_prime: ActivationFunction,
    ) -> Layer {
        let mut neuron_list = Vec::new();

        for _ in 0..layer_width {
            neuron_list.push(Neuron::new(
                input_width,
                learning_rate,
                activation_function,
                activation_function_prime,
            ));
        }

        Layer { neuron_list }
    }

    pub fn compute_n_to_n(&self, input_params: Vec<f64>) -> Vec<f64> {
        let mut result_vec = Vec::new();

        for neuron in self.neuron_list.iter() {
            result_vec.push(neuron.compute(input_params.clone()));
        }

        result_vec
    }

    pub fn compute_n_to_1(&self, input_params: Vec<f64>) -> f64 {
        let mut result_vec = Vec::new();

        for neuron in self.neuron_list.iter() {
            result_vec.push(neuron.compute(input_params.clone()));
        }

        result_vec.into_iter().fold(0_f64, |acc, item| acc + item)
    }

    pub fn compute_mse(&self, input_params: Vec<f64>, output: f64) -> f64 {
        ((self.compute_n_to_1(input_params.clone()) - output)
            * (self.compute_n_to_1(input_params) - output))
            .sqrt()
    }

    pub fn set_final_layer_error(&mut self, input_params: Vec<f64>, output: f64) {
        let temp = self.compute_mse(input_params, output);
        let neuron = self.neuron_list.first_mut().unwrap();
        neuron.set_error(temp)
    }

    pub fn accumulate_bias(&self) -> f64 {
        let mut acc = 0.0;
        self.neuron_list
            .iter()
            .for_each(|neuron| acc += neuron.get_bias());
        acc
    }
}
