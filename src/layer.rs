use std::fmt::Display;

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

    pub fn compute_m_to_n(&self, input_params: &Vec<f64>) -> Vec<f64> {
        let mut result_vec = Vec::new();

        for neuron in self.neuron_list.iter() {
            result_vec.push(neuron.compute(input_params));
        }

        result_vec
    }

    pub fn compute_n_to_1(&self, input_params: &Vec<f64>) -> f64 {
        let mut result_vec = Vec::new();

        for neuron in self.neuron_list.iter() {
            result_vec.push(neuron.compute(input_params));
        }

        result_vec.into_iter().fold(0_f64, |acc, item| acc + item)
    }

    pub fn compute_layer_errors(
        &mut self,
        inputs: &Vec<f64>,
        next_layer_errors: &Vec<f64>,
    ) -> Vec<f64> {
        self.neuron_list
            .iter_mut()
            .map(|item| item.calculate_error(inputs, next_layer_errors))
            .collect::<Vec<f64>>()
    }

    pub fn compute_absolute_error(&self, input_params: &Vec<f64>, output: f64) -> f64 {
        self.compute_n_to_1(&input_params) - output
    }

    pub fn accumulate_bias(&self) -> f64 {
        let mut acc = 0.0;
        self.neuron_list
            .iter()
            .for_each(|neuron| acc += neuron.get_bias());
        acc
    }

    pub fn step_gradient(&mut self, inputs: &Vec<f64>) {
        for neuron in self.neuron_list.iter_mut() {
            neuron.step_gradient(inputs);
        }
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut current_string: String = "".to_owned();
        current_string += "Layer:";
        for neuron in self.neuron_list.iter() {
            current_string += &format!("\n\t\t{:#}", neuron);
        }
        write!(f, "{}", current_string)
    }
}
