use core::f64;
use std::{fmt::Display, vec};

use crate::{
    network::NetworkError,
    neuron::{ActivationFunction, Neuron},
};

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
        next_layer_errors_caused: &Vec<f64>,
        next_layer_weights_by_neuron: &Vec<Vec<f64>>,
    ) -> Result<Vec<f64>, NetworkError> {
        self.neuron_list
            .iter_mut()
            .enumerate()
            .map(|(i, neuron)| {
                let mut next_layer_weights_for_this_neuron = vec![];
                for neuron_weights_list in next_layer_weights_by_neuron {
                    next_layer_weights_for_this_neuron.push(
                        *neuron_weights_list
                            .get(i)
                            .ok_or(NetworkError::ErrorsIncomplete)?,
                    );
                }
                neuron.calculate_error(
                    inputs,
                    next_layer_errors_caused,
                    &next_layer_weights_for_this_neuron,
                )
            })
            .collect()
    }

    pub fn compute_absolute_error(&self, input_params: &Vec<f64>, output: f64) -> f64 {
        self.compute_n_to_1(input_params) - output
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

    pub fn set_final_layer_error(&mut self, error: f64) {
        let neuron = self
            .neuron_list
            .first_mut()
            .expect("There should awlays be at least one neuron when calling this function.");
        neuron.set_error(error);
    }

    pub fn get_weights_by_neurons(&self) -> Vec<Vec<f64>> {
        let mut weights_by_neurons = vec![];

        for neuron in self.neuron_list.iter() {
            weights_by_neurons.push(neuron.weights.clone());
        }

        weights_by_neurons
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
