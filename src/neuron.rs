use std::fmt::Display;

use rand::Rng;

pub type ActivationFunction = fn(f64) -> f64;

pub struct Neuron {
    weights: Vec<f64>,
    pub bias: f64,
    current_error: f64,
    learning_rate: f64,
    activation_function: ActivationFunction,
    activation_function_prime: ActivationFunction,
}

impl Neuron {
    pub fn new(
        number_of_weights: usize,
        learning_rate: f64,
        activation_function: ActivationFunction,
        activation_function_prime: ActivationFunction,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let mut weights = Vec::new();

        for _ in 0..number_of_weights {
            let new_weight = rng.gen::<f64>();
            weights.push(new_weight);
        }

        let bias = rng.gen::<f64>();
        let current_error = 0.0;

        Neuron {
            weights,
            bias,
            current_error,
            learning_rate,
            activation_function,
            activation_function_prime,
        }
    }

    fn multiply_and_accumulate(&self, inputs: &Vec<f64>) -> f64 {
        self.weights
            .iter()
            .zip(inputs.iter())
            .fold(self.bias, |acc, (weight, input)| acc + (weight * input))
    }

    pub fn get_bias(&self) -> f64 {
        self.bias
    }

    pub fn compute(&self, inputs: &Vec<f64>) -> f64 {
        (self.activation_function)(self.multiply_and_accumulate(inputs))
    }

    pub fn calculate_error(mut self, inputs: &Vec<f64>, next_layer_errors: &Vec<f64>) {
        let temp_factor = (self.activation_function_prime)(self.multiply_and_accumulate(inputs));
        let total = self.multiply_and_accumulate(next_layer_errors);
        self.current_error = total * temp_factor;
    }

    pub fn step_gradient(mut self, inputs: Vec<f64>) {
        self.weights = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(weight, input)| -> f64 {
                weight.to_owned() - self.learning_rate * (self.current_error * input)
            })
            .collect::<Vec<f64>>();
        self.bias -= self.learning_rate * self.current_error;
    }

    pub fn set_error(&mut self, error: f64) {
        self.current_error = error
    }
}

impl Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut current_string: String = "".to_owned();
        //        for (i, weight) in self.weights.iter().enumerate() {
        //            current_string += "\n";
        //            current_string = current_string + &format!("Weight {}: {}", i, weight);
        //        }
        current_string = current_string + &format!("\nBias :{}", self.bias);
        write!(f, "{}", current_string)
    }
}
