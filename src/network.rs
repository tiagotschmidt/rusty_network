use crate::{layer::Layer, neuron::ActivationFunction};

pub struct Network {
    pub network_depth: usize,
    pub network_width: usize,
    pub input_width: usize,
    pub learning_rate: f64,
    common_layers: Vec<Layer>,
    output_layer: Layer,
    input_layer: Layer,
}

impl Network {
    pub fn new(
        network_depth: usize,
        network_width: usize,
        input_width: usize,
        learning_rate: f64,
        activation_function: ActivationFunction,
        activation_function_prime: ActivationFunction,
    ) -> Self {
        let mut common_layers = Vec::new();
        let input_layer = Layer::new(
            network_width,
            input_width,
            learning_rate,
            activation_function,
            activation_function_prime,
        );
        let output_layer = Layer::new(
            network_width,
            input_width,
            learning_rate,
            activation_function,
            activation_function_prime,
        );
        for _ in 0..network_depth - 2 {
            common_layers.push(Layer::new(
                network_width,
                network_width,
                learning_rate,
                activation_function,
                activation_function_prime,
            ))
        }
        Network {
            network_depth,
            network_width,
            input_width,
            learning_rate,
            common_layers,
            output_layer,
            input_layer,
        }
    }

    pub fn compute_input(&self, inputs: Vec<f64>) -> Option<f64> {
        if inputs.len() != self.input_width {
            return None;
        }

        let mut intermidiate_values: Vec<Vec<f64>> = Vec::new();

        intermidiate_values.push(inputs);

        let first_layer_result = self
            .input_layer
            .compute_m_to_n(intermidiate_values.first().unwrap());

        intermidiate_values.push(first_layer_result);

        for i in 0..self.network_depth - 2 {
            intermidiate_values.push(
                self.common_layers
                    .get(i)?
                    .compute_m_to_n(intermidiate_values.last()?),
            )
        }

        Some(
            self.output_layer
                .compute_n_to_1(intermidiate_values.last()?),
        )
    }
}
