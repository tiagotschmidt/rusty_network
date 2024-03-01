use crate::{layer::Layer, neuron::ActivationFunction};

pub struct Network {
    intermidiate_values: Vec<Vec<f64>>,
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

        let mut intermidiate_values: Vec<Vec<f64>> = Vec::new();

        Network {
            intermidiate_values,
            network_depth,
            network_width,
            input_width,
            learning_rate,
            common_layers,
            output_layer,
            input_layer,
        }
    }

    pub fn feedforward_compute(&mut self, inputs: Vec<f64>) -> Option<f64> {
        if inputs.len() != self.input_width {
            return None;
        }

        self.intermidiate_values.push(inputs);

        let first_layer_result = self
            .input_layer
            .compute_m_to_n(self.intermidiate_values.first().unwrap());

        self.intermidiate_values.push(first_layer_result);

        for i in 0..self.network_depth - 2 {
            self.intermidiate_values.push(
                self.common_layers
                    .get(i)?
                    .compute_m_to_n(self.intermidiate_values.last()?),
            )
        }

        let output = self
            .output_layer
            .compute_n_to_1(self.intermidiate_values.last()?);

        //        for layer in &intermidiate_values {
        //            for item in layer {
        //                print!("{:.1}\t", item);
        //            }
        //            println!("\n");
        //        }

        Some(output)
    }

    pub fn backpropagate_error(&mut self, final_error: f64) {
        let mut intermidiate_errors: Vec<Vec<f64>> = Vec::new();

        intermidiate_errors.push(vec![final_error]);

        let last_layer_result = self.output_layer.compute_layer_errors(
            self.intermidiate_values.last().unwrap(),
            intermidiate_errors.last().unwrap(),
        );

        intermidiate_errors.push(last_layer_result);

        for i in (self.network_depth - 2)..0 {
            intermidiate_errors.push(self.common_layers.get_mut(i).unwrap().compute_layer_errors(
                self.intermidiate_values.get(i).unwrap(),
                intermidiate_errors.last().unwrap(),
            ))
        }

        self.input_layer.compute_layer_errors(
            self.intermidiate_values.first().unwrap(),
            intermidiate_errors.last().unwrap(),
        );
    }
}
