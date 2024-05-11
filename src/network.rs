use crate::{layer::Layer, neuron::ActivationFunction};
use std::fmt::Display;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Input data is width{0}, incompatible with network width {1}.")]
    InputIncompatibleWidth(usize, usize),
    #[error("Intermediate values is incorrect.")]
    IntermediateValuesIncomplete,
    #[error("Failed to get/interact with the common layers in the network. Check network depth")]
    InvalidCommonLayers,
    #[error("Failed to get/interact with the common layers in the network. Check network depth")]
    ErrorsIncomplete,
}

pub struct Network {
    intermediate_values: Vec<Vec<f64>>,
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

        let (input_layer, output_layer, common_layers) = match network_depth {
            i if i == 1 => {
                let output_layer = Layer::new(
                    1,
                    input_width,
                    learning_rate,
                    activation_function,
                    activation_function_prime,
                );

                let empty_layer = Layer::new(
                    0,
                    input_width,
                    learning_rate,
                    activation_function,
                    activation_function_prime,
                );

                (empty_layer, output_layer, Vec::new())
            }
            i if i == 2 => {
                let input_layer = Layer::new(
                    network_width,
                    input_width,
                    learning_rate,
                    activation_function,
                    activation_function_prime,
                );

                let output_layer = Layer::new(
                    1,
                    input_width,
                    learning_rate,
                    activation_function,
                    activation_function_prime,
                );

                (input_layer, output_layer, Vec::new())
            }
            _ => {
                let input_layer = Layer::new(
                    network_width,
                    input_width,
                    learning_rate,
                    activation_function,
                    activation_function_prime,
                );
                let output_layer = Layer::new(
                    1,
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

                (input_layer, output_layer, common_layers)
            }
        };

        let intermidiate_values: Vec<Vec<f64>> = Vec::new();

        Network {
            intermediate_values: intermidiate_values,
            network_depth,
            network_width,
            input_width,
            learning_rate,
            common_layers,
            output_layer,
            input_layer,
        }
    }

    pub fn feedforward_compute(&mut self, inputs: &Vec<f64>) -> Result<f64, NetworkError> {
        self.intermediate_values.push(inputs.clone());

        if self.network_depth > 1 {
            let first_layer_result = self.input_layer.compute_m_to_n(
                self.intermediate_values
                    .first()
                    .ok_or(NetworkError::IntermediateValuesIncomplete)?,
            );

            self.intermediate_values.push(first_layer_result);
        }

        if self.network_depth > 2 {
            for i in 0..self.network_depth - 2 {
                self.intermediate_values.push(
                    self
                        .common_layers
                        .get(i)
                        .ok_or(NetworkError::InvalidCommonLayers)?
                        .compute_m_to_n(
                            self.intermediate_values
                                .last()
                                .ok_or(NetworkError::IntermediateValuesIncomplete)?,
                        ),
                )
            }
        }

        let output = self.output_layer.compute_n_to_1(
            self.intermediate_values
                .last()
                .ok_or(NetworkError::IntermediateValuesIncomplete)?,
        );

        Ok(output)
    }

    pub fn backpropagate_error(&mut self, final_error: f64) -> Result<(), NetworkError> {
        let mut intermediate_errors: Vec<Vec<f64>> = Vec::new();

        self.output_layer.set_final_layer_error(final_error);

        intermediate_errors.push(vec![final_error]);

        if self.network_depth > 2 {
            for i in (self.network_depth - 2)..0 {
                let next_layer_weights_by_neuron = self
                    .common_layers
                    .get(i + 1)
                    .unwrap_or(&self.output_layer)
                    .get_weights_by_neurons();
                intermediate_errors.push(
                    self.common_layers
                        .get_mut(i)
                        .ok_or(NetworkError::InvalidCommonLayers)?
                        .compute_layer_errors(
                            self.intermediate_values
                                .get(i)
                                .ok_or(NetworkError::IntermediateValuesIncomplete)?,
                            intermediate_errors
                                .last()
                                .ok_or(NetworkError::ErrorsIncomplete)?,
                            &next_layer_weights_by_neuron,
                        )?,
                )
            }
        }

        let next_layer_weights_by_neuron = self
            .common_layers
            .get(0)
            .unwrap_or(&self.output_layer)
            .get_weights_by_neurons();

        self.input_layer.compute_layer_errors(
            self.intermediate_values
                .first()
                .ok_or(NetworkError::IntermediateValuesIncomplete)?,
            intermediate_errors
                .last()
                .ok_or(NetworkError::ErrorsIncomplete)?,
            &next_layer_weights_by_neuron,
        )?;

        Ok(())
    }

    pub fn step_gradient(&mut self, inputs: &Vec<f64>) -> Result<(), NetworkError> {
        self.input_layer.step_gradient(inputs);

        if self.network_depth > 2 {
            for i in 0..self.network_depth - 2 {
                self.common_layers
                    .get_mut(i)
                    .ok_or(NetworkError::InvalidCommonLayers)?
                    .step_gradient(
                        self.intermediate_values
                            .get(i + 1)
                            .expect("Intermidiate values must be initialized!"),
                    );
            }
        }

        self.output_layer.step_gradient(
            self.intermediate_values
                .last()
                .ok_or(NetworkError::IntermediateValuesIncomplete)?,
        );

        self.reset_intermediate_values();

        Ok(())
    }

    pub fn batch_train_one_iteration(
        &mut self,
        inputs: &Vec<f64>,
        aim: f64,
    ) -> Result<(), NetworkError> {
        let final_answer = self.feedforward_compute(inputs)?;
        let last_neuron_error = -2.0 * (aim - final_answer);
        //println!("Result: {final_answer}, {aim}");
        println!("Network error: {:.2?}", aim - final_answer);
        self.backpropagate_error(last_neuron_error)?;
        //println!("Pos backprogation: {}", self);
        self.step_gradient(inputs)?;
        //println!("Pos gradiente: {}", self);
        Ok(())
    }

    pub fn reset_intermediate_values(&mut self) {
        self.intermediate_values = vec![];
    }
}

impl Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut current_string: String = "".to_owned();
        current_string += "\nNetwork:";
        current_string += &format!("\n\t*{:#}", self.input_layer);
        for layer in self.common_layers.iter() {
            current_string += &format!("\n\t*{:#}", layer);
        }
        current_string += &format!("\n\t*{:#}", self.output_layer);
        write!(f, "{}", current_string)
    }
}
