use crate::{layer::Layer, neuron::ActivationFunction};
use std::fmt::Display;
use thiserror::Error;

pub type ErrorFunction = fn(f64, f64) -> f64;

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
    #[error("Acessed and empty neuron list.")]
    EmptyNeuronList,
    #[error("Incorrect definition of network width list.")]
    IncorrectNetworkWidthList,
    #[error("Input length is incompatible with network definition.")]
    InvalidInputInserted,
}

enum NetworkType {
    MultiLayerPerceptron,
    TwoLayerPerceptron,
    SingleNeuron,
}

pub struct Network {
    intermediate_values: Vec<Vec<f64>>,
    pub network_depth: usize,
    pub network_width: Vec<usize>,
    pub input_width: usize,
    pub learning_rate: f64,
    network_type: NetworkType,
    common_layers: Vec<Layer>,
    output_layer: Layer,
    input_layer: Layer,
    error_function: ErrorFunction,
}

impl Network {
    pub fn new(
        network_depth: usize,
        network_width: &[usize],
        input_width: usize,
        learning_rate: f64,
        activation_function: ActivationFunction,
        activation_function_prime: ActivationFunction,
        error_function: ErrorFunction,
    ) -> Result<Self, NetworkError> {
        let network_type = match network_depth {
            i if i == 1 => NetworkType::SingleNeuron,
            i if i == 2 => NetworkType::TwoLayerPerceptron,
            _ => NetworkType::MultiLayerPerceptron,
        };

        let (input_layer, output_layer, common_layers) = match network_type {
            NetworkType::MultiLayerPerceptron => Network::generate_layers_for_mlp(
                network_width,
                input_width,
                learning_rate,
                activation_function,
                activation_function_prime,
                network_depth,
            )?,
            NetworkType::TwoLayerPerceptron => Network::generate_layers_for_two_layer_perceptron(
                network_width,
                input_width,
                learning_rate,
                activation_function,
                activation_function_prime,
            )?,
            NetworkType::SingleNeuron => Network::generate_layers_for_single_neuron_model(
                input_width,
                learning_rate,
                activation_function,
                activation_function_prime,
            ),
        };

        let intermidiate_values: Vec<Vec<f64>> = Vec::new();

        let network_width = network_width.to_vec();

        let network = Network {
            intermediate_values: intermidiate_values,
            network_depth,
            network_width,
            input_width,
            learning_rate,
            network_type,
            common_layers,
            output_layer,
            input_layer,
            error_function,
        };
        Ok(network)
    }

    pub fn feedforward_compute_iteration(&mut self, inputs: &[f64]) -> Result<f64, NetworkError> {
        if inputs.len() != self.input_width {
            return Err(NetworkError::InvalidInputInserted);
        }

        self.intermediate_values.push(inputs.to_vec());

        match self.network_type {
            NetworkType::SingleNeuron => (),
            _ => {
                let inputs = self
                    .intermediate_values
                    .first()
                    .ok_or(NetworkError::IntermediateValuesIncomplete)?;

                let first_layer_result = self.input_layer.compute_m_to_n(inputs);

                self.intermediate_values.push(first_layer_result);
            }
        }

        if let NetworkType::MultiLayerPerceptron = self.network_type {
            for i in 0..self.network_depth - 2 {
                let inputs = self
                    .intermediate_values
                    .last()
                    .ok_or(NetworkError::IntermediateValuesIncomplete)?;

                let value = self
                    .common_layers
                    .get(i)
                    .ok_or(NetworkError::InvalidCommonLayers)?
                    .compute_m_to_n(inputs);

                self.intermediate_values.push(value)
            }
        }

        let output = self.output_layer.compute_n_to_1_without_activation_layer(
            self.intermediate_values
                .last()
                .ok_or(NetworkError::IntermediateValuesIncomplete)?,
        );

        Ok(output)
    }

    pub fn feedforward_compute_batch(
        &mut self,
        inputs: &[f64],
    ) -> Result<(Vec<Vec<f64>>, f64), NetworkError> {
        let mut intermediate_values = Vec::new();
        if inputs.len() != self.input_width {
            return Err(NetworkError::InvalidInputInserted);
        }

        intermediate_values.push(inputs.to_vec());

        match self.network_type {
            NetworkType::SingleNeuron => (),
            _ => {
                let inputs = intermediate_values
                    .first()
                    .ok_or(NetworkError::IntermediateValuesIncomplete)?;

                let first_layer_result = self.input_layer.compute_m_to_n(inputs);

                intermediate_values.push(first_layer_result);
            }
        }

        if let NetworkType::MultiLayerPerceptron = self.network_type {
            for i in 0..self.network_depth - 2 {
                let inputs = intermediate_values
                    .last()
                    .ok_or(NetworkError::IntermediateValuesIncomplete)?;

                let value = self
                    .common_layers
                    .get(i)
                    .ok_or(NetworkError::InvalidCommonLayers)?
                    .compute_m_to_n(inputs);

                intermediate_values.push(value)
            }
        }

        let output = self.output_layer.compute_n_to_1_without_activation_layer(
            intermediate_values
                .last()
                .ok_or(NetworkError::IntermediateValuesIncomplete)?,
        );

        Ok((intermediate_values, output))
    }

    pub fn backpropagate_error_batch(
        &mut self,
        final_error: f64,
        intermediate_values: Vec<Vec<f64>>,
    ) -> Result<(), NetworkError> {
        let mut intermediate_errors: Vec<Vec<f64>> = Vec::new();

        self.output_layer.set_final_layer_error(final_error)?;

        intermediate_errors.push(vec![final_error]);

        if let NetworkType::MultiLayerPerceptron = self.network_type {
            for i in (self.network_depth - 2)..0 {
                let next_layer_weights_by_neuron = self
                    .common_layers
                    .get(i + 1)
                    .unwrap_or(&self.output_layer)
                    .get_weights_by_neurons();

                let inputs = intermediate_values
                    .get(i)
                    .ok_or(NetworkError::IntermediateValuesIncomplete)?;

                let next_layer_errors_caused = intermediate_errors
                    .last()
                    .ok_or(NetworkError::ErrorsIncomplete)?;

                let value = self
                    .common_layers
                    .get_mut(i)
                    .ok_or(NetworkError::InvalidCommonLayers)?
                    .compute_layer_errors(
                        inputs,
                        next_layer_errors_caused,
                        &next_layer_weights_by_neuron,
                    )?;

                intermediate_errors.push(value)
            }
        }

        let next_layer_weights_by_neuron = self
            .common_layers
            .get(0)
            .unwrap_or(&self.output_layer)
            .get_weights_by_neurons();

        let inputs = intermediate_values
            .first()
            .ok_or(NetworkError::IntermediateValuesIncomplete)?;

        let next_layer_errors_caused = intermediate_errors
            .last()
            .ok_or(NetworkError::ErrorsIncomplete)?;

        self.input_layer.compute_layer_errors(
            inputs,
            next_layer_errors_caused,
            &next_layer_weights_by_neuron,
        )?;

        Ok(())
    }

    pub fn backpropagate_error_iteration(&mut self, final_error: f64) -> Result<(), NetworkError> {
        let mut intermediate_errors: Vec<Vec<f64>> = Vec::new();

        self.output_layer.set_final_layer_error(final_error)?;

        intermediate_errors.push(vec![final_error]);

        if let NetworkType::MultiLayerPerceptron = self.network_type {
            for i in (self.network_depth - 2)..0 {
                let next_layer_weights_by_neuron = self
                    .common_layers
                    .get(i + 1)
                    .unwrap_or(&self.output_layer)
                    .get_weights_by_neurons();

                let inputs = self
                    .intermediate_values
                    .get(i)
                    .ok_or(NetworkError::IntermediateValuesIncomplete)?;

                let next_layer_errors_caused = intermediate_errors
                    .last()
                    .ok_or(NetworkError::ErrorsIncomplete)?;

                let value = self
                    .common_layers
                    .get_mut(i)
                    .ok_or(NetworkError::InvalidCommonLayers)?
                    .compute_layer_errors(
                        inputs,
                        next_layer_errors_caused,
                        &next_layer_weights_by_neuron,
                    )?;

                intermediate_errors.push(value)
            }
        }

        let next_layer_weights_by_neuron = self
            .common_layers
            .get(0)
            .unwrap_or(&self.output_layer)
            .get_weights_by_neurons();

        let inputs = self
            .intermediate_values
            .first()
            .ok_or(NetworkError::IntermediateValuesIncomplete)?;

        let next_layer_errors_caused = intermediate_errors
            .last()
            .ok_or(NetworkError::ErrorsIncomplete)?;

        self.input_layer.compute_layer_errors(
            inputs,
            next_layer_errors_caused,
            &next_layer_weights_by_neuron,
        )?;

        Ok(())
    }

    pub fn step_gradient_iteration(&mut self, inputs: &[f64]) -> Result<(), NetworkError> {
        self.input_layer.step_gradient(inputs);

        if let NetworkType::MultiLayerPerceptron = self.network_type {
            for i in 0..self.network_depth - 2 {
                self.common_layers
                    .get_mut(i)
                    .ok_or(NetworkError::InvalidCommonLayers)?
                    .step_gradient(
                        self.intermediate_values
                            .get(i + 1)
                            .ok_or(NetworkError::IntermediateValuesIncomplete)?,
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

    pub fn step_gradient_batch(&mut self) -> Result<(), NetworkError> {
        self.input_layer.step_gradient_batch();

        if let NetworkType::MultiLayerPerceptron = self.network_type {
            for i in 0..self.network_depth - 2 {
                self.common_layers
                    .get_mut(i)
                    .ok_or(NetworkError::InvalidCommonLayers)?
                    .step_gradient_batch();
            }
        }

        self.output_layer.step_gradient_batch();

        self.reset_intermediate_values();

        Ok(())
    }

    fn train_iteration(&mut self, inputs: &[f64], aim: f64) -> Result<(), NetworkError> {
        let final_answer = self.feedforward_compute_iteration(inputs)?;
        let last_neuron_error = (self.error_function)(aim, final_answer);
        //println!("Inputs: {:?}", inputs);
        //println!("Resposta: {final_answer}. Objetivo:{aim}");
        //println!("Network error: {:.2?}", last_neuron_error);
        self.backpropagate_error_iteration(last_neuron_error)?;
        //println!("Pos backprogation: {}", self);
        self.step_gradient_iteration(inputs)?;
        //println!("Pos gradiente: {}", self);
        Ok(())
    }

    pub fn train_by_iterations(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[f64],
    ) -> Result<(), NetworkError> {
        for (i, input) in inputs.iter().enumerate() {
            self.train_iteration(
                input,
                *targets.get(i).ok_or(NetworkError::InvalidInputInserted)?,
            )?;
        }
        Ok(())
    }

    pub fn batch_train_all_input(
        &mut self,
        inputs: &Vec<Vec<f64>>,
        targets: &[f64],
    ) -> Result<(), NetworkError> {
        let mut total_error = 0.0;
        let mut total_intermediate_values = Vec::new();
        for (i, input) in inputs.iter().enumerate() {
            let (current_intermediate_values, final_answer) =
                self.feedforward_compute_batch(input)?;
            let last_neuron_error = (self.error_function)(
                *targets.get(i).ok_or(NetworkError::InvalidInputInserted)?,
                final_answer,
            );
            total_error += last_neuron_error;

            if total_intermediate_values.is_empty() {
                total_intermediate_values = current_intermediate_values;
            } else {
                for (index, list) in current_intermediate_values.iter().enumerate() {
                    for (secondary_index, item) in list.iter().enumerate() {
                        total_intermediate_values[index][secondary_index] += item;
                    }
                }
            }
        }
        let average_intermediate_values = total_intermediate_values
            .iter()
            .map(|item| {
                item.iter()
                    .map(|value| value / inputs.len() as f64)
                    .collect()
            })
            .collect();
        let average_error = total_error / inputs.len() as f64;
        self.backpropagate_error_batch(average_error, average_intermediate_values)?;
        self.step_gradient_batch()?;
        Ok(())
    }

    pub fn reset_intermediate_values(&mut self) {
        self.intermediate_values = vec![];
    }

    fn generate_layers_for_single_neuron_model(
        input_width: usize,
        learning_rate: f64,
        activation_function: fn(f64) -> f64,
        activation_function_prime: fn(f64) -> f64,
    ) -> (Layer, Layer, Vec<Layer>) {
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

    fn generate_layers_for_two_layer_perceptron(
        network_width: &[usize],
        input_width: usize,
        learning_rate: f64,
        activation_function: fn(f64) -> f64,
        activation_function_prime: fn(f64) -> f64,
    ) -> Result<(Layer, Layer, Vec<Layer>), NetworkError> {
        let input_layer = Layer::new(
            *network_width
                .first()
                .ok_or(NetworkError::IncorrectNetworkWidthList)?,
            input_width,
            learning_rate,
            activation_function,
            activation_function_prime,
        );

        let output_layer = Layer::new(
            1,
            *network_width
                .first()
                .ok_or(NetworkError::IncorrectNetworkWidthList)?,
            learning_rate,
            activation_function,
            activation_function_prime,
        );

        Ok((input_layer, output_layer, Vec::new()))
    }

    fn generate_layers_for_mlp(
        network_width: &[usize],
        input_width: usize,
        learning_rate: f64,
        activation_function: fn(f64) -> f64,
        activation_function_prime: fn(f64) -> f64,
        network_depth: usize,
    ) -> Result<(Layer, Layer, Vec<Layer>), NetworkError> {
        let mut common_layers = Vec::new();
        let input_layer = Layer::new(
            *network_width
                .first()
                .ok_or(NetworkError::IncorrectNetworkWidthList)?,
            input_width,
            learning_rate,
            activation_function,
            activation_function_prime,
        );

        for index in 0..network_depth - 2 {
            common_layers.push(Layer::new(
                *network_width
                    .get(index + 1)
                    .ok_or(NetworkError::IncorrectNetworkWidthList)?,
                *network_width
                    .get(index)
                    .ok_or(NetworkError::IncorrectNetworkWidthList)?,
                learning_rate,
                activation_function,
                activation_function_prime,
            ))
        }

        let output_layer = Layer::new(
            1,
            *network_width
                .last()
                .ok_or(NetworkError::IncorrectNetworkWidthList)?,
            learning_rate,
            activation_function,
            activation_function_prime,
        );

        Ok((input_layer, output_layer, common_layers))
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
