use crate::functions::activation_functions::ActivationFunction;
use crate::functions::error_functions::ErrorFunction;
use crate::layer::Layer;
use crate::network_model::{NetworkError, NetworkType};
use std::fmt::Display;

pub struct PipelineNetwork {
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

impl PipelineNetwork {
    new_network_function!(PipelineNetwork);

    fn feedforward_compute_iteration_no_activation(
        &mut self,
        inputs: &[f64],
    ) -> Result<f64, NetworkError> {
        self.base_feedforward_compute(inputs)?;

        let output = self.output_layer.compute_n_to_1_without_activation_layer(
            self.intermediate_values
                .last()
                .ok_or(NetworkError::IntermediateValuesIncomplete)?,
        );

        Ok(output)
    }

    fn base_feedforward_compute(&mut self, inputs: &[f64]) -> Result<(), NetworkError> {
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

        Ok(())
    }

    fn feedforward_compute_batch(
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

    fn backpropagate_error_batch(
        &mut self,
        final_error: f64,
        intermediate_values: &[Vec<f64>],
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

    fn backpropagate_error_iteration(&mut self, final_error: f64) -> Result<(), NetworkError> {
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

    fn step_gradient_iteration(&mut self, inputs: &[f64]) -> Result<(), NetworkError> {
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

    fn step_gradient_batch(
        &mut self,
        intermediate_values: &[Vec<f64>],
    ) -> Result<(), NetworkError> {
        let input = intermediate_values
            .first()
            .ok_or(NetworkError::IntermediateValuesIncomplete)?;
        self.input_layer.step_gradient(input);

        if let NetworkType::MultiLayerPerceptron = self.network_type {
            for i in 0..self.network_depth - 2 {
                let input = intermediate_values
                    .get(i + 1)
                    .ok_or(NetworkError::IntermediateValuesIncomplete)?;
                self.common_layers
                    .get_mut(i)
                    .ok_or(NetworkError::InvalidCommonLayers)?
                    .step_gradient(input);
            }
        }

        let input = intermediate_values
            .last()
            .ok_or(NetworkError::IntermediateValuesIncomplete)?;

        self.output_layer.step_gradient(input);

        self.reset_intermediate_values();

        Ok(())
    }

    fn train_iteration(&mut self, inputs: &[f64], aim: f64) -> Result<(), NetworkError> {
        let final_answer = self.feedforward_compute_iteration_no_activation(inputs)?;
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

    pub fn predict_iteration_no_activation(&mut self, inputs: &[f64]) -> Result<f64, NetworkError> {
        self.feedforward_compute_iteration_no_activation(inputs)
    }

    pub fn predict_batch_no_activation(&mut self, inputs: &[f64]) -> Result<f64, NetworkError> {
        self.feedforward_compute_batch(inputs)
            .map(|(_, return_value)| return_value)
    }

    pub fn iterations_train(
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

    pub fn batch_train(
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
        println!("Inputs: {:?}", inputs);
        let average_intermediate_values = total_intermediate_values
            .iter()
            .map(|item| {
                item.iter()
                    .map(|value| value / inputs.len() as f64)
                    .collect()
            })
            .collect::<Vec<Vec<f64>>>();
        let average_error = total_error / inputs.len() as f64;
        println!("Network error: {:.2?}", average_error);
        self.backpropagate_error_batch(average_error, &average_intermediate_values)?;
        println!("Pos backprogation: {}", self);
        self.step_gradient_batch(&average_intermediate_values)?;
        println!("Pos gradiente: {}", self);
        Ok(())
    }

    pub fn reset_intermediate_values(&mut self) {
        self.intermediate_values = vec![];
    }
}

network_display!(PipelineNetwork);
