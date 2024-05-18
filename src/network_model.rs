use crate::layer::Layer;
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
    #[error("Acessed and empty neuron list.")]
    EmptyNeuronList,
    #[error("Incorrect definition of network width list.")]
    IncorrectNetworkWidthList,
    #[error("Input length is incompatible with network definition.")]
    InvalidInputInserted,
}
pub enum NetworkType {
    MultiLayerPerceptron,
    TwoLayerPerceptron,
    SingleNeuron,
}

pub fn generate_layers_for_single_neuron_model(
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

pub fn generate_layers_for_two_layer_perceptron(
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

pub fn generate_layers_for_mlp(
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

#[macro_export]
macro_rules! new_network_function {
    ($network_type:ident) => {
        pub fn new(
            network_depth: usize,
            network_width: &[usize],
            input_width: usize,
            learning_rate: f64,
            activation_function: $crate::functions::activation_functions::ActivationFunctionType,
            activation_function_prime: $crate::functions::activation_functions::ActivationFunctionType,
            error_function: $crate::functions::error_functions::ErrorFunctionType,
        ) -> Result<$network_type, NetworkError> {
            let network_type = match network_depth {
                i if i == 1 => NetworkType::SingleNeuron,
                i if i == 2 => NetworkType::TwoLayerPerceptron,
                _ => NetworkType::MultiLayerPerceptron,
            };

            let (input_layer, output_layer, common_layers) = match network_type {
                NetworkType::MultiLayerPerceptron => $crate::network_model::generate_layers_for_mlp(
                    network_width,
                    input_width,
                    learning_rate,
                    activation_function,
                    activation_function_prime,
                    network_depth,
                )?,
                NetworkType::TwoLayerPerceptron => $crate::network_model::generate_layers_for_two_layer_perceptron(
                    network_width,
                    input_width,
                    learning_rate,
                    activation_function,
                    activation_function_prime,
                )?,
                NetworkType::SingleNeuron => $crate::network_model::generate_layers_for_single_neuron_model(
                    input_width,
                    learning_rate,
                    activation_function,
                    activation_function_prime,
                ),
            };

            let intermidiate_values: Vec<Vec<f64>> = Vec::new();

            let network_width = network_width.to_vec();

            let network = $network_type {
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
    };
}

#[macro_export]
macro_rules! network_display {
    ($network_type:ident) => {
        impl Display for $network_type {
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
    };
}
