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
