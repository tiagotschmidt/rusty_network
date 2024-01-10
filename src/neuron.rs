use rand::Rng;

pub type ActivationFunction = fn(f64) -> f64;

pub struct Neuron {
    weights: Vec<f64>,
    pub bias: f64,
    activation_function: ActivationFunction,
}

impl Neuron {
    pub fn new(
        number_of_weights: usize,
        attribute_activation_function: ActivationFunction,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let mut weights = Vec::new();

        for _ in 0..number_of_weights {
            let new_weight = rng.gen::<f64>() * (rng.gen::<i64>() as f64);
            weights.push(new_weight);
        }

        let bias = rng.gen::<f64>() * (rng.gen::<i64>() as f64);

        Neuron {
            weights,
            bias,
            activation_function: attribute_activation_function,
        }
    }

    pub fn compute(&self, inputs: Vec<f64>) -> f64 {
        (self.activation_function)(
            self.weights
                .iter()
                .zip(inputs.iter())
                .fold(self.bias, |acc, (weight, input)| acc + (weight * input)),
        )
    }

    pub fn get_bias(&self) -> f64 {
        self.bias
    }
}
