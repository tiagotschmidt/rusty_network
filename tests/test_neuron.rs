use rand::Rng;
use rusty_network::{
    functions::activation_functions::{relu, relu_prime},
    neuron::Neuron,
};

#[test]
fn test_zero_input_vec() {
    let mut rng = rand::thread_rng();
    let random_integer: usize = rng.gen::<usize>() % 1000_usize;

    let neuron = Neuron::new(random_integer, 0.01, relu, relu_prime);

    let mut input_vec = Vec::with_capacity(random_integer);
    for _ in 0..random_integer {
        input_vec.push(0.0)
    }

    assert_eq!(neuron.get_bias(), neuron.compute(&input_vec));
}
