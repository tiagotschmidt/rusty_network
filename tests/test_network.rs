use std::{
    convert::identity,
    fs::File,
    io::{BufRead, BufReader},
};

use rand::Rng;
use rusty_network::{
    activation_functions::{identity_prime, relu, relu_prime},
    error_functions::squared_loss_prime,
    network::Network,
};

#[test]
fn test_zero_input_network() {
    let mut rng = rand::thread_rng();

    let first_random_integer: usize = rng.gen::<usize>() % 10_usize + 1_usize;
    let second_random_integer: usize = rng.gen::<usize>() % 10_usize + 1_usize;

    let mut network_width_vec = vec![];

    for _ in 0..first_random_integer {
        let current_random_integer = rng.gen::<usize>() % 10_usize + 1_usize;
        network_width_vec.push(current_random_integer);
    }

    println!("Profundidade da rede: {}", first_random_integer);
    println!("Largura da rede: {:?}", network_width_vec);

    let mut new_network = match Network::new(
        first_random_integer,
        &network_width_vec,
        second_random_integer,
        0.01,
        relu,
        relu_prime,
        squared_loss_prime,
    ) {
        Ok(item) => item,
        Err(error) => panic!("{:#?}", error),
    };

    let mut input_vec = Vec::with_capacity(second_random_integer);
    for _ in 0..second_random_integer {
        input_vec.push(0.0)
    }

    let optional_value = new_network.feedforward_compute_iteration(&input_vec);
    assert!(optional_value.is_ok())
}

#[test]
fn test_simple_dollar_architecture() {
    let input_file = File::open("tests/test_inputs/simple_dollar.txt").unwrap();
    let reader = BufReader::new(input_file);

    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut outputs: Vec<f64> = Vec::new();

    for line in reader.lines() {
        let line_str = line.unwrap();
        let values = line_str.split_whitespace();

        let mut input_data_line = vec![];

        for (index, value) in values.enumerate() {
            if index < 2 {
                input_data_line.push(value.parse::<f64>().unwrap());
            } else {
                outputs.push(value.parse::<f64>().unwrap());
            }
        }
        inputs.push(input_data_line);
    }

    //println!("{:?}", inputs);
    //println!("{:?}", outputs);

    let network_depth: usize = 1;
    let input_width: usize = inputs.first().map(|list| list.len()).unwrap_or(0);
    let network_width = vec![1];

    println!("Profundidade da rede: {}", network_depth);
    println!("Largura da rede: {:?}", network_width);

    let learning_rate = 0.01;
    let mut new_network = Network::new(
        network_depth,
        &network_width,
        input_width,
        learning_rate,
        identity,
        identity_prime,
        squared_loss_prime,
    )
    .unwrap();

    for _ in 0..100 {
        let training_result = new_network.train_by_iterations(&inputs, &outputs);
        match training_result {
            Ok(_) => (),
            Err(error) => panic!("{:?}", error),
        }
    }

    let test_return = new_network.feedforward_compute_batch(&[5.0, 5.09]);
    let final_result = test_return.unwrap().1;
    assert!(final_result < 26.0 && final_result > 24.0);
}
