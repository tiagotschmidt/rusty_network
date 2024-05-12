use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use rusty_network::{
    activation_functions::{identity, identity_prime},
    error_functions::squared_loss_prime,
    network::Network,
};

fn main() {
    let input_file = File::open("input.txt").unwrap();
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

    for _ in 0..200 {
        println!("############################################");
        let training_result = new_network.train_by_iterations(&inputs, &outputs);
        //println!("{}", new_network);
        match training_result {
            Ok(_) => (),
            Err(error) => panic!("{:?}", error),
        }
        println!("############################################");
    }

    //println!("{}", new_network);

    let test_return = new_network.feedforward_compute_batch(&[1.0, 5.09]);
    println!("Retorno de 1 - 5.09: {}", test_return.unwrap().1);
    let test_return = new_network.feedforward_compute_batch(&[2.0, 5.09]);
    println!("Retorno de 2 - 10.18: {}", test_return.unwrap().1);
    let test_return = new_network.feedforward_compute_batch(&[3.0, 5.09]);
    println!("Retorno de 3 - 15.27: {}", test_return.unwrap().1);
    let test_return = new_network.feedforward_compute_batch(&[4.0, 5.09]);
    println!("Retorno de 4 - 20.36: {}", test_return.unwrap().1);
    let test_return = new_network.feedforward_compute_batch(&[5.0, 5.09]);
    println!("Retorno de 5 - 25.45: {}", test_return.unwrap().1);
}
