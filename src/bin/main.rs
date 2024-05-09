use core::f64;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use rusty_network::network::Network;

fn main() {
    let input_file = File::open("input.txt").unwrap();
    let reader = BufReader::new(input_file);

    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut outputs: Vec<f64> = Vec::new();

    for line in reader.lines() {
        let line_str = line.unwrap();
        let values = line_str.split_whitespace();

        let mut input_data_line = vec![];
        let mut index = 0;

        for value in values {
            if index < 2 {
                input_data_line.push(value.parse::<f64>().unwrap());
            } else {
                outputs.push(value.parse::<f64>().unwrap());
            }
            index += 1;
        }
        inputs.push(input_data_line);
    }

    println!("{:?}", inputs);
    println!("{:?}", outputs);

    let first_random_integer: usize = 2;
    let network_width: usize = 2_usize;

    println!("Profundidade da rede: {}", first_random_integer);
    println!("Largura da rede: {}", network_width);

    let mut new_network = Network::new(
        first_random_integer,
        network_width,
        network_width,
        0.01,
        |value| match value > 0.0 {
            true => value,
            false => 0.0,
        },
        |value| match value > 0.0 {
            true => 1.0,
            false => 0.0,
        },
    );

    println!("{}", new_network);
    for input in inputs {
        let optional_value = new_network.feedforward_compute(input);
        println!("{:#?}", optional_value);
    }
}
