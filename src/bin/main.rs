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

    let network_depth: usize = 2;
    let input_width: usize = inputs.first().map(|list| list.len()).unwrap_or(0);
    let network_width: usize = 1000_usize;

    println!("Profundidade da rede: {}", network_depth);
    println!("Largura da rede: {}", network_width);

    let mut new_network = Network::new(
        network_depth,
        network_width,
        input_width,
        0.001,
        |value| match value > 0.0 {
            true => value,
            false => 0.0,
        },
        |value| match value > 0.0 {
            true => 1.0,
            false => 0.0,
        },
    );

    //println!("{}", new_network);
    for _ in 0..1000 {
        for (i, input) in inputs.iter().enumerate() {
            let training_result =
                new_network.batch_train_one_iteration(input, *outputs.get(i).unwrap());
            //println!("{}", new_network);
            match training_result {
                Ok(_) => (),
                Err(error) => panic!("{:?}", error),
            }
        }
    }
    let test_return = new_network.feedforward_compute(&[1.0, 5.09]);
    println!("Retorno de 1 - 5.09: {}", test_return.unwrap());
}
