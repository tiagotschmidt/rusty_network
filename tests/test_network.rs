use rand::Rng;
use rusty_network::network::Network;

#[test]
fn create_network() {
    let mut rng = rand::thread_rng();

    let first_random_integer: usize = rng.gen::<usize>() % 10_usize + 1_usize;
    let second_random_integer: usize = rng.gen::<usize>() % 10_usize + 1_usize;

    let mut network_width_vec = vec![];

    for _ in 0..second_random_integer {
        let current_random_integer = rng.gen::<usize>() % 10_usize + 1_usize;
        network_width_vec.push(current_random_integer);
    }

    println!("Profundidade da rede: {}", first_random_integer);
    println!("Largura da rede: {}", second_random_integer);

    let mut new_network = match Network::new(
        first_random_integer,
        &network_width_vec,
        second_random_integer,
        0.01,
        |value| match value > 0.0 {
            true => value,
            false => 0.0,
        },
        |value| match value > 0.0 {
            true => 1.0,
            false => 0.0,
        },
        |aim, final_answer| -2.0 * (aim - final_answer),
    ) {
        Ok(item) => item,
        Err(error) => panic!("{}", error),
    };

    let mut input_vec = Vec::with_capacity(second_random_integer);
    for _ in 0..second_random_integer {
        input_vec.push(0.0)
    }

    //println!("{}", new_network);
    let optional_value = new_network.feedforward_compute(&input_vec);
    assert!(optional_value.is_ok())
}
