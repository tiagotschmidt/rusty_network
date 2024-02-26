#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::{layer::Layer, network::Network, neuron::Neuron};

    #[test]
    fn create_neuron() {
        let mut rng = rand::thread_rng();
        let random_integer: usize = rng.gen::<usize>() % 1000_usize;

        //println!("Current random integer:{}", random_integer);

        let neuron = Neuron::new(
            random_integer,
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

        let mut input_vec = Vec::with_capacity(random_integer);
        for _ in 0..random_integer {
            input_vec.push(0.0)
        }

        assert_eq!(neuron.get_bias(), neuron.compute(&input_vec));
        //println!("Neuron: {}", neuron);
    }

    #[test]
    fn create_layer() {
        let mut rng = rand::thread_rng();
        let random_integer: usize = rng.gen::<usize>() % 1000_usize;

        //println!("Current random integer:{}", random_integer);

        let layer = Layer::new(
            random_integer,
            random_integer,
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

        let mut input_vec = Vec::with_capacity(random_integer);
        for _ in 0..random_integer {
            input_vec.push(0.0)
        }

        assert_eq!(layer.accumulate_bias(), layer.compute_n_to_1(&input_vec));
        //println!("Neuron: {}", neuron);
    }

    #[test]
    fn create_network() {
        let mut rng = rand::thread_rng();

        let first_random_integer: usize = rng.gen::<usize>() % 10_usize;
        let second_random_integer: usize = rng.gen::<usize>() % 10_usize;

        println!("Profundidade da rede: {}", first_random_integer);
        println!("Largura da rede: {}", second_random_integer);

        let new_network = Network::new(
            first_random_integer,
            second_random_integer,
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
        );

        let mut input_vec = Vec::with_capacity(second_random_integer);
        for _ in 0..second_random_integer {
            input_vec.push(0.0)
        }

        //println!("Input {:#?}", input_vec);
        let optional_value = new_network.compute_input(input_vec);
        //println!("Output {:#?}", optional_value);

        assert!(optional_value.is_some())
    }
}
