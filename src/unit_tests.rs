#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::{layer::Layer, neuron::Neuron};

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
}
