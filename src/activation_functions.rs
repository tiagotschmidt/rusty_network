pub fn relu(value: f64) -> f64 {
    match value > 0.0 {
        true => value,
        false => 0.0,
    }
}

pub fn relu_prime(value: f64) -> f64 {
    match value > 0.0 {
        true => 1.0,
        false => 0.0,
    }
}
pub fn identity(value: f64) -> f64 {
    value
}

pub fn identity_prime(_value: f64) -> f64 {
    1.0
}

pub fn sigmoid(input: f64) -> f64 {
    // Returns sigmoid of the input
    1.0 / (1.0 + (-input).exp())
}

pub fn sigmoid_prime(input: f64) -> f64 {
    sigmoid(input) * (1.0 - sigmoid(input))
}
