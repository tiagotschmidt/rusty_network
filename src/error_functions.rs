pub fn squared_loss_prime(aim: f64, final_answer: f64) -> f64 {
    2.0 * (final_answer - aim)
}
