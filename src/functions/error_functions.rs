pub type ErrorFunction = fn(f64, f64) -> f64;
pub fn squared_loss_prime(aim: f64, final_answer: f64) -> f64 {
    -2.0 * (aim - final_answer)
}
