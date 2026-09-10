use risc0_zkvm::guest::env;

#[derive(serde::Deserialize)]
struct AdamTraceInput {
    initial_weight: f64,
    gradients: Vec<f64>,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

fn main() {
    let input: AdamTraceInput = env::read();

    let mut weight = input.initial_weight;
    let mut m = 0.0_f64;
    let mut v = 0.0_f64;

    for (index, gradient) in input.gradients.iter().enumerate() {
        let t = (index + 1) as i32;

        m = input.beta1 * m
            + (1.0 - input.beta1) * gradient;

        v = input.beta2 * v
            + (1.0 - input.beta2) * gradient * gradient;

        let m_hat =
            m / (1.0 - input.beta1.powi(t));

        let v_hat =
            v / (1.0 - input.beta2.powi(t));

        weight -= input.learning_rate
            * m_hat
            / (v_hat.sqrt() + input.epsilon);
    }

    // proofに最終監視パラメータをbinding
    env::commit(&weight);
}
