use methods::{METHOD_ELF, METHOD_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use serde::{Deserialize, Serialize};
use std::{env, fs, time::Instant};

#[derive(Debug, Deserialize, Serialize)]
struct AdamTraceInput {
    initial_weight: f64,
    gradients: Vec<f64>,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

#[derive(Debug, Deserialize)]
struct TraceFile {
    parameter: String,
    client_id: usize,
    initial_weight: f64,
    gradients: Vec<f64>,
    pytorch_final_weight: f64,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    steps: usize,
    loss: f64,
}

fn main() {
    let trace_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "adam_trace.json".to_string());

    let json = fs::read_to_string(&trace_path)
        .unwrap_or_else(|_| panic!("failed to read {}", trace_path));

    let trace: TraceFile =
        serde_json::from_str(&json).expect("invalid adam_trace.json");

    let input = AdamTraceInput {
        initial_weight: trace.initial_weight,
        gradients: trace.gradients.clone(),
        learning_rate: trace.learning_rate,
        beta1: trace.beta1,
        beta2: trace.beta2,
        epsilon: trace.epsilon,
    };

    println!("Parameter: {}", trace.parameter);
    println!("Client ID: {}", trace.client_id);
    println!("Steps: {}", trace.steps);
    println!("Loss: {:.8}", trace.loss);
    println!("Initial weight: {:.10}", trace.initial_weight);
    println!("PyTorch final weight: {:.10}", trace.pytorch_final_weight);

    let env = ExecutorEnv::builder()
        .write(&input)
        .unwrap()
        .build()
        .unwrap();

    let prover = default_prover();

    let prove_start = Instant::now();

    let prove_info = prover
        .prove(env, METHOD_ELF)
        .expect("proof generation failed");

    let proving_time = prove_start.elapsed();

    let receipt = prove_info.receipt;

    let verify_start = Instant::now();

    receipt
        .verify(METHOD_ID)
        .expect("receipt verification failed");

    let verification_time = verify_start.elapsed();

    let proved_final_weight: f64 =
        receipt.journal.decode().expect("journal decode failed");

    let abs_error =
        (proved_final_weight - trace.pytorch_final_weight).abs();

    println!("RISC Zero final weight: {:.10}", proved_final_weight);
    println!("Absolute error: {:.12}", abs_error);

    let tolerance = 1e-6;

    let honest_submitted_weight = trace.pytorch_final_weight;
    let honest_accept =
        (honest_submitted_weight - proved_final_weight).abs() < tolerance;

    let sign_flip_submitted_weight =
        -trace.pytorch_final_weight;

    let sign_flip_accept =
        (sign_flip_submitted_weight - proved_final_weight).abs() < tolerance;

    println!(
        "Honest submission: {:.10} -> {}",
        honest_submitted_weight,
        if honest_accept { "ACCEPT" } else { "REJECT" }
    );

    println!(
        "Sign Flip submission: {:.10} -> {}",
        sign_flip_submitted_weight,
        if sign_flip_accept { "ACCEPT" } else { "REJECT" }
    );

    println!("Proving time: {:.3?}", proving_time);
    println!("Verification time: {:.3?}", verification_time);
    println!("Receipt verification: SUCCESS");

    println!(
        "ZK_RESULT proved_final_weight={:.10} proving_time_ms={} verification_time_ms={}",
        proved_final_weight,
        proving_time.as_millis(),
        verification_time.as_millis(),
    );
}
