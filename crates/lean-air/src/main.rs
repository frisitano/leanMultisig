use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use backend::{BasedVectorSpace, EFPacking, FoldingFactor, PFPacking, WhirConfigBuilder, packing_log_width};
use clap::{Args as ClapArgs, Parser, Subcommand};
use lean_air::{
    DEFAULT_CELL_LEN_EXT, LeanAirDedicatedHashTable, LeanAirShape, deterministic_codewords,
    prove_lean_air_table_with_config, verify_lean_air_with_config,
};
use lean_prover::default_whir_config;
use lean_vm::{EF, F};

const F_BITS: usize = 31;
const LEAN_AIR_MAX_NUM_VARIABLES_TO_SEND_COEFFS: usize = 12;
const LEAN_AIR_INITIAL_FOLDING_FACTOR: usize = 8;
const LEAN_AIR_SUBSEQUENT_FOLDING_FACTOR: usize = 5;
const LEAN_AIR_PARALLEL_BENCH_CELL_LEN_EXT: usize = 128;

#[derive(Debug, Parser)]
struct Args {
    #[command(subcommand)]
    command: Option<CliCommand>,
    #[arg(long, default_value_t = 10)]
    log_m: usize,
    #[arg(long, default_value_t = 10)]
    n_rows: usize,
    #[arg(long, default_value_t = DEFAULT_CELL_LEN_EXT)]
    cell_len_ext: usize,
    #[arg(long)]
    prove: bool,
    #[arg(long)]
    tracing: bool,
    #[arg(long, default_value_t = 1)]
    runs: usize,
    #[arg(long)]
    csv: bool,
    #[arg(long, conflicts_with = "csv")]
    table: bool,
    #[arg(long)]
    cpu: bool,
    #[arg(long)]
    bandwidth: bool,
    #[arg(long, help = "Use the Metal GPU backend for WHIR Merkle commitments on macOS")]
    gpu: bool,
    #[arg(long, value_name = "SVG_PATH")]
    cpu_graph: Option<PathBuf>,
    #[arg(long, default_value_t = 50)]
    cpu_sample_ms: u64,
    #[arg(long, default_value_t = LEAN_AIR_MAX_NUM_VARIABLES_TO_SEND_COEFFS)]
    max_num_variables_to_send_coeffs: usize,
    #[arg(long, default_value_t = LEAN_AIR_INITIAL_FOLDING_FACTOR)]
    initial_folding_factor: usize,
    #[arg(long, default_value_t = LEAN_AIR_SUBSEQUENT_FOLDING_FACTOR)]
    subsequent_folding_factor: usize,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    BenchParallel(BenchParallelArgs),
}

#[derive(Debug, Clone, ClapArgs)]
struct BenchParallelArgs {
    #[arg(long, default_value_t = 13)]
    log_m: usize,
    #[arg(long, default_value_t = 101)]
    n_rows: usize,
    #[arg(long, default_value_t = LEAN_AIR_PARALLEL_BENCH_CELL_LEN_EXT)]
    cell_len_ext: usize,
    #[arg(long)]
    concurrency: usize,
    #[arg(long)]
    rayon_threads: usize,
    #[arg(long, default_value_t = 1)]
    runs_per_worker: usize,
    #[arg(long)]
    pin: bool,
    #[arg(long)]
    cpu: bool,
    #[arg(long, help = "Use the Metal GPU backend in each worker on macOS")]
    gpu: bool,
    #[arg(long, default_value = "/tmp/lean-air-parallel-runs.csv")]
    out: PathBuf,
    #[arg(long, default_value = "/tmp/lean-air-parallel-summary.csv")]
    summary_out: PathBuf,
    #[arg(long, value_name = "PATH")]
    child_binary: Option<PathBuf>,
    #[arg(long, default_value_t = LEAN_AIR_MAX_NUM_VARIABLES_TO_SEND_COEFFS)]
    max_num_variables_to_send_coeffs: usize,
    #[arg(long, default_value_t = LEAN_AIR_INITIAL_FOLDING_FACTOR)]
    initial_folding_factor: usize,
    #[arg(long, default_value_t = LEAN_AIR_SUBSEQUENT_FOLDING_FACTOR)]
    subsequent_folding_factor: usize,
}

#[derive(Clone, Debug)]
struct StageMetrics {
    wall: Duration,
    cpu: Duration,
}

#[derive(Clone, Debug)]
struct RunMetrics {
    generated_codewords: StageMetrics,
    built_table: StageMetrics,
    proved: Option<StageMetrics>,
    verified: Option<StageMetrics>,
    proof_size_fe: Option<usize>,
    stage_timings: Vec<(String, Duration)>,
    chunks_per_cell: usize,
    active_poseidon_rows: usize,
    padded_poseidon_rows: usize,
    committed_poseidon_columns: usize,
    peak_rss_bytes: u64,
    row_commitment_root: [F; 8],
    column_commitment_root: [F; 8],
    commitment_root: [F; 8],
}

fn main() {
    let args = Args::parse();
    if let Some(command) = &args.command {
        match command {
            CliCommand::BenchParallel(bench_args) => run_parallel_benchmark(bench_args),
        }
        return;
    }
    assert!(args.runs > 0, "--runs must be nonzero");
    if args.tracing {
        utils::init_tracing();
    }
    configure_gpu(args.gpu);
    if args.bandwidth && !system_info::cpu_stage_profiling_enabled() {
        eprintln!(
            "--bandwidth requires stage profiling; rerun with `cargo run -p lean-air --features profiling -- ...`"
        );
        std::process::exit(2);
    }
    if args.cpu_graph.is_some() && !system_info::cpu_stage_profiling_enabled() {
        eprintln!(
            "--cpu-graph requires stage profiling; rerun with `cargo run -p lean-air --features profiling -- ...`"
        );
        std::process::exit(2);
    }
    let shape = LeanAirShape::with_cell_len(args.log_m, args.n_rows, args.cell_len_ext);
    let whir_config = whir_config_from_args(&args);
    let measure_cpu = args.cpu || args.cpu_graph.is_some();

    let cpu_profiler = args
        .cpu_graph
        .as_ref()
        .map(|_| CpuProfiler::start(Duration::from_millis(args.cpu_sample_ms.max(1))));
    let mut runs = Vec::with_capacity(args.runs);
    for run_idx in 0..args.runs {
        let run_label = if args.runs == 1 {
            "run".to_string()
        } else {
            format!("run{}", run_idx + 1)
        };
        runs.push(run_once(
            shape,
            args.prove,
            &whir_config,
            cpu_profiler.as_ref(),
            &run_label,
            args.bandwidth,
            measure_cpu,
        ));
    }
    if let (Some(path), Some(profiler)) = (&args.cpu_graph, cpu_profiler) {
        let sample_count = profiler.finish(path).expect("write CPU timeline graph");
        println!(
            "cpu timeline graph: {} ({} samples, raw csv: {})",
            path.display(),
            sample_count,
            cpu_csv_path(path).display()
        );
    }
    print_report(
        shape,
        &runs,
        args.prove,
        args.csv,
        args.table,
        args.cpu,
        args.bandwidth,
        &whir_config,
    );
}

fn configure_gpu(gpu: bool) {
    if !gpu {
        return;
    }
    if gpu_poseidon::metal_available() {
        gpu_poseidon::set_gpu_enabled(true);
        eprintln!("GPU backend: Metal enabled for WHIR Merkle commitments");
    } else {
        eprintln!("GPU backend requested but Metal is unavailable on this platform; falling back to CPU");
    }
}

fn whir_config_from_args(args: &Args) -> WhirConfigBuilder {
    let mut config = default_whir_config(1);
    config.max_num_variables_to_send_coeffs = args.max_num_variables_to_send_coeffs;
    config.folding_factor = FoldingFactor::new(args.initial_folding_factor, args.subsequent_folding_factor);
    config
}

#[derive(Clone, Debug)]
struct ParallelRunRow {
    worker_id: usize,
    child_run: usize,
    concurrency: usize,
    rayon_threads: usize,
    core_range: String,
    data_kib: f64,
    generated_codewords_ms: f64,
    built_table_ms: f64,
    proved_ms: f64,
    verification_ms: f64,
    proof_size_fe: usize,
    proof_size_kib: f64,
    build_throughput_kib_per_s: f64,
    prover_throughput_kib_per_s: f64,
    peak_rss_bytes: u64,
    worker_wall_ms: f64,
    exit_status: i32,
    generated_codewords_cpu_x: Option<f64>,
    built_table_cpu_x: Option<f64>,
    proved_cpu_x: Option<f64>,
    verification_cpu_x: Option<f64>,
}

#[derive(Debug)]
struct RunningWorker {
    worker_id: usize,
    core_range: String,
    started_at: Instant,
    child: std::process::Child,
}

fn run_parallel_benchmark(args: &BenchParallelArgs) {
    assert!(args.concurrency > 0, "--concurrency must be nonzero");
    assert!(args.rayon_threads > 0, "--rayon-threads must be nonzero");
    assert!(args.runs_per_worker > 0, "--runs-per-worker must be nonzero");

    if args.pin && !cfg!(target_os = "linux") {
        eprintln!("--pin requires Linux taskset; rerun without --pin on this platform");
        std::process::exit(2);
    }

    let child_binary = args
        .child_binary
        .clone()
        .unwrap_or_else(|| env::current_exe().expect("current executable path"));
    let parent_started_at = Instant::now();
    let mut workers = Vec::with_capacity(args.concurrency);
    for worker_id in 0..args.concurrency {
        let core_range = if args.pin {
            let start = worker_id * args.rayon_threads;
            let end = start + args.rayon_threads - 1;
            format!("{start}-{end}")
        } else {
            String::new()
        };
        let mut command = benchmark_child_command(args, &child_binary, &core_range);
        let child = command.spawn().expect("spawn lean-air benchmark worker");
        workers.push(RunningWorker {
            worker_id,
            core_range,
            started_at: Instant::now(),
            child,
        });
    }

    let mut rows = Vec::new();
    for worker in workers {
        let output = worker.child.wait_with_output().expect("wait for lean-air worker");
        let worker_wall = worker.started_at.elapsed();
        let exit_status = output.status.code().unwrap_or(-1);
        if !output.status.success() {
            eprintln!("lean-air worker {} failed with status {exit_status}", worker.worker_id);
            eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
            eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
            std::process::exit(output.status.code().unwrap_or(1));
        }
        let stdout = String::from_utf8(output.stdout).expect("worker stdout is UTF-8");
        let parsed_rows = parse_child_csv(&stdout).unwrap_or_else(|err| {
            panic!(
                "parse CSV from worker {} failed: {err}\nstdout:\n{stdout}",
                worker.worker_id
            )
        });
        for mut row in parsed_rows {
            row.worker_id = worker.worker_id;
            row.concurrency = args.concurrency;
            row.rayon_threads = args.rayon_threads;
            row.core_range = worker.core_range.clone();
            row.worker_wall_ms = duration_ms(worker_wall);
            row.exit_status = exit_status;
            rows.push(row);
        }
    }

    let parent_wall = parent_started_at.elapsed();
    write_parallel_run_csv(&args.out, args, &rows).expect("write parallel benchmark run CSV");
    write_parallel_summary_csv(&args.summary_out, args, &rows, parent_wall)
        .expect("write parallel benchmark summary CSV");

    let total_data_kib = rows.iter().map(|row| row.data_kib).sum::<f64>();
    let aggregate_wall_kib_per_s = total_data_kib / parent_wall.as_secs_f64();
    let aggregate_prove_kib_per_s = total_data_kib / max_worker_sum_ms(&rows, |row| row.proved_ms) * 1_000.0;
    println!("lean-air parallel benchmark");
    println!("  child_binary: {}", child_binary.display());
    println!("  workers: {}", args.concurrency);
    println!("  rayon_threads/worker: {}", args.rayon_threads);
    println!("  runs/worker: {}", args.runs_per_worker);
    println!("  parent_wall_ms: {:.2}", duration_ms(parent_wall));
    println!("  aggregate_wall_KiB/s: {:.2}", aggregate_wall_kib_per_s);
    println!("  aggregate_prove_KiB/s: {:.2}", aggregate_prove_kib_per_s);
    println!("  runs_csv: {}", args.out.display());
    println!("  summary_csv: {}", args.summary_out.display());
}

fn benchmark_child_command(args: &BenchParallelArgs, child_binary: &Path, core_range: &str) -> Command {
    let child_args = benchmark_child_args(args);
    let mut command = if args.pin {
        let mut command = Command::new("taskset");
        command.arg("-c").arg(core_range).arg(child_binary);
        command
    } else {
        Command::new(child_binary)
    };
    command
        .args(child_args)
        .env("RAYON_NUM_THREADS", args.rayon_threads.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

fn benchmark_child_args(args: &BenchParallelArgs) -> Vec<String> {
    let mut child_args = vec![
        "--log-m".to_string(),
        args.log_m.to_string(),
        "--n-rows".to_string(),
        args.n_rows.to_string(),
        "--cell-len-ext".to_string(),
        args.cell_len_ext.to_string(),
        "--prove".to_string(),
        "--runs".to_string(),
        args.runs_per_worker.to_string(),
        "--csv".to_string(),
        "--max-num-variables-to-send-coeffs".to_string(),
        args.max_num_variables_to_send_coeffs.to_string(),
        "--initial-folding-factor".to_string(),
        args.initial_folding_factor.to_string(),
        "--subsequent-folding-factor".to_string(),
        args.subsequent_folding_factor.to_string(),
    ];
    if args.cpu {
        child_args.push("--cpu".to_string());
    }
    if args.gpu {
        child_args.push("--gpu".to_string());
    }
    child_args
}

fn parse_child_csv(stdout: &str) -> Result<Vec<ParallelRunRow>, String> {
    let mut lines = stdout.lines().filter(|line| !line.trim().is_empty());
    let header = lines.next().ok_or_else(|| "missing CSV header".to_string())?;
    let headers = header.split(',').collect::<Vec<_>>();
    let header_indexes = headers
        .iter()
        .enumerate()
        .map(|(idx, header)| (*header, idx))
        .collect::<BTreeMap<_, _>>();
    let mut rows = Vec::new();
    for line in lines {
        let values = line.split(',').collect::<Vec<_>>();
        if values.len() != headers.len() {
            return Err(format!(
                "row has {} cells but header has {} cells: {line}",
                values.len(),
                headers.len()
            ));
        }
        rows.push(ParallelRunRow {
            worker_id: 0,
            child_run: parse_csv_cell(&header_indexes, &values, "run")?,
            concurrency: 0,
            rayon_threads: 0,
            core_range: String::new(),
            data_kib: parse_csv_cell(&header_indexes, &values, "systematic_data_kib")?,
            generated_codewords_ms: parse_csv_cell(&header_indexes, &values, "generated_codewords_ms")?,
            built_table_ms: parse_csv_cell(&header_indexes, &values, "built_table_ms")?,
            proved_ms: parse_csv_cell(&header_indexes, &values, "proved_ms")?,
            verification_ms: parse_csv_cell(&header_indexes, &values, "verification_ms")?,
            proof_size_fe: parse_csv_cell(&header_indexes, &values, "proof_size_fe")?,
            proof_size_kib: parse_csv_cell(&header_indexes, &values, "proof_size_kib")?,
            build_throughput_kib_per_s: parse_csv_cell(&header_indexes, &values, "build_throughput_kib_per_s")?,
            prover_throughput_kib_per_s: parse_csv_cell(&header_indexes, &values, "prover_throughput_kib_per_s")?,
            peak_rss_bytes: parse_csv_cell(&header_indexes, &values, "peak_rss_bytes")?,
            worker_wall_ms: 0.0,
            exit_status: 0,
            generated_codewords_cpu_x: parse_optional_csv_cell(&header_indexes, &values, "generated_codewords_cpu_x")?,
            built_table_cpu_x: parse_optional_csv_cell(&header_indexes, &values, "built_table_cpu_x")?,
            proved_cpu_x: parse_optional_csv_cell(&header_indexes, &values, "proved_cpu_x")?,
            verification_cpu_x: parse_optional_csv_cell(&header_indexes, &values, "verification_cpu_x")?,
        });
    }
    Ok(rows)
}

fn parse_csv_cell<T: std::str::FromStr>(
    header_indexes: &BTreeMap<&str, usize>,
    values: &[&str],
    name: &str,
) -> Result<T, String> {
    let idx = *header_indexes
        .get(name)
        .ok_or_else(|| format!("missing CSV column {name}"))?;
    values[idx]
        .parse::<T>()
        .map_err(|_| format!("invalid CSV value for {name}: {}", values[idx]))
}

fn parse_optional_csv_cell<T: std::str::FromStr>(
    header_indexes: &BTreeMap<&str, usize>,
    values: &[&str],
    name: &str,
) -> Result<Option<T>, String> {
    let Some(&idx) = header_indexes.get(name) else {
        return Ok(None);
    };
    values[idx]
        .parse::<T>()
        .map(Some)
        .map_err(|_| format!("invalid CSV value for {name}: {}", values[idx]))
}

fn write_parallel_run_csv(path: &Path, args: &BenchParallelArgs, rows: &[ParallelRunRow]) -> std::io::Result<()> {
    let mut out = String::from(
        "worker_id,child_run,concurrency,rayon_threads,core_range,log_m,n_rows,cell_len_ext,data_kib,generated_codewords_ms,built_table_ms,proved_ms,verification_ms,proof_size_fe,proof_size_kib,build_throughput_kib_per_s,prover_throughput_kib_per_s,peak_rss_bytes,worker_wall_ms,exit_status,generated_codewords_cpu_x,built_table_cpu_x,proved_cpu_x,verification_cpu_x\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{},{:.2},{:.2},{:.2},{},{:.2},{},{},{},{},{}\n",
            row.worker_id,
            row.child_run,
            row.concurrency,
            row.rayon_threads,
            row.core_range,
            args.log_m,
            args.n_rows,
            args.cell_len_ext,
            row.data_kib,
            row.generated_codewords_ms,
            row.built_table_ms,
            row.proved_ms,
            row.verification_ms,
            row.proof_size_fe,
            row.proof_size_kib,
            row.build_throughput_kib_per_s,
            row.prover_throughput_kib_per_s,
            row.peak_rss_bytes,
            row.worker_wall_ms,
            row.exit_status,
            optional_f64_csv(row.generated_codewords_cpu_x),
            optional_f64_csv(row.built_table_cpu_x),
            optional_f64_csv(row.proved_cpu_x),
            optional_f64_csv(row.verification_cpu_x),
        ));
    }
    fs::write(path, out)
}

fn write_parallel_summary_csv(
    path: &Path,
    args: &BenchParallelArgs,
    rows: &[ParallelRunRow],
    parent_wall: Duration,
) -> std::io::Result<()> {
    assert!(!rows.is_empty(), "parallel benchmark produced no rows");
    let total_runs = rows.len();
    let total_data_kib = rows.iter().map(|row| row.data_kib).sum::<f64>();
    let parent_wall_ms = duration_ms(parent_wall);
    let aggregate_wall_kib_per_s = total_data_kib / parent_wall.as_secs_f64();
    let aggregate_prove_kib_per_s = total_data_kib / max_worker_sum_ms(rows, |row| row.proved_ms) * 1_000.0;
    let aggregate_build_kib_per_s = total_data_kib / max_worker_sum_ms(rows, |row| row.built_table_ms) * 1_000.0;
    let prove_ms = rows.iter().map(|row| row.proved_ms).collect::<Vec<_>>();
    let build_ms = rows.iter().map(|row| row.built_table_ms).collect::<Vec<_>>();
    let verification_ms = rows.iter().map(|row| row.verification_ms).collect::<Vec<_>>();
    let proof_kib = rows.iter().map(|row| row.proof_size_kib).collect::<Vec<_>>();
    let max_peak_rss_bytes = rows.iter().map(|row| row.peak_rss_bytes).max().unwrap_or(0);
    let summed_worker_peak_rss_bytes = sum_worker_peak_rss_bytes(rows);

    let out = format!(
        "concurrency,rayon_threads,runs_per_worker,total_runs,log_m,n_rows,cell_len_ext,total_data_kib,parent_wall_ms,aggregate_wall_kib_per_s,aggregate_prove_kib_per_s,aggregate_build_kib_per_s,prove_ms_mean,prove_ms_min,prove_ms_max,prove_ms_p50,prove_ms_p95,build_ms_mean,verification_ms_mean,proof_kib_mean,max_peak_rss_bytes,summed_worker_peak_rss_bytes\n{},{},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{}\n",
        args.concurrency,
        args.rayon_threads,
        args.runs_per_worker,
        total_runs,
        args.log_m,
        args.n_rows,
        args.cell_len_ext,
        total_data_kib,
        parent_wall_ms,
        aggregate_wall_kib_per_s,
        aggregate_prove_kib_per_s,
        aggregate_build_kib_per_s,
        mean(&prove_ms),
        min(&prove_ms),
        max(&prove_ms),
        percentile(&prove_ms, 0.50),
        percentile(&prove_ms, 0.95),
        mean(&build_ms),
        mean(&verification_ms),
        mean(&proof_kib),
        max_peak_rss_bytes,
        summed_worker_peak_rss_bytes,
    );
    fs::write(path, out)
}

fn optional_f64_csv(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.2}"))
}

fn max_worker_sum_ms(rows: &[ParallelRunRow], value: impl Fn(&ParallelRunRow) -> f64) -> f64 {
    let mut by_worker = BTreeMap::<usize, f64>::new();
    for row in rows {
        *by_worker.entry(row.worker_id).or_default() += value(row);
    }
    by_worker.into_values().fold(0.0, f64::max)
}

fn sum_worker_peak_rss_bytes(rows: &[ParallelRunRow]) -> u64 {
    let mut by_worker = BTreeMap::<usize, u64>::new();
    for row in rows {
        let entry = by_worker.entry(row.worker_id).or_default();
        *entry = (*entry).max(row.peak_rss_bytes);
    }
    by_worker.into_values().sum()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn min(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let idx = ((sorted.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

fn run_once(
    shape: LeanAirShape,
    prove: bool,
    whir_config: &WhirConfigBuilder,
    cpu_profiler: Option<&CpuProfiler>,
    run_label: &str,
    bandwidth: bool,
    measure_cpu: bool,
) -> RunMetrics {
    if bandwidth {
        system_info::reset_cpu_stage_timings();
    }

    let (codewords, generated_codewords) =
        measure_stage(cpu_profiler, measure_cpu, format!("{run_label}/generate"), || {
            deterministic_codewords(shape)
        });

    let (table, built_table) = measure_stage(cpu_profiler, measure_cpu, format!("{run_label}/build"), || {
        LeanAirDedicatedHashTable::build(shape, &codewords)
    });
    let chunks_per_cell = table.schedule.chunks_per_cell;
    let active_poseidon_rows = table.active_rows();
    let padded_poseidon_rows = table.padded_rows();
    let committed_poseidon_columns = table.committed_columns().len();
    let row_commitment_root = table.commitments.row_commitment_root;
    let column_commitment_root = table.commitments.column_commitment_root;
    let commitment_root = table.commitments.commitment_root;

    let mut proved = None;
    let mut verified = None;
    let mut proof_size_fe = None;

    if prove {
        let _ = measure_stage(
            cpu_profiler,
            measure_cpu,
            format!("{run_label}/precompute_twiddles"),
            || {
                backend::precompute_dft_twiddles::<F>(2 * table.padded_rows());
            },
        );
        let (proof, prove_duration) = measure_stage(cpu_profiler, measure_cpu, format!("{run_label}/prove"), || {
            prove_lean_air_table_with_config(table, whir_config)
        });
        let metadata = proof.metadata.clone();

        let (_, verify_duration) = measure_stage(cpu_profiler, measure_cpu, format!("{run_label}/verify"), || {
            verify_lean_air_with_config(shape, metadata.commitment_root, proof.proof, whir_config)
                .expect("lean-air proof verifies")
        });

        proved = Some(prove_duration);
        verified = Some(verify_duration);
        proof_size_fe = Some(metadata.proof_size_fe);
    }

    let stage_timings = if bandwidth {
        system_info::take_cpu_stage_timings()
    } else {
        Vec::new()
    };

    RunMetrics {
        generated_codewords,
        built_table,
        proved,
        verified,
        proof_size_fe,
        stage_timings,
        chunks_per_cell,
        active_poseidon_rows,
        padded_poseidon_rows,
        committed_poseidon_columns,
        peak_rss_bytes: system_info::peak_rss_bytes(),
        row_commitment_root,
        column_commitment_root,
        commitment_root,
    }
}

fn measure_stage<T>(
    cpu_profiler: Option<&CpuProfiler>,
    measure_cpu: bool,
    stage: impl Into<String>,
    f: impl FnOnce() -> T,
) -> (T, StageMetrics) {
    let _stage = cpu_profiler.map(|_| system_info::enter_cpu_stage(stage.into()));
    let cpu_start = measure_cpu.then(system_info::process_cpu_time);
    let wall_start = Instant::now();
    let value = f();
    let wall = wall_start.elapsed();
    let cpu = cpu_start
        .and_then(|start| system_info::process_cpu_time().checked_sub(start))
        .unwrap_or(Duration::ZERO);
    (value, StageMetrics { wall, cpu })
}

#[derive(Clone, Debug)]
struct CpuTimelineSample {
    elapsed: Duration,
    cpu_x: f64,
    stage: String,
}

#[derive(Debug)]
struct CpuProfilerState {
    start: Instant,
    stop: AtomicBool,
    samples: Mutex<Vec<CpuTimelineSample>>,
}

#[derive(Debug)]
struct CpuProfiler {
    state: Arc<CpuProfilerState>,
    handle: Option<JoinHandle<()>>,
}

impl CpuProfiler {
    fn start(interval: Duration) -> Self {
        let state = Arc::new(CpuProfilerState {
            start: Instant::now(),
            stop: AtomicBool::new(false),
            samples: Mutex::new(Vec::new()),
        });
        let thread_state = Arc::clone(&state);
        let handle = thread::spawn(move || {
            let mut last_wall = Instant::now();
            let mut last_cpu = system_info::process_cpu_time();
            while !thread_state.stop.load(Ordering::Relaxed) {
                thread::sleep(interval);
                let now = Instant::now();
                let cpu_now = system_info::process_cpu_time();
                let wall_delta = now.saturating_duration_since(last_wall);
                let cpu_delta = cpu_now.checked_sub(last_cpu).unwrap_or(Duration::ZERO);
                last_wall = now;
                last_cpu = cpu_now;

                if wall_delta.is_zero() {
                    continue;
                }

                let stage = system_info::current_cpu_stage();
                thread_state.samples.lock().unwrap().push(CpuTimelineSample {
                    elapsed: now.saturating_duration_since(thread_state.start),
                    cpu_x: cpu_delta.as_secs_f64() / wall_delta.as_secs_f64(),
                    stage,
                });
            }
        });
        Self {
            state,
            handle: Some(handle),
        }
    }

    fn finish(mut self, path: &Path) -> std::io::Result<usize> {
        self.state.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().expect("CPU profiler thread joins");
        }
        let samples = self.state.samples.lock().unwrap().clone();
        write_cpu_timeline_csv(&cpu_csv_path(path), &samples)?;
        write_cpu_timeline_svg(path, &samples)?;
        Ok(samples.len())
    }
}

fn cpu_csv_path(path: &Path) -> PathBuf {
    let mut csv = path.to_path_buf();
    csv.set_extension("csv");
    csv
}

fn write_cpu_timeline_csv(path: &Path, samples: &[CpuTimelineSample]) -> std::io::Result<()> {
    let mut out = String::from("elapsed_ms,cpu_x,stage\n");
    for sample in samples {
        out.push_str(&format!(
            "{:.3},{:.6},{}\n",
            duration_ms(sample.elapsed),
            sample.cpu_x,
            sample.stage
        ));
    }
    fs::write(path, out)
}

fn write_cpu_timeline_svg(path: &Path, samples: &[CpuTimelineSample]) -> std::io::Result<()> {
    const WIDTH: f64 = 1400.0;
    const HEIGHT: f64 = 720.0;
    const LEFT: f64 = 76.0;
    const RIGHT: f64 = 26.0;
    const TOP: f64 = 44.0;
    const BOTTOM: f64 = 88.0;

    let plot_w = WIDTH - LEFT - RIGHT;
    let plot_h = HEIGHT - TOP - BOTTOM;
    let total_s = samples
        .last()
        .map(|sample| sample.elapsed.as_secs_f64())
        .unwrap_or(1.0)
        .max(0.001);
    let max_cpu = samples.iter().map(|sample| sample.cpu_x).fold(1.0_f64, f64::max);
    let core_hint = thread::available_parallelism().map(|n| n.get() as f64).unwrap_or(1.0);
    let y_max = (max_cpu.max(core_hint) * 1.10).ceil().max(1.0);

    let x = |elapsed: Duration| LEFT + elapsed.as_secs_f64() / total_s * plot_w;
    let y = |cpu_x: f64| TOP + plot_h - (cpu_x / y_max).min(1.0) * plot_h;

    let mut out = String::new();
    out.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<rect width="100%" height="100%" fill="#fbfbf8"/>
<text x="{LEFT}" y="26" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="20" font-weight="700" fill="#161616">LeanAIR CPU Saturation Timeline</text>
<text x="{LEFT}" y="688" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="#525252">cpu_x = process user+system CPU time / wall time over each sample interval</text>
"##
    ));

    let segments = cpu_stage_segments(samples);
    for (idx, segment) in segments.iter().enumerate() {
        let x0 = x(segment.0);
        let x1 = x(segment.1).max(x0 + 1.0);
        out.push_str(&format!(
            r##"<rect x="{:.2}" y="{TOP}" width="{:.2}" height="{plot_h}" fill="{}" opacity="0.18"/>"##,
            x0,
            x1 - x0,
            stage_color(&segment.2),
        ));
        let is_compact_verify = stage_phase(&segment.2) == "verify" && x1 - x0 > 34.0;
        if x1 - x0 > 86.0 || is_compact_verify {
            let label_y = TOP + 17.0 + (idx % 2) as f64 * 16.0;
            out.push_str(&format!(
                r##"<text x="{:.2}" y="{:.2}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" fill="#383838">{}</text>"##,
                x0 + 6.0,
                label_y,
                xml_escape(&short_stage_label(&segment.2)),
            ));
        }
    }
    for window in segments.windows(2) {
        let previous = &window[0];
        let next = &window[1];
        let xx = x(next.0);
        let phase_boundary = stage_phase(&previous.2) != stage_phase(&next.2);
        let (stroke, width, dash, opacity) = if phase_boundary {
            ("#2b2b2b", 1.8, "", 0.78)
        } else {
            ("#5f5f5a", 1.0, r#" stroke-dasharray="5 5""#, 0.42)
        };
        out.push_str(&format!(
            r##"<line x1="{xx:.2}" y1="{TOP}" x2="{xx:.2}" y2="{:.2}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"{dash}/>"##,
            TOP + plot_h
        ));
    }

    let y_grid_count = y_max.min(16.0) as usize;
    for idx in 0..=y_grid_count {
        let value = y_max * idx as f64 / y_grid_count.max(1) as f64;
        let yy = y(value);
        out.push_str(&format!(
            r##"<line x1="{LEFT}" y1="{yy:.2}" x2="{:.2}" y2="{yy:.2}" stroke="#deded8" stroke-width="1"/>"##,
            LEFT + plot_w
        ));
        out.push_str(&format!(
            r##"<text x="{:.2}" y="{:.2}" text-anchor="end" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" fill="#555">{value:.1}</text>"##,
            LEFT - 10.0,
            yy + 4.0
        ));
    }

    let x_grid_count = 10usize;
    for idx in 0..=x_grid_count {
        let secs = total_s * idx as f64 / x_grid_count as f64;
        let xx = LEFT + secs / total_s * plot_w;
        out.push_str(&format!(
            r##"<line x1="{xx:.2}" y1="{TOP}" x2="{xx:.2}" y2="{:.2}" stroke="#ecece6" stroke-width="1"/>"##,
            TOP + plot_h
        ));
        out.push_str(&format!(
            r##"<text x="{xx:.2}" y="{:.2}" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" fill="#555">{secs:.1}s</text>"##,
            TOP + plot_h + 21.0
        ));
    }

    out.push_str(&format!(
        r##"<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{:.2}" stroke="#1f1f1f" stroke-width="1.2"/>
<line x1="{LEFT}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="#1f1f1f" stroke-width="1.2"/>
"##,
        TOP + plot_h,
        TOP + plot_h,
        LEFT + plot_w,
        TOP + plot_h
    ));
    out.push_str(&format!(
        r##"<text x="20" y="{:.2}" transform="rotate(-90 20,{:.2})" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="#333">CPU saturation (core equivalents)</text>"##,
        TOP + plot_h / 2.0,
        TOP + plot_h / 2.0
    ));

    let points = samples
        .iter()
        .map(|sample| format!("{:.2},{:.2}", x(sample.elapsed), y(sample.cpu_x)))
        .collect::<Vec<_>>()
        .join(" ");
    out.push_str(&format!(
        r##"<polyline points="{points}" fill="none" stroke="#163f8c" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>"##
    ));

    for sample in samples.iter().step_by((samples.len() / 600).max(1)) {
        out.push_str(&format!(
            r##"<circle cx="{:.2}" cy="{:.2}" r="1.5" fill="#163f8c" opacity="0.55"/>"##,
            x(sample.elapsed),
            y(sample.cpu_x),
        ));
    }

    out.push_str("</svg>\n");
    fs::write(path, out)
}

fn cpu_stage_segments(samples: &[CpuTimelineSample]) -> Vec<(Duration, Duration, String)> {
    let Some(first) = samples.first() else {
        return Vec::new();
    };
    let mut segments = Vec::new();
    let mut start = Duration::ZERO;
    let mut current = first.stage.clone();
    let mut last = first.elapsed;
    for sample in samples {
        if sample.stage != current {
            segments.push((start, sample.elapsed, current));
            start = sample.elapsed;
            current = sample.stage.clone();
        }
        last = sample.elapsed;
    }
    segments.push((start, last, current));
    segments
}

fn short_stage_label(stage: &str) -> String {
    stage.rsplit('/').next().unwrap_or(stage).replace('_', " ")
}

fn stage_phase(stage: &str) -> &str {
    if stage.starts_with("run/generate") || stage.ends_with("/generate") {
        "generate"
    } else if stage.starts_with("run/build") || stage.ends_with("/build") {
        "build"
    } else if stage.starts_with("run/verify") || stage.ends_with("/verify") {
        "verify"
    } else if stage.starts_with("prove/") || stage.ends_with("/prove") {
        "prove"
    } else {
        stage.split('/').next().unwrap_or(stage)
    }
}

fn stage_color(stage: &str) -> &'static str {
    match stage_phase(stage) {
        "generate" => "#6b8fce",
        "build" => "#49a078",
        "prove" => "#9b6bd3",
        "verify" => "#cf5c5c",
        _ => "#c9c9c2",
    }
}

fn xml_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn print_report(
    shape: LeanAirShape,
    runs: &[RunMetrics],
    prove: bool,
    csv: bool,
    table: bool,
    cpu: bool,
    bandwidth: bool,
    whir_config: &WhirConfigBuilder,
) {
    if csv {
        print_csv(shape, runs, prove, cpu);
        return;
    }
    if table {
        print_markdown_table(shape, runs, prove);
        if cpu {
            print_cpu_markdown_table(runs, prove);
        }
        if bandwidth {
            print_bandwidth_markdown_table(runs, whir_config);
        }
        print_metric_definitions(prove);
        if cpu {
            print_cpu_metric_definitions();
        }
        if bandwidth {
            print_bandwidth_metric_definitions();
        }
        return;
    }

    let first = &runs[0];
    println!("lean-air dedicated poseidon hash table");
    println!("  log_m: {}", shape.log_m);
    println!("  rows: {}", shape.n_rows);
    println!("  cell_len_ext: {}", shape.cell_len_ext);
    println!("  cell_size_kib: {:.2}", cell_size_kib(shape));
    println!("  cells_per_codeword: {}", shape.num_cells());
    println!("  systematic_cells_per_codeword: {}", shape.num_systematic_cells());
    println!("  chunks_per_cell: {}", first.chunks_per_cell);
    println!("  active_poseidon_rows: {}", first.active_poseidon_rows);
    println!("  padded_poseidon_rows: {}", first.padded_poseidon_rows);
    println!("  committed_poseidon_columns: {}", first.committed_poseidon_columns);
    println!("  systematic_data_kib: {:.2}", data_kib(shape));
    println!("  runs: {}", runs.len());
    println!("  row_commitment_root: {:?}", first.row_commitment_root);
    println!("  column_commitment_root: {:?}", first.column_commitment_root);
    println!("  commitment_root: {:?}", first.commitment_root);

    print_run_table(shape, runs, prove);
    if cpu {
        print_cpu_table(runs, prove);
    }
    if bandwidth {
        print_bandwidth_table(runs, whir_config);
    }
    print_summary(shape, runs, prove);
    print_metric_definitions(prove);
    if cpu {
        print_cpu_metric_definitions();
    }
    if bandwidth {
        print_bandwidth_metric_definitions();
    }
}

fn print_markdown_table(shape: LeanAirShape, runs: &[RunMetrics], prove: bool) {
    println!("lean-air dedicated poseidon hash table benchmark");
    let first = &runs[0];
    println!();
    println!("shape");
    print_aligned_markdown_table(
        &[
            "log_m",
            "rows",
            "cell_ext",
            "cell_KiB",
            "cells/codeword",
            "active_rows",
            "padded_rows",
            "cols",
            "data_KiB",
        ],
        &[vec![
            shape.log_m.to_string(),
            shape.n_rows.to_string(),
            shape.cell_len_ext.to_string(),
            format!("{:.2}", cell_size_kib(shape)),
            shape.num_cells().to_string(),
            first.active_poseidon_rows.to_string(),
            first.padded_poseidon_rows.to_string(),
            first.committed_poseidon_columns.to_string(),
            format!("{:.2}", data_kib(shape)),
        ]],
    );
    println!();
    println!("runs");
    if prove {
        print_aligned_markdown_table(
            &[
                "run",
                "gen_ms",
                "build_ms",
                "prove_ms",
                "verification_ms",
                "proof_FE",
                "proof_KiB",
                "build_KiB/s",
                "prove_KiB/s",
            ],
            &runs
                .iter()
                .enumerate()
                .map(|(idx, run)| {
                    let proved = run.proved.as_ref().expect("prove metrics");
                    let verified = run.verified.as_ref().expect("verify metrics");
                    let proof_size_fe = run.proof_size_fe.expect("proof size");
                    vec![
                        (idx + 1).to_string(),
                        format!("{:.2}", duration_ms(run.generated_codewords.wall)),
                        format!("{:.2}", duration_ms(run.built_table.wall)),
                        format!("{:.2}", duration_ms(proved.wall)),
                        format!("{:.2}", duration_ms(verified.wall)),
                        proof_size_fe.to_string(),
                        format!("{:.2}", proof_size_kib(proof_size_fe)),
                        format!("{:.2}", build_throughput_kib_per_s(shape, run)),
                        format!("{:.2}", prover_throughput_kib_per_s(shape, run)),
                    ]
                })
                .collect::<Vec<_>>(),
        );
    } else {
        print_aligned_markdown_table(
            &["run", "gen_ms", "build_ms"],
            &runs
                .iter()
                .enumerate()
                .map(|(idx, run)| {
                    vec![
                        (idx + 1).to_string(),
                        format!("{:.2}", duration_ms(run.generated_codewords.wall)),
                        format!("{:.2}", duration_ms(run.built_table.wall)),
                    ]
                })
                .collect::<Vec<_>>(),
        );
    }
}

fn print_cpu_markdown_table(runs: &[RunMetrics], prove: bool) {
    println!();
    println!("cpu saturation");
    if prove {
        print_aligned_markdown_table(
            &["run", "gen_cpu_x", "build_cpu_x", "prove_cpu_x", "verification_cpu_x"],
            &runs
                .iter()
                .enumerate()
                .map(|(idx, run)| {
                    vec![
                        (idx + 1).to_string(),
                        format!("{:.2}", cpu_saturation(&run.generated_codewords)),
                        format!("{:.2}", cpu_saturation(&run.built_table)),
                        format!("{:.2}", cpu_saturation(run.proved.as_ref().expect("prove metrics"))),
                        format!("{:.2}", cpu_saturation(run.verified.as_ref().expect("verify metrics"))),
                    ]
                })
                .collect::<Vec<_>>(),
        );
    } else {
        print_aligned_markdown_table(
            &["run", "gen_cpu_x", "build_cpu_x"],
            &runs
                .iter()
                .enumerate()
                .map(|(idx, run)| {
                    vec![
                        (idx + 1).to_string(),
                        format!("{:.2}", cpu_saturation(&run.generated_codewords)),
                        format!("{:.2}", cpu_saturation(&run.built_table)),
                    ]
                })
                .collect::<Vec<_>>(),
        );
    }
}

fn print_aligned_markdown_table(headers: &[&str], rows: &[Vec<String>]) {
    let mut widths = headers.iter().map(|header| header.len()).collect::<Vec<_>>();
    for row in rows {
        for (idx, cell) in row.iter().enumerate() {
            widths[idx] = widths[idx].max(cell.len());
        }
    }

    print!("|");
    for (header, width) in headers.iter().zip(&widths) {
        let width = *width;
        print!(" {header:>width$} |");
    }
    println!();

    print!("|");
    for width in &widths {
        let width = *width;
        print!(" {:-<width$} |", "-");
    }
    println!();

    for row in rows {
        print!("|");
        for (cell, width) in row.iter().zip(&widths) {
            let width = *width;
            print!(" {cell:>width$} |");
        }
        println!();
    }
}

fn print_run_table(shape: LeanAirShape, runs: &[RunMetrics], prove: bool) {
    if prove {
        println!(
            "{:>4} {:>9} {:>9} {:>9} {:>15} {:>12} {:>13} {:>13}",
            "run", "gen_ms", "build_ms", "prove_ms", "verification_ms", "proof_kib", "build_KiB/s", "prove_KiB/s",
        );
    } else {
        println!("{:>4} {:>9} {:>9}", "run", "gen_ms", "build_ms");
    }

    for (idx, run) in runs.iter().enumerate() {
        if prove {
            let proved = run.proved.as_ref().expect("prove metrics");
            let verified = run.verified.as_ref().expect("verify metrics");
            let proof_size_fe = run.proof_size_fe.expect("proof size");
            println!(
                "{:>4} {:>9.2} {:>9.2} {:>9.2} {:>15.2} {:>12.2} {:>13.2} {:>13.2}",
                idx + 1,
                duration_ms(run.generated_codewords.wall),
                duration_ms(run.built_table.wall),
                duration_ms(proved.wall),
                duration_ms(verified.wall),
                proof_size_kib(proof_size_fe),
                build_throughput_kib_per_s(shape, run),
                prover_throughput_kib_per_s(shape, run),
            );
        } else {
            println!(
                "{:>4} {:>9.2} {:>9.2}",
                idx + 1,
                duration_ms(run.generated_codewords.wall),
                duration_ms(run.built_table.wall)
            );
        }
    }
}

fn print_cpu_table(runs: &[RunMetrics], prove: bool) {
    if prove {
        println!(
            "{:>4} {:>10} {:>11} {:>11} {:>18}",
            "run", "gen_cpu_x", "build_cpu_x", "prove_cpu_x", "verification_cpu_x"
        );
    } else {
        println!("{:>4} {:>10} {:>11}", "run", "gen_cpu_x", "build_cpu_x");
    }

    for (idx, run) in runs.iter().enumerate() {
        if prove {
            println!(
                "{:>4} {:>10.2} {:>11.2} {:>11.2} {:>18.2}",
                idx + 1,
                cpu_saturation(&run.generated_codewords),
                cpu_saturation(&run.built_table),
                cpu_saturation(run.proved.as_ref().expect("prove metrics")),
                cpu_saturation(run.verified.as_ref().expect("verify metrics")),
            );
        } else {
            println!(
                "{:>4} {:>10.2} {:>11.2}",
                idx + 1,
                cpu_saturation(&run.generated_codewords),
                cpu_saturation(&run.built_table),
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MemoryTrafficEstimate {
    stage: &'static str,
    bytes: u128,
    note: &'static str,
}

fn print_bandwidth_markdown_table(runs: &[RunMetrics], whir_config: &WhirConfigBuilder) {
    println!();
    println!("memory bandwidth estimates");
    print_aligned_markdown_table(
        &["stage", "wall_ms", "est_GiB", "eff_GiB/s", "model"],
        &bandwidth_rows(runs, whir_config),
    );
}

fn print_bandwidth_table(runs: &[RunMetrics], whir_config: &WhirConfigBuilder) {
    println!("memory bandwidth estimates");
    println!(
        "{:>34} {:>10} {:>10} {:>11}  {}",
        "stage", "wall_ms", "est_GiB", "eff_GiB/s", "model"
    );
    for row in bandwidth_rows(runs, whir_config) {
        println!(
            "{:>34} {:>10} {:>10} {:>11}  {}",
            row[0], row[1], row[2], row[3], row[4]
        );
    }
}

fn bandwidth_rows(runs: &[RunMetrics], whir_config: &WhirConfigBuilder) -> Vec<Vec<String>> {
    memory_traffic_estimates(&runs[0], whir_config)
        .into_iter()
        .map(|estimate| {
            let wall = mean_stage_duration(runs, estimate.stage);
            let wall_ms = wall.map_or_else(|| "n/a".to_string(), |duration| format!("{:.2}", duration_ms(duration)));
            let gib = bytes_gib(estimate.bytes);
            let bandwidth = wall.filter(|duration| !duration.is_zero()).map_or_else(
                || "n/a".to_string(),
                |duration| format!("{:.2}", gib / duration.as_secs_f64()),
            );
            vec![
                estimate.stage.to_string(),
                wall_ms,
                format!("{gib:.2}"),
                bandwidth,
                estimate.note.to_string(),
            ]
        })
        .collect()
}

fn memory_traffic_estimates(run: &RunMetrics, whir_config: &WhirConfigBuilder) -> Vec<MemoryTrafficEstimate> {
    let actual_data_len = run.committed_poseidon_columns * run.padded_poseidon_rows;
    let global_len = 1usize << log2_ceil(actual_data_len);
    let global_n_vars = log2_ceil(global_len);
    let f_bytes = std::mem::size_of::<F>() as u128;
    let pf_pack_bytes = std::mem::size_of::<PFPacking<EF>>() as u128;
    let ef_pack_bytes = std::mem::size_of::<EFPacking<EF>>() as u128;
    let pack_log = packing_log_width::<EF>();
    let packed_len = global_len >> pack_log;

    let first_fold = whir_config.folding_factor.at_round(0).min(global_n_vars);
    let n_blocks = 1usize << first_fold;
    let domain_size = 1usize << (global_n_vars + whir_config.starting_log_inv_rate);
    let commit_height = domain_size / n_blocks;
    let evals_len = 1usize << global_n_vars;
    let effective_n_cols = actual_data_len.div_ceil(evals_len / n_blocks);
    let dft_n_cols = effective_n_cols
        .next_multiple_of(backend::packing_width::<EF>())
        .min(n_blocks);
    let digest_bytes = (lean_vm::DIGEST_LEN as u128) * f_bytes;

    vec![
        MemoryTrafficEstimate {
            stage: "prove/global_polynomial",
            bytes: (actual_data_len as u128 + global_len as u128) * f_bytes,
            note: "read committed columns + write padded global MLE",
        },
        MemoryTrafficEstimate {
            stage: "prove/whir/commit_fft",
            bytes: (global_len as u128 * f_bytes) + (commit_height as u128 * dft_n_cols as u128 * f_bytes),
            note: "lower bound: read global MLE + write first folded matrix",
        },
        MemoryTrafficEstimate {
            stage: "prove/whir/commit_merkle",
            bytes: (commit_height as u128 * dft_n_cols as u128 * f_bytes) + (commit_height as u128 * digest_bytes * 3),
            note: "lower bound: hash first matrix layer and digest tree",
        },
        MemoryTrafficEstimate {
            stage: "prove/whir/combine_statement",
            bytes: combine_statement_bytes(run, global_len, pack_log, ef_pack_bytes),
            note: "zero combined weights + copy weighted row-eq chunks",
        },
        MemoryTrafficEstimate {
            stage: "prove/whir/initial_product_sumcheck",
            bytes: initial_product_sumcheck_bytes(packed_len, first_fold, pf_pack_bytes, ef_pack_bytes),
            note: "read evals/weights and fold packed vectors in place",
        },
    ]
}

fn combine_statement_bytes(run: &RunMetrics, global_len: usize, pack_log: usize, ef_pack_bytes: u128) -> u128 {
    let combined_weight_bytes = (global_len >> pack_log) as u128 * ef_pack_bytes;
    let inner_packed_len = run.padded_poseidon_rows >> pack_log;
    let dense_statement_bytes = run.committed_poseidon_columns as u128 * inner_packed_len as u128 * ef_pack_bytes;
    combined_weight_bytes + 2 * dense_statement_bytes
}

fn initial_product_sumcheck_bytes(
    packed_len: usize,
    first_fold: usize,
    pf_pack_bytes: u128,
    ef_pack_bytes: u128,
) -> u128 {
    let mut bytes = packed_len as u128 * (pf_pack_bytes + ef_pack_bytes);
    if first_fold == 0 {
        return bytes;
    }

    bytes += packed_len as u128 * (pf_pack_bytes + ef_pack_bytes);
    bytes += packed_len as u128 * ef_pack_bytes;

    let mut len = packed_len / 2;
    for _ in 0..first_fold.saturating_sub(2) {
        bytes += 3 * len as u128 * ef_pack_bytes;
        len /= 2;
    }
    bytes += 3 * len as u128 * ef_pack_bytes;
    bytes
}

fn mean_stage_duration(runs: &[RunMetrics], stage: &str) -> Option<Duration> {
    let mut total = Duration::ZERO;
    let mut count = 0u32;
    for run in runs {
        if let Some(duration) = stage_duration(run, stage) {
            total += duration;
            count += 1;
        }
    }
    (count != 0).then(|| total / count)
}

fn stage_duration(run: &RunMetrics, stage: &str) -> Option<Duration> {
    run.stage_timings
        .iter()
        .find_map(|(label, duration)| (label == stage).then_some(*duration))
}

fn bytes_gib(bytes: u128) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn print_summary(shape: LeanAirShape, runs: &[RunMetrics], prove: bool) {
    print_stat(
        "generated_codewords_ms",
        &runs
            .iter()
            .map(|run| duration_ms(run.generated_codewords.wall))
            .collect::<Vec<_>>(),
    );
    print_stat(
        "built_table_ms",
        &runs
            .iter()
            .map(|run| duration_ms(run.built_table.wall))
            .collect::<Vec<_>>(),
    );

    if prove {
        print_stat(
            "proved_ms",
            &runs
                .iter()
                .map(|run| duration_ms(run.proved.as_ref().expect("prove metrics").wall))
                .collect::<Vec<_>>(),
        );
        print_stat(
            "verification_ms",
            &runs
                .iter()
                .map(|run| duration_ms(run.verified.as_ref().expect("verify metrics").wall))
                .collect::<Vec<_>>(),
        );
        print_stat(
            "proof_size_kib",
            &runs
                .iter()
                .map(|run| proof_size_kib(run.proof_size_fe.expect("proof size")))
                .collect::<Vec<_>>(),
        );
        print_stat(
            "build_throughput_kib_per_s",
            &runs
                .iter()
                .map(|run| build_throughput_kib_per_s(shape, run))
                .collect::<Vec<_>>(),
        );
        print_stat(
            "prover_throughput_kib_per_s",
            &runs
                .iter()
                .map(|run| prover_throughput_kib_per_s(shape, run))
                .collect::<Vec<_>>(),
        );
    }
}

fn print_csv(shape: LeanAirShape, runs: &[RunMetrics], prove: bool, cpu: bool) {
    let cpu_header = if cpu {
        if prove {
            ",generated_codewords_cpu_x,built_table_cpu_x,proved_cpu_x,verification_cpu_x"
        } else {
            ",generated_codewords_cpu_x,built_table_cpu_x"
        }
    } else {
        ""
    };
    if prove {
        println!(
            "run,log_m,n_rows,cell_len_ext,cell_size_kib,cells_per_codeword,active_poseidon_rows,padded_poseidon_rows,columns,systematic_data_kib,generated_codewords_ms,built_table_ms,proved_ms,verification_ms,proof_size_fe,proof_size_kib,peak_rss_bytes,build_throughput_kib_per_s,prover_throughput_kib_per_s{cpu_header}"
        );
    } else {
        println!(
            "run,log_m,n_rows,cell_len_ext,cell_size_kib,cells_per_codeword,active_poseidon_rows,padded_poseidon_rows,columns,systematic_data_kib,generated_codewords_ms,built_table_ms,peak_rss_bytes{cpu_header}"
        );
    }

    for (idx, run) in runs.iter().enumerate() {
        if prove {
            let proved = run.proved.as_ref().expect("prove metrics");
            let verified = run.verified.as_ref().expect("verify metrics");
            let proof_size_fe = run.proof_size_fe.expect("proof size");
            let cpu_values = if cpu {
                format!(
                    ",{:.2},{:.2},{:.2},{:.2}",
                    cpu_saturation(&run.generated_codewords),
                    cpu_saturation(&run.built_table),
                    cpu_saturation(proved),
                    cpu_saturation(verified),
                )
            } else {
                String::new()
            };
            println!(
                "{},{},{},{},{:.2},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{},{:.2},{},{:.2},{:.2}{}",
                idx + 1,
                shape.log_m,
                shape.n_rows,
                shape.cell_len_ext,
                cell_size_kib(shape),
                shape.num_cells(),
                run.active_poseidon_rows,
                run.padded_poseidon_rows,
                run.committed_poseidon_columns,
                data_kib(shape),
                duration_ms(run.generated_codewords.wall),
                duration_ms(run.built_table.wall),
                duration_ms(proved.wall),
                duration_ms(verified.wall),
                proof_size_fe,
                proof_size_kib(proof_size_fe),
                run.peak_rss_bytes,
                build_throughput_kib_per_s(shape, run),
                prover_throughput_kib_per_s(shape, run),
                cpu_values,
            );
        } else {
            let cpu_values = if cpu {
                format!(
                    ",{:.2},{:.2}",
                    cpu_saturation(&run.generated_codewords),
                    cpu_saturation(&run.built_table),
                )
            } else {
                String::new()
            };
            println!(
                "{},{},{},{},{:.2},{},{},{},{},{:.2},{:.2},{:.2},{}{}",
                idx + 1,
                shape.log_m,
                shape.n_rows,
                shape.cell_len_ext,
                cell_size_kib(shape),
                shape.num_cells(),
                run.active_poseidon_rows,
                run.padded_poseidon_rows,
                run.committed_poseidon_columns,
                data_kib(shape),
                duration_ms(run.generated_codewords.wall),
                duration_ms(run.built_table.wall),
                run.peak_rss_bytes,
                cpu_values,
            );
        }
    }
}

fn data_kib(shape: LeanAirShape) -> f64 {
    let data_len_base = shape.n_rows * shape.message_len_ext() * <EF as BasedVectorSpace<F>>::DIMENSION;
    (data_len_base * F_BITS) as f64 / (8.0 * 1024.0)
}

fn cell_size_kib(shape: LeanAirShape) -> f64 {
    (shape.cell_len_base() * F_BITS) as f64 / (8.0 * 1024.0)
}

fn proof_size_kib(proof_size_fe: usize) -> f64 {
    (proof_size_fe * F_BITS) as f64 / (8.0 * 1024.0)
}

fn build_throughput_kib_per_s(shape: LeanAirShape, run: &RunMetrics) -> f64 {
    data_kib(shape) / run.built_table.wall.as_secs_f64()
}

fn prover_throughput_kib_per_s(shape: LeanAirShape, run: &RunMetrics) -> f64 {
    data_kib(shape) / run.proved.as_ref().expect("prove metrics").wall.as_secs_f64()
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn log2_ceil(n: usize) -> usize {
    assert!(n > 0);
    usize::BITS as usize - (n - 1).leading_zeros() as usize
}

fn cpu_saturation(stage: &StageMetrics) -> f64 {
    if stage.wall.is_zero() {
        0.0
    } else {
        stage.cpu.as_secs_f64() / stage.wall.as_secs_f64()
    }
}

fn print_stat(label: &str, values: &[f64]) {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    println!("  {label}: mean={mean:.2} min={min:.2} max={max:.2}");
}

fn print_metric_definitions(prove: bool) {
    println!();
    println!("metric definitions:");
    println!("  data_KiB/systematic_data_kib: original systematic payload size, counted as 31-bit base-field limbs");
    println!("  cell_KiB/cell_size_kib: one cell size, counted as 31-bit base-field limbs");
    println!("  cells/codeword/cells_per_codeword: full RS codeword cells, including systematic and parity halves");
    println!("  gen_ms: deterministic benchmark codeword generation and RS encoding time");
    println!("  build_ms/built_table_ms: Construction 4 hash table, commitments, and trace-column build time");
    if prove {
        println!("  prove_ms/proved_ms: proof generation time after the hash table has already been built");
        println!("  verification_ms: verifier time for the generated proof");
        println!("  build_KiB/s: data_KiB divided by build_ms only");
        println!("  prove_KiB/s: data_KiB divided by prove_ms only");
        println!("  proof_KiB/proof_size_kib: proof size measured as 31 bits per proof field element");
    }
}

fn print_cpu_metric_definitions() {
    println!("  *_cpu_x: process user+system CPU time divided by wall time for that stage; 1.0 is one saturated core");
}

fn print_bandwidth_metric_definitions() {
    println!(
        "  est_GiB: modeled memory traffic for the named stage; this is a software lower-bound estimate, not a hardware counter"
    );
    println!("  eff_GiB/s: est_GiB divided by the measured stage wall time from --bandwidth stage timing");
}
