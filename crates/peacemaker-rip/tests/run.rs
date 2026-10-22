//! `peacemaker-rip run` CLI: cookbook cases exit 0, negatives exit 1 with
//! structured reports, dumps round-trip. Case JSON lives in `tests/cases/`
//! and mirrors the cookbook_a/b coverage, never invented kernels.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const ARCHES: [&str; 2] = ["gfx1151", "gfx1201"];

fn rip(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_peacemaker-rip"))
        .args(args)
        .env("ROCR_VISIBLE_DEVICES", "-1")
        .env("HIP_VISIBLE_DEVICES", "-1")
        .output()
        .expect("run peacemaker-rip")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rip-run-{}-{test}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn write(dir: &Path, name: &str, body: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, body).expect("write fixture");
    p.display().to_string()
}

/// Extract the single complete kernel from a cookbook file; no copied fixtures.
fn recipe(file: &str) -> (String, String) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("docs/cookbook")
        .join(file);
    let markdown =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut blocks = Vec::new();
    let mut lines = markdown.lines();
    while let Some(line) = lines.next() {
        if let Some(info) = line.trim().strip_prefix("```rip") {
            let variant = info
                .split_whitespace()
                .find_map(|w| w.strip_prefix("variant="))
                .expect("recipe must declare variant=NAME")
                .to_string();
            let mut source = String::new();
            let mut closed = false;
            for body in lines.by_ref() {
                if body.trim() == "```" {
                    closed = true;
                    break;
                }
                source.push_str(body);
                source.push('\n');
            }
            assert!(closed, "{}: unclosed rip block", path.display());
            blocks.push((source, variant));
        }
    }
    assert_eq!(
        blocks.len(),
        1,
        "{}: expected exactly one complete rip kernel",
        path.display()
    );
    blocks.pop().unwrap()
}

fn case_source(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/cases")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Stage one cookbook kernel plus a copy of a checked-in case; run and parse
/// the single-target schema-1 report.
fn stage(
    dir: &Path,
    recipe_file: &str,
    case_file: &str,
    case_override: Option<&str>,
) -> (String, String, String) {
    let (source, variant) = recipe(recipe_file);
    let rip = write(dir, "kernel.rip", &source);
    let body = case_override
        .map(str::to_string)
        .unwrap_or_else(|| case_source(case_file));
    let case = write(dir, "kernel.json", &body);
    (rip, case, variant)
}

fn run_one(rip: &str, case: &str, arch: &str, variant: &str, extra: &[&str]) -> (Output, Value) {
    let mut args = vec![
        "run",
        rip,
        "--case",
        case,
        "--arch",
        arch,
        "--variant",
        variant,
        "--json",
    ];
    args.extend_from_slice(extra);
    let output = self::rip(&args);
    let report: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|e| {
        panic!(
            "parse run JSON: {e}\nstdout: {}\nstderr: {}",
            text(&output.stdout),
            text(&output.stderr)
        )
    });
    (output, report)
}

fn ok_report(report: &Value, arch: &str, variant: &str) {
    assert_eq!(report["schema"], 1, "{report}");
    assert_eq!(report["arch"], arch, "{report}");
    assert_eq!(report["variant"], variant, "{report}");
    assert_eq!(report["error"], Value::Null, "{report}");
    assert_eq!(report["ok"], true, "{report}");
}

/// (recipe markdown, checked-in case, variants to cover).
const COOKBOOK: &[(&str, &str, &[&str])] = &[
    ("01-minimal-kernel.md", "01_fill_n5.json", &["fill"]),
    (
        "02-elementwise-bounds.md",
        "02_subtract_n3.json",
        &["subtract"],
    ),
    ("03-workgroup-ids.md", "03_grid_ids_3x2.json", &["grid_ids"]),
    (
        "04-wave-reduction.md",
        "04_wave_min_2groups.json",
        &["wave_min"],
    ),
    (
        "05-lds-handoff.md",
        "05_lds_reverse_groups2.json",
        &["reverse"],
    ),
    (
        "06-device-loop.md",
        "06_device_loop_n5.json",
        &["default", "copy_carried"],
    ),
    (
        "07-uniform-branch.md",
        "07_branch_add_n3.json",
        &["default"],
    ),
    (
        "07-uniform-branch.md",
        "07_branch_passthrough_n3.json",
        &["default"],
    ),
    (
        "08-wmma.md",
        "08_wmma_bf16_tile_pattern0.json",
        &["bf16_tile"],
    ),
    (
        "09-kernarg-loads-and-waits.md",
        "09_kernarg_affine_n103_first5.json",
        &["kernarg_affine"],
    ),
];

#[test]
fn cookbook_cases_exit_zero_on_both_arches() {
    for (recipe_file, case_file, variants) in COOKBOOK {
        for arch in ARCHES {
            for request in *variants {
                let dir = scratch("cookbook");
                let (rip, case, declared) = stage(&dir, recipe_file, case_file, None);
                let variant = if recipe_file == &"06-device-loop.md" {
                    (*request).to_string()
                } else {
                    declared
                };
                let (output, report) = run_one(&rip, &case, arch, &variant, &[]);
                assert!(
                    output.status.success(),
                    "{recipe_file} {arch} {variant}: {}",
                    text(&output.stderr)
                );
                ok_report(&report, arch, &variant);
                for (name, entry) in report["buffers"].as_object().expect("buffers object") {
                    if entry["kind"] == "output" {
                        assert_eq!(entry["guard_ok"], true, "{recipe_file} {name}: {entry}");
                        assert!(
                            entry.get("first_mismatch").is_none(),
                            "{recipe_file} {name}: {entry}"
                        );
                    } else {
                        assert_eq!(entry["changed"], false, "{recipe_file} {name}: {entry}");
                    }
                }
            }
        }
    }
}

#[test]
fn wrong_expect_reports_element_index() {
    let mut case: Value =
        serde_json::from_str(&case_source("01_fill_n5.json")).expect("parse case");
    case["buffers"][0]["expect"][2] = Value::from(999u64);
    let dir = scratch("wrong-expect");
    let (rip, case_path, variant) = stage(
        &dir,
        "01-minimal-kernel.md",
        "01_fill_n5.json",
        Some(&case.to_string()),
    );
    let (output, report) = run_one(&rip, &case_path, "gfx1151", &variant, &[]);
    assert!(!output.status.success(), "wrong expect must fail");
    assert_eq!(report["schema"], 1, "{report}");
    assert_eq!(report["ok"], false, "{report}");
    assert_eq!(report["error"], Value::Null, "{report}");
    assert_eq!(report["buffers"]["out"]["first_mismatch"], 2, "{report}");
    assert_eq!(report["buffers"]["out"]["guard_ok"], true, "{report}");
}

#[test]
fn mutated_input_reports_byte_offset() {
    // Drop `expect` from the fill output and seed zeros: it becomes an input
    // the kernel overwrites, so the run must report the first changed byte.
    let mut case: Value =
        serde_json::from_str(&case_source("01_fill_n5.json")).expect("parse case");
    case["buffers"][0]
        .as_object_mut()
        .expect("buffer object")
        .remove("expect");
    case["buffers"][0]["values"] = Value::Array(vec![Value::from(0u64); 5]);
    let dir = scratch("mutated-input");
    let (rip, case_path, variant) = stage(
        &dir,
        "01-minimal-kernel.md",
        "01_fill_n5.json",
        Some(&case.to_string()),
    );
    let (output, report) = run_one(&rip, &case_path, "gfx1151", &variant, &[]);
    assert!(!output.status.success(), "mutated input must fail");
    assert_eq!(report["ok"], false, "{report}");
    assert_eq!(report["buffers"]["out"]["kind"], "input", "{report}");
    assert_eq!(report["buffers"]["out"]["changed"], true, "{report}");
    assert_eq!(report["buffers"]["out"]["first_mismatch"], 0, "{report}");
}
#[test]
fn truncated_expect_trips_the_output_guard() {
    // Keep n=5 and the grid, but expect only four words: the fifth store
    // lands in the 256-byte 0xAA guard and must fail the run.
    let mut case: Value =
        serde_json::from_str(&case_source("01_fill_n5.json")).expect("parse case");
    let expect = case["buffers"][0]["expect"]
        .as_array()
        .expect("expect array")
        .clone();
    case["buffers"][0]["expect"] = Value::Array(expect[..4].to_vec());
    let dir = scratch("guard");
    let (rip, case_path, variant) = stage(
        &dir,
        "01-minimal-kernel.md",
        "01_fill_n5.json",
        Some(&case.to_string()),
    );
    let (output, report) = run_one(&rip, &case_path, "gfx1151", &variant, &[]);
    assert!(!output.status.success(), "guard damage must fail");
    assert_eq!(report["ok"], false, "{report}");
    assert_eq!(report["buffers"]["out"]["guard_ok"], false, "{report}");
    assert!(
        report["buffers"]["out"].get("first_mismatch").is_none(),
        "{report}"
    );
}

/// Each entry is a whole mutated 01 case; all fail closed at `case` stage.
fn invalid_cases() -> Vec<(String, Value)> {
    let base: Value = serde_json::from_str(&case_source("01_fill_n5.json")).expect("parse case");
    let mut cases: Vec<(String, Value)> = Vec::new();
    let mut push = |name: &str, f: &dyn Fn(&mut Value)| {
        let mut case = base.clone();
        f(&mut case);
        cases.push((name.to_string(), case));
    };
    push("unknown_dtype", &|c| {
        c["buffers"][0]["dtype"] = Value::from("f17")
    });
    push("missing_size", &|c| {
        c["kernargs"]
            .as_object_mut()
            .expect("kernargs")
            .remove("size");
    });
    push("overlap", &|c| {
        c["kernargs"]["fields"]
            .as_array_mut()
            .expect("fields")
            .push(serde_json::json!({"offset": 8, "u32": 1}));
    });
    push("unknown_ptr", &|c| {
        c["kernargs"]["fields"][0]["ptr"] = Value::from("nope")
    });
    push("unknown_root", &|c| c["bogus"] = Value::from(1));
    push("unknown_buffer", &|c| {
        c["buffers"][0]["bogus"] = Value::from(1)
    });
    push("unknown_kernargs", &|c| {
        c["kernargs"]["bogus"] = Value::from(1)
    });
    push("unknown_field", &|c| {
        c["kernargs"]["fields"][1]["bogus"] = Value::from(1)
    });
    push("duplicate_name", &|c| {
        c["buffers"]
            .as_array_mut()
            .expect("buffers")
            .push(serde_json::json!({"name": "out", "dtype": "u32", "expect": [1]}));
    });
    push("length_mismatch", &|c| {
        c["buffers"][0]["values"] = serde_json::json!([1, 2, 3, 4, 5]);
        c["buffers"][0]["expect"] = serde_json::json!([1, 2, 3, 4]);
    });
    push("two_payloads", &|c| {
        c["kernargs"]["fields"][1]["u16"] = Value::from(9)
    });
    push("no_payload", &|c| {
        c["kernargs"]["fields"][1]
            .as_object_mut()
            .expect("field")
            .remove("u32");
    });
    push("past_size", &|c| {
        c["kernargs"]["fields"][1]["offset"] = Value::from(99)
    });
    push("invalid_hex", &|c| {
        c["kernargs"]["fields"][0] = serde_json::json!({"offset": 0, "bytes": "zz"});
    });
    cases
}

#[test]
fn invalid_cases_fail_closed_at_case_stage() {
    let dir = scratch("invalid");
    let (source, variant) = recipe("01-minimal-kernel.md");
    let rip = write(&dir, "kernel.rip", &source);
    for (name, case) in invalid_cases() {
        let case_path = write(&dir, &format!("{name}.json"), &case.to_string());
        let (output, report) = run_one(&rip, &case_path, "gfx1151", &variant, &[]);
        assert!(!output.status.success(), "{name} must fail");
        assert_eq!(report["schema"], 1, "{name}: {report}");
        assert_eq!(report["ok"], false, "{name}: {report}");
        assert_eq!(report["error"]["stage"], "case", "{name}: {report}");
        assert!(report["error"]["message"].is_string(), "{name}: {report}");
    }
}

#[test]
fn dump_roundtrip_replays_exactly() {
    let dir = scratch("dump");
    let (rip, case, variant) = stage(&dir, "01-minimal-kernel.md", "01_fill_n5.json", None);
    let dump = dir.join("dump.json").display().to_string();
    let (output, report) = run_one(&rip, &case, "gfx1151", &variant, &["--dump", &dump]);
    assert!(
        output.status.success(),
        "dump run: {}",
        text(&output.stderr)
    );
    assert_eq!(report["ok"], true, "{report}");
    let body = std::fs::read_to_string(&dump).expect("read dump");
    assert!(body.ends_with('\n'), "dump ends with a newline");
    let dumped: Value = serde_json::from_str(&body).expect("parse dump");
    assert_eq!(dumped["source"], "dump", "{dumped}");
    assert!(
        dumped["description"]
            .as_str()
            .is_some_and(|d| d.contains("not correctness")),
        "{dumped}"
    );
    let replay = write(&dir, "replay-case.json", &body);
    let (second, replayed) = run_one(&rip, &replay, "gfx1151", &variant, &[]);
    assert!(second.status.success(), "replay: {}", text(&second.stderr));
    assert_eq!(report["ok"], replayed["ok"], "{report} vs {replayed}");
    assert_eq!(replayed["ok"], true, "{replayed}");
}

#[test]
fn dump_of_a_failing_case_replays_its_behavior() {
    let mut case: Value =
        serde_json::from_str(&case_source("01_fill_n5.json")).expect("parse case");
    case["buffers"][0]["expect"][2] = Value::from(999u64);
    let dir = scratch("dump-fail");
    let (rip, case_path, variant) = stage(
        &dir,
        "01-minimal-kernel.md",
        "01_fill_n5.json",
        Some(&case.to_string()),
    );
    let dump = dir.join("dump.json").display().to_string();
    let (output, report) = run_one(&rip, &case_path, "gfx1151", &variant, &["--dump", &dump]);
    assert!(!output.status.success(), "failing dump run must fail");
    assert_eq!(report["ok"], false, "{report}");
    let body = std::fs::read_to_string(&dump).expect("dump written even when expectations fail");
    let replay = write(&dir, "replay-case.json", &body);
    let (second, replayed) = run_one(&rip, &replay, "gfx1151", &variant, &[]);
    assert!(second.status.success(), "replay: {}", text(&second.stderr));
    assert_ne!(report["ok"], replayed["ok"], "{report} vs {replayed}");
    assert_eq!(replayed["ok"], true, "{replayed}");
}

#[test]
fn default_case_is_adjacent_to_the_source() {
    let dir = scratch("adjacent");
    let (source, variant) = recipe("01-minimal-kernel.md");
    let rip = write(&dir, "fill.rip", &source);
    write(&dir, "fill.json", &case_source("01_fill_n5.json"));
    let output = self::rip(&[
        "run",
        &rip,
        "--arch",
        "gfx1151",
        "--variant",
        &variant,
        "--json",
    ]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    let report: Value = serde_json::from_slice(&output.stdout).expect("parse run JSON");
    assert_eq!(report["ok"], true, "{report}");
    assert_eq!(report["case"], "n=5,k=3,c=19", "{report}");
}

#[test]
fn multi_target_report_is_an_envelope() {
    let dir = scratch("multi");
    let (rip, case, variant) = stage(&dir, "01-minimal-kernel.md", "01_fill_n5.json", None);
    let output = self::rip(&[
        "run",
        &rip,
        "--case",
        &case,
        "--arch",
        "gfx1151,gfx1201",
        "--variant",
        &variant,
        "--json",
    ]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse run JSON");
    assert_eq!(envelope["schema"], 1, "{envelope}");
    assert_eq!(envelope["ok"], true, "{envelope}");
    let results = envelope["results"].as_array().expect("results array");
    assert_eq!(results.len(), 2, "{envelope}");
    for report in results {
        assert_eq!(report["schema"], 1, "{report}");
        assert_eq!(report["ok"], true, "{report}");
    }
}

#[test]
fn human_output_names_buffers_and_verdict() {
    let dir = scratch("human");
    let (rip, case, variant) = stage(&dir, "01-minimal-kernel.md", "01_fill_n5.json", None);
    let output = self::rip(&[
        "run",
        &rip,
        "--case",
        &case,
        "--arch",
        "gfx1151",
        "--variant",
        &variant,
    ]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    let stdout = text(&output.stdout).to_lowercase();
    assert!(stdout.contains("out"), "{stdout}");
    assert!(stdout.contains("ok") || stdout.contains("pass"), "{stdout}");
}
