// SPDX-License-Identifier: Apache-2.0
//! Frozen from crossover_model.py SHA256 d50b20c66d1c62b243e88e1b23e956c3e072d31937b3352305b7ba126f4acc4a.
use hipfire_arch_qwen4::npu_trunk::*;
use serde_json::{json, Value};
use std::path::PathBuf;
fn fixture(name: &str) -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/npu_trunk").join(name) }
fn cal(name: &str) -> Calibration {
    // Preserve original lab JSON fixtures verbatim. Only adapt their legacy key
    // and absent version/margin to the production record; measured costs unchanged.
    let mut v: Value = serde_json::from_slice(&std::fs::read(fixture(name)).unwrap()).unwrap();
    v["schema_version"] = json!(1);
    // Python's legacy get('panel_cols', 1), not a production input default.
    if v["npu"].get("panel_cols").is_none() { v["npu"]["panel_cols"] = json!(1); }
    v["margin"] = json!(0);
    v["key"] = json!({"device":{"arch":"gfx1151","pci_bdf":"0000:c1:00.1","npu_fw_version":"fixture"},"build":"fixture","kernel":{"uname_release":"fixture","perfopt_armed":true},"model_sha256":"0000000000000000000000000000000000000000000000000000000000000000","epilogue":v["npu"]["epilogue"]});
    let c: Calibration = serde_json::from_value(v).unwrap(); c.validate().unwrap(); c
}
fn mask(names: &str) -> FamilyMask {
    if names.contains("zab") { FamilyMask::ALL }
    else { FamilyMask::from(Family::GdnQkv) | FamilyMask::QSA }
}
fn golden(names: &str) {
    let vectors: Value = serde_json::from_str(include_str!("fixtures/npu_trunk/golden.json")).unwrap();
    let mut count = 0;
    for v in vectors.as_array().unwrap().iter().filter(|v| v["families"] == names) {
        let name = v["file"].as_str().unwrap(); let c = cal(name); let families = mask(names);
        let star = v["n_star"].as_u64().map(|n| n as usize);
        assert_eq!(crossover(&c, families), star, "{name} {names}");
        for p in v["plans"].as_array().unwrap() {
            let rows = p["rows"].as_u64().unwrap() as usize;
            let actual = plan(&c, families, rows);
            let expected: Vec<_> = p["steps"].as_array().unwrap().iter().map(|s| {
                if let Some(r) = s.get("gpu") { Step::Gpu { chunk: r[0].as_u64().unwrap() as usize..r[1].as_u64().unwrap() as usize } }
                else { let r = &s["pipe"]; Step::Pipe {
                    a: r[0][0].as_u64().unwrap() as usize..r[0][1].as_u64().unwrap() as usize,
                    b: r[1][0].as_u64().unwrap() as usize..r[1][1].as_u64().unwrap() as usize,
                } }
            }).collect();
            assert_eq!(actual.steps, expected, "{name} {names} rows={rows}: {}", p["python_plan"]);
            for (value, expected) in [(actual.t_pipe_ms, p["t_pipe_ms"].as_f64().unwrap()), (actual.t_gpu_ms, p["t_gpu_ms"].as_f64().unwrap())] {
                assert!((value - expected).abs() <= expected.abs() * 1e-6, "{name} {names} rows={rows}: {value} != {expected}");
            }
            if name == "g80bf16-measured.json" {
                println!("families={names} n*={star:?} rows={rows} gpu={:.9} pipe={:.9} {}", actual.t_gpu_ms, actual.t_pipe_ms, request_log_line(rows, &actual));
            }
            count += 1;
        }
    }
    assert_eq!(count, 45);
}
#[test] fn npu_trunk_golden_family_a() { golden("qkv,q,kvidx"); }
#[test] fn npu_trunk_golden_family_b() { golden("qkv,zab,q,kvidx"); }
#[test] fn npu_trunk_margin_and_determinism() {
    let mut c = cal("g80bf16-measured.json");
    let initial = plan(&c, FamilyMask::ALL, 8192);
    assert!(matches!(initial.steps[0], Step::Pipe { .. }));
    c.margin = 0.10;
    assert!(matches!(plan(&c, FamilyMask::ALL, 8192).steps[0], Step::Gpu { .. }));
    c.margin = 0.99;
    for n in [0, 2048, 8192, 9319, 16384, 24576, 65536] {
        let p = plan(&c, FamilyMask::ALL, n);
        assert_eq!(p, plan(&c, FamilyMask::ALL, n));
        assert_eq!(p.t_pipe_ms, p.t_gpu_ms);
        assert!(p.steps.iter().all(|s| matches!(s, Step::Gpu { .. })));
    }
    assert_eq!(crossover(&c, FamilyMask::ALL), None);
}
#[test] fn npu_trunk_key_mismatches_and_logs() {
    let c = cal("g80bf16-measured.json"); let p = plan(&c, FamilyMask::ALL, 16384);
    assert_eq!(c.key_matches(&c.key), Ok(()));
    let mutations: [(&str, fn(&mut CalKey)); 8] = [
        ("device.arch", |k| k.device.arch += "x"), ("device.pci_bdf", |k| k.device.pci_bdf += "x"),
        ("device.npu_fw_version", |k| k.device.npu_fw_version += "x"), ("build", |k| k.build += "x"),
        ("kernel.uname_release", |k| k.kernel.uname_release += "x"), ("kernel.perfopt_armed", |k| k.kernel.perfopt_armed = !k.kernel.perfopt_armed),
        ("model_sha256", |k| k.model_sha256 += "x"), ("epilogue", |k| k.epilogue += "x"),
    ];
    for (field, mutate) in mutations {
        let mut key = c.key.clone(); mutate(&mut key);
        assert_eq!(c.key_matches(&key), Err(format!("calibration key mismatch: {field}")));
        let log_key = LoadLogKey { current: &key, mode: "auto", families: FamilyMask::ALL };
        assert_eq!(load_log_line(&c, &log_key, None, &p, &p), format!("npu-trunk: off (calibration key mismatch: {field})"));
    }
    let key = LoadLogKey { current: &c.key, mode: "auto", families: FamilyMask::ALL };
    let line = load_log_line(&c, &key, Some(4736), &p, &plan(&c, FamilyMask::ALL, 65536));
    assert!(line.starts_with("npu-trunk: mode=auto contract=bf16 families=qkv,zab,q,kvidx cal_key="));
    assert!(line.contains(" cal_sha256=")); assert!(line.contains(" n*=4736 T(16384)="));
    assert_eq!(request_log_line(16384, &p), "npu-trunk: rows=16384 plan=[pipe 0..8192+8192..16384] predicted=-14.7%");
}
#[test] fn npu_trunk_record_roundtrip_and_validation() {
    let c = cal("g80bf16-measured.json");
    let path = std::env::temp_dir().join(format!("hipfire-npu-trunk-cal-{}.json", std::process::id()));
    c.store(&path).unwrap(); let loaded = Calibration::load(&path).unwrap(); std::fs::remove_file(path).unwrap();
    assert_eq!(serde_json::to_value(&c).unwrap(), serde_json::to_value(loaded).unwrap());
    let mut v = serde_json::to_value(&c).unwrap(); v.as_object_mut().unwrap().remove("margin");
    assert!(serde_json::from_value::<Calibration>(v).is_err());
    let mut invalid = c.clone(); invalid.schema_version = 2; assert!(invalid.validate().is_err());
    invalid = c.clone(); invalid.npu.panel_cols = 0; assert!(invalid.validate().is_err());
    invalid = c.clone(); invalid.npu.tops_corun = f64::NAN; assert!(invalid.validate().is_err());
}
#[test] fn npu_trunk_empty_partial_masks_and_padding() {
    let c = cal("g80bf16-measured.json");
    assert_eq!(npu_ms(&c, 8192, 1), npu_ms(&c, 8192, 1280));
    assert!(npu_ms(&c, 8192, 1281) > npu_ms(&c, 8192, 1280));
    for families in [FamilyMask::NONE, FamilyMask::from(Family::GdnZ)] {
        let p = plan(&c, families, 16384); assert_eq!(p.t_pipe_ms, p.t_gpu_ms);
        assert!(p.steps.iter().all(|s| matches!(s, Step::Gpu { .. })));
    }
    assert!(plan(&c, FamilyMask::ALL, 0).steps.is_empty());
}
