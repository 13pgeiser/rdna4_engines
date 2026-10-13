// SPDX-License-Identifier: Apache-2.0
//! Python↔Rust cost-model golden vectors, including the original 90 plans.
use super::*;
use serde_json::{json, Value};
use std::path::PathBuf;
fn fixture(name: &str) -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/npu_trunk").join(name) }
fn cal(name: &str) -> Calibration {
    // Preserve legacy lab fixtures; explicitly adapt identity and zero-cost
    // direct consume so the original 90 cost vectors remain unchanged.
    let mut v: Value = serde_json::from_slice(&std::fs::read(fixture(name)).unwrap()).unwrap();
    v["schema_version"] = json!(CALIBRATION_SCHEMA_VERSION);
    // Python's legacy get('panel_cols', 1), not a production input default.
    if v["npu"].get("panel_cols").is_none() { v["npu"]["panel_cols"] = json!(1); }
    v["margin"] = json!(0);
    v["key"] = json!({"device":{"arch":"gfx1151","pci_bdf":"0000:c0:00.0","npu_fw_version":"fixture"},"build":"fixture","kernel":{"uname_release":"fixture"},"model_index_sha256":"0000000000000000000000000000000000000000000000000000000000000000","epilogue":v["npu"]["epilogue"],"trunk_route":"stock"});
    v["glue"]["consume_mode"] = json!("direct");
    v["glue"]["consume_ms_per_col_8192"] = json!({"bf16":2.85/16480.0,"f32":4.60/16480.0});
    for name in ["qkv", "zab", "q", "kvidx"] {
        v["families"][name]["out_dtype"] = json!(if name == "qkv" { "bf16" } else { "f32" });
    }
    let c: Calibration = serde_json::from_value(v).unwrap(); c.validate().unwrap(); c
}
fn mask(names: &str) -> FamilyMask {
    if names.contains("zab") { FamilyMask::ALL }
    else { FamilyMask::from(Family::GdnQkv) | FamilyMask::QSA }
}
fn golden(names: &str) {
    let vectors: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/npu_trunk/golden.json"))).unwrap();
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
    assert_eq!(count, 27);
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
        ("kernel.uname_release", |k| k.kernel.uname_release += "x"),
        ("model_index_sha256", |k| k.model_index_sha256 += "x"), ("epilogue", |k| k.epilogue += "x"),
        ("trunk_route", |k| k.trunk_route += "x"),
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
    let mut invalid = c.clone(); invalid.schema_version = 1; assert!(invalid.validate().is_err());
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

fn grouped_mask(names: &Value) -> FamilyMask {
    names.as_array().unwrap().iter().fold(FamilyMask::NONE, |mask, name| mask | match name.as_str().unwrap() {
        "qkv" => FamilyMask::from(Family::GdnQkv),
        "zab" => FamilyMask::from_bits(0x0e),
        "q" => FamilyMask::from(Family::QsaQ),
        "kvidx" => FamilyMask::from_bits(0xe0),
        _ => panic!("unknown golden family"),
    })
}
fn close(actual: f64, expected: &Value) {
    let expected = expected.as_f64().unwrap();
    assert!((actual - expected).abs() <= expected.abs().max(1.0) * 1e-10, "{actual} != {expected}");
}
#[test]
fn npu_trunk_v2_consume_and_selection_goldens() {
    let vectors: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/npu_trunk/golden-v2.json"))).unwrap();
    for v in vectors.as_array().unwrap() {
        let cal: Calibration = serde_json::from_value(v["calibration"].clone()).unwrap();
        cal.validate().unwrap();
        let (families, report) = select_families(&cal);
        assert_eq!(families, grouped_mask(&v["selected"]), "{}", v["name"]);
        assert_eq!(report.best_families, grouped_mask(&v["best_families"]), "{}", v["name"]);
        assert_eq!(report.subsets.len(), 15);
        for (actual, expected) in report.subsets.iter().zip(v["subsets"].as_array().unwrap()) {
            assert_eq!(actual.families, grouped_mask(&expected["families"]));
            assert_eq!(actual.beats_margin_16384, expected["beats_margin_16384"].as_bool().unwrap());
            close(actual.mean_relative, &expected["mean_relative"]);
            for i in 0..3 {
                close(actual.t_pipe_ms[i], &expected["t_pipe_ms"][i]);
                close(actual.t_gpu_ms[i], &expected["t_gpu_ms"][i]);
                close(actual.relative[i], &expected["relative"][i]);
            }
        }
        for expected in v["bank_segments"].as_array().unwrap() {
            let actual = bank_segments(&cal, grouped_mask(&expected["families"]), 4096, 8192);
            for i in 0..2 {
                close(actual.gpu[i], &expected["gpu"][i]);
                close(actual.npu[i], &expected["npu"][i]);
            }
        }
        for family_plans in v["plans"].as_array().unwrap() {
            let mask = grouped_mask(&family_plans["families"]);
            for expected in family_plans["plans"].as_array().unwrap() {
                let actual = plan(&cal, mask, expected["rows"].as_u64().unwrap() as usize);
                let steps: Vec<_> = expected["steps"].as_array().unwrap().iter().map(|s| {
                    let range = |r: &Value| r[0].as_u64().unwrap() as usize..r[1].as_u64().unwrap() as usize;
                    if let Some(r) = s.get("gpu") { Step::Gpu { chunk: range(r) } }
                    else { Step::Pipe { a: range(&s["pipe"][0]), b: range(&s["pipe"][1]) } }
                }).collect();
                assert_eq!(actual.steps, steps, "{} {}", v["name"], expected["python_plan"]);
                close(actual.t_pipe_ms, &expected["t_pipe_ms"]);
                close(actual.t_gpu_ms, &expected["t_gpu_ms"]);
            }
        }
    }
}
#[test]
fn npu_trunk_v2_consume_is_gpu_work_on_live_columns() {
    let mut c = cal("g80bf16-measured.json");
    let direct = bank_segments(&c, FamilyMask::ALL, 8192, 0);
    c.glue.consume_mode = "retile".into();
    let retile = bank_segments(&c, FamilyMask::ALL, 8192, 0);
    let expected_gdn = c.glue.consume_ms_per_col_8192["bf16"] * c.families["qkv"].cols as f64
        + c.glue.consume_ms_per_col_8192["f32"] * c.families["zab"].cols as f64;
    let expected_qsa = c.glue.consume_ms_per_col_8192["f32"] * (c.families["q"].cols + c.families["kvidx"].cols) as f64;
    assert!((retile.gpu[0] - direct.gpu[0] - expected_gdn).abs() < 1e-10);
    assert!((retile.gpu[1] - direct.gpu[1] - expected_qsa).abs() < 1e-10);
    assert_eq!(retile.npu, direct.npu);
    let half = bank_segments(&c, FamilyMask::ALL, 4096, 0);
    assert!((half.gpu[0] * 2.0 - retile.gpu[0]).abs() < 1e-10);
}
#[test]
fn npu_trunk_v2_rejects_missing_invalid_fields() {
    let c = cal("g80bf16-measured.json");
    for pointer in ["/key/trunk_route", "/glue/consume_mode", "/glue/consume_ms_per_col_8192", "/families/qkv/out_dtype"] {
        let mut v = serde_json::to_value(&c).unwrap();
        let (parent, field) = pointer.rsplit_once('/').unwrap();
        v.pointer_mut(parent).unwrap().as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<Calibration>(v).is_err(), "{pointer}");
    }
    for (pointer, value) in [
        ("/key/trunk_route", json!("overlay:")),
        ("/key/trunk_route", json!("other")),
        ("/glue/consume_mode", json!("other")),
        ("/glue/consume_ms_per_col_8192/bf16", json!(-1.0)),
        ("/families/qkv/out_dtype", json!("f16")),
    ] {
        let mut v = serde_json::to_value(&c).unwrap();
        *v.pointer_mut(pointer).unwrap() = value;
        assert!(serde_json::from_value::<Calibration>(v).unwrap().validate().is_err(), "{pointer}");
    }
    for dtype in ["bf16", "f32"] {
        let mut invalid = c.clone();
        invalid.glue.consume_ms_per_col_8192.remove(dtype);
        assert!(invalid.validate().is_err());
        let mut invalid = c.clone();
        invalid.glue.consume_ms_per_col_8192.insert(dtype.into(), f64::NAN);
        assert!(invalid.validate().is_err());
    }
    let mut invalid = c.clone();
    invalid.glue.consume_ms_per_col_8192.insert("f16".into(), 0.0);
    assert!(invalid.validate().is_err());
    let mut key = c.key.clone();
    key.epilogue = "other".into();
    key.trunk_route = "overlay:gfx1151-zab-v1".into();
    assert_eq!(c.key_matches(&key), Err("calibration key mismatch: epilogue".into()));
    key.epilogue = c.key.epilogue.clone();
    assert_eq!(c.key_matches(&key), Err("calibration key mismatch: trunk_route".into()));
}

#[test]
fn npu_trunk_runtime_key_is_cheap_deterministic_metadata() {
    let digest = "8703bb1626ff48b976a84d5f4f7d18b4aad8e9d1d9b780a52f963d73cb83c013";
    let key = calibration_key("gfx1151", "0000:C0:00.0", "1.1.2.65\n", "7.3.0-rc7-perfopt+\n", digest, "overlay:gfx1151-zab-v1");
    assert_eq!(key.device.pci_bdf, "0000:c0:00.0");
    assert_eq!(key.device.npu_fw_version, "1.1.2.65");
    assert_eq!(key.kernel.uname_release, "7.3.0-rc7-perfopt+");
    assert_eq!(key.build, "hipfire-0.4.2:npu-trunk-bf16-rowmajor-v1");
    assert_eq!(key.model_index_sha256, digest);
    assert_eq!(key.trunk_route, "overlay:gfx1151-zab-v1");
    assert_eq!(key, calibration_key("gfx1151", "0000:c0:00.0", "1.1.2.65", "7.3.0-rc7-perfopt+", digest, "overlay:gfx1151-zab-v1"));
}

#[test]
fn npu_trunk_auto_admission_matrix() {
    let c = Calibration::load(fixture("g80bf16-stock-v2.json")).unwrap();
    assert_eq!(auto_admission(&c, &c.key), Ok(FamilyMask::ALL));
    let mut wrong = c.key.clone();
    wrong.device.arch = "gfx1201".into();
    assert_eq!(auto_admission(&c, &wrong), Err("calibration key mismatch: device.arch".into()));
    let mut subset = c.clone();
    subset.gpu.trunk_in_ms_per_8192.insert("zab".into(), 0.0);
    assert_eq!(select_families(&subset).0, FamilyMask::from(Family::GdnQkv) | FamilyMask::QSA);
    assert_eq!(auto_admission(&subset, &subset.key), Ok(FamilyMask::from(Family::GdnQkv) | FamilyMask::QSA));
    let mut none = c.clone();
    none.npu.tops_corun = 0.001;
    assert_eq!(select_families(&none).0, FamilyMask::NONE);
    assert_eq!(auto_admission(&none, &none.key), Err("calibration selects no families".into()));
}

#[test]
fn npu_trunk_auto_maps_only_planned_natural_pairs() {
    for file in ["g80bf16-stock-v2.json", "g80bf16-overlay-v2.json"] {
        let c = Calibration::load(fixture(file)).unwrap();
        let star = crossover(&c, FamilyMask::ALL);
        for (rows, expected) in [
            (8000, vec![]), (9000, vec![]), (16343, vec![(0, 16343)]),
            (20000, vec![(0, 16384)]),
            (57544, vec![(0, 16384), (16384, 16384), (32768, 16384)]),
            (65536, vec![(0, 16384), (16384, 16384), (32768, 16384), (49152, 16384)]),
        ] {
            let planned = plan(&c, FamilyMask::ALL, rows);
            let executable = executable_plan(&c, &planned, rows, star);
            let pairs: Vec<_> = executable.steps.iter().filter_map(|step| match step {
                Step::Pipe { a, b } => Some((a.start, b.end - a.start)),
                _ => None,
            }).collect();
            assert_eq!(pairs, expected, "{file} {rows}");
            let mut at = 0;
            for step in &executable.steps {
                let (start, end) = match step {
                    Step::Gpu { chunk } => (chunk.start, chunk.end),
                    Step::Pipe { a, b } => {
                        assert_eq!(planned_pair_rows(&executable, a.start), Some(b.end - a.start));
                        assert_eq!(a.len(), CHUNK);
                        assert!(b.len() >= 512.max(c.gpu.min_bank_rows));
                        (a.start, b.end)
                    }
                };
                assert_eq!(start, at); at = end;
            }
            assert_eq!(at, rows);
        }
        let p = plan(&c, FamilyMask::ALL, 16384);
        for cutoff in [None, Some(16385)] {
            assert!(executable_plan(&c, &p, 16384, cutoff).steps.iter().all(|s| matches!(s, Step::Gpu { .. })));
        }
        let mut strict = c.clone(); strict.gpu.min_bank_rows = 8192;
        let p = plan(&c, FamilyMask::ALL, 16343);
        assert!(executable_plan(&strict, &p, 16343, Some(0)).steps.iter().all(|s| matches!(s, Step::Gpu { .. })));
        let rebank = Plan { steps: vec![Step::Pipe { a: 0..5000, b: 5000..10000 },
            Step::Pipe { a: 10000..15000, b: 15000..20000 }], t_gpu_ms: 1.0, t_pipe_ms: 0.5 };
        assert!(executable_plan(&c, &rebank, 20000, Some(0)).steps.iter().all(|s| matches!(s, Step::Gpu { .. })));
    }
}
