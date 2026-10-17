//! Declared ABI per kernel symbol: HIP source declarations (as the JIT sees
//! them, including Rust-side `#define` prefixes) and PeaceMaker image metadata.

use crate::hip::{decls_of, Arg, Decl};
use crate::lex::*;
use crate::scan::*;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use peacemaker_ir::{envelope::Source, inst::Arch};
use peacemaker_lift::bundle::SourceCodec;

pub type DeclList = Vec<(String, Decl)>;

#[derive(Default)]
pub struct Symbols {
    pub hip: BTreeMap<String, DeclList>,
    pub pm: BTreeMap<String, DeclList>,
    /// Constants whose text is (derived from) a changed HIP source.
    pub changed_consts: BTreeSet<(String, String)>,
    /// Changed constants whose text could not be evaluated (must stay empty).
    pub unevaluable: Vec<(String, String)>,
    /// Origin string -> symbols it declares (`const:NAME@file`, `file:path`, `image:path`).
    pub origin_symbols: BTreeMap<String, Vec<String>>,
    /// PM image -> (kernarg_segment_size per symbol).
    pub pm_segment: BTreeMap<(String, String), u32>,
}

impl Symbols {
    pub fn all_names(&self) -> BTreeSet<String> {
        self.hip.keys().chain(self.pm.keys()).cloned().collect()
    }

    pub fn decls(&self, name: &str) -> DeclList {
        let mut v = self.hip.get(name).cloned().unwrap_or_default();
        v.extend(self.pm.get(name).cloned().unwrap_or_default());
        v
    }

    pub fn is_changed_const(&self, name: &str, file: &str) -> bool {
        self.changed_consts
            .contains(&(name.to_string(), file.to_string()))
    }
}

pub fn const_origin(name: &str, file: &str) -> String {
    format!("const:{name}@{file}")
}

/// Index every changed HIP source: constants derived from them (through
/// `include_str!`/`concat!`/const references) and the raw files.
pub fn build_hip(
    world: &World,
    root: &Path,
    changed_files: &[String],
    old_consts: &BTreeMap<String, String>,
    tables: &BTreeMap<String, String>,
    syms: &mut Symbols,
) -> Result<(), String> {
    let base: BTreeSet<String> = changed_files
        .iter()
        .map(|f| {
            Path::new(f)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string()
        })
        .collect();
    // Direct: include_str! of a changed file.
    for f in &world.files {
        for (name, defs) in &f.consts {
            for d in defs {
                let mut lits = Vec::new();
                strings_in(&d.tokens, &mut lits);
                if lits.iter().any(|s| {
                    s.ends_with(".hip") && base.contains(s.rsplit('/').next().unwrap_or(""))
                }) {
                    syms.changed_consts.insert((name.clone(), f.path.clone()));
                }
            }
        }
    }
    // Historical OLD_* source file names are not current source: classify per const.
    for key in old_consts.keys() {
        let (file, name) = split_key(key)?;
        if !syms.is_changed_const(name, file) {
            return Err(format!(
                "old_consts {key}: not a const naming a changed .hip file (stale entry)"
            ));
        }
        verify_old_const(world, name, file)?;
        syms.changed_consts
            .remove(&(name.to_string(), file.to_string()));
    }
    // Transitive: const expressions over changed consts.
    loop {
        let mut added = false;
        for f in &world.files {
            for (name, defs) in &f.consts {
                if defs.len() != 1 || syms.is_changed_const(name, &f.path) {
                    continue;
                }
                let mut ids = Vec::new();
                idents_in(&defs[0].init(), &mut ids);
                if ids.iter().any(|i| {
                    i != name
                        && (syms.is_changed_const(i, &f.path)
                            || syms.is_changed_const(i, KERNELS_RS))
                }) {
                    syms.changed_consts.insert((name.clone(), f.path.clone()));
                    added = true;
                }
            }
        }
        if !added {
            break;
        }
    }
    let consts: Vec<(String, String)> = syms.changed_consts.iter().cloned().collect();
    let mut used_tables: BTreeSet<String> = BTreeSet::new();
    for (name, file) in consts {
        match world.const_text(&name, &file) {
            None => {
                let key = format!("{file}::{name}");
                if tables.contains_key(&key) {
                    verify_table_const(world, syms, &name, &file)?;
                    used_tables.insert(key);
                } else {
                    syms.unevaluable.push((name, file));
                }
            }
            Some(text) => {
                if !text.contains("__global__") {
                    continue;
                }
                let found = decls_of(&text).map_err(|e| format!("const {name} ({file}): {e}"))?;
                let origin = const_origin(&name, &file);
                for (sym, d) in found {
                    syms.origin_symbols
                        .entry(origin.clone())
                        .or_default()
                        .push(sym.clone());
                    syms.hip.entry(sym).or_default().push((origin.clone(), d));
                }
            }
        }
    }
    for key in tables.keys() {
        if !used_tables.contains(key) {
            return Err(format!(
                "source_tables {key}: not an unevaluable changed const (stale entry)"
            ));
        }
    }
    for f in changed_files {
        let text = std::fs::read_to_string(root.join(f)).map_err(|e| format!("{f}: {e}"))?;
        let found = decls_of(&text).map_err(|e| format!("{f}: {e}"))?;
        let origin = format!("file:{f}");
        for (sym, d) in found {
            syms.origin_symbols
                .entry(origin.clone())
                .or_default()
                .push(sym.clone());
            syms.hip.entry(sym).or_default().push((origin.clone(), d));
        }
    }
    Ok(())
}

fn split_key(key: &str) -> Result<(&str, &str), String> {
    key.rsplit_once("::")
        .ok_or_else(|| format!("inventory key `{key}` is not `file::NAME`"))
}

/// An `OLD_*` constant names a historical kernel source FILE for the baseline
/// loader; it is not current JIT source. Verified, not assumed: a plain
/// `OldFile { file: "<name>.hip", syms: .. }` literal (no include_str!/concat!),
/// the loader renames every symbol with the `__beta` suffix before compiling,
/// and every use in the file is either `load_old(&NAME)`, `&NAME` in a list, or
/// the `NAME.syms` field.
fn verify_old_const(world: &World, name: &str, file: &str) -> Result<(), String> {
    let key = format!("{file}::{name}");
    let f = world
        .file(file)
        .ok_or_else(|| format!("old_consts {key}: file not scanned"))?;
    if !name.starts_with("OLD_") {
        return Err(format!("old_consts {key}: not an OLD_* constant"));
    }
    let defs = f
        .consts
        .get(name)
        .filter(|d| d.len() == 1)
        .ok_or_else(|| format!("old_consts {key}: not a unique const"))?;
    let init = defs[0].init();
    let mut ids = Vec::new();
    idents_in(&init, &mut ids);
    if ids.iter().any(|i| {
        matches!(
            i.as_str(),
            "include_str" | "concat" | "format" | "env" | "stringify"
        )
    }) {
        return Err(format!("old_consts {key}: builds source text (include_str!/concat!/format!); it must be a plain file-name literal"));
    }
    let mut lits = Vec::new();
    strings_in(&init, &mut lits);
    let hips: Vec<&String> = lits.iter().filter(|s| s.ends_with(".hip")).collect();
    if hips.len() != 1 || hips[0].contains('/') {
        return Err(format!(
            "old_consts {key}: expected exactly one bare `.hip` file-name literal, found {hips:?}"
        ));
    }
    // Loader contract.
    let suffix = f
        .consts
        .get("OLD_SUFFIX")
        .filter(|d| d.len() == 1)
        .ok_or_else(|| format!("old_consts {key}: no OLD_SUFFIX const"))?;
    let mut sl = Vec::new();
    strings_in(&suffix[0].init(), &mut sl);
    if sl != ["__beta".to_string()] {
        return Err(format!(
            "old_consts {key}: OLD_SUFFIX is {sl:?}, expected \"__beta\""
        ));
    }
    let fn_ids = |fname: &str| -> Option<(Vec<String>, Vec<String>)> {
        let func = f.fns.iter().find(|x| x.name == fname)?;
        let mut i = Vec::new();
        idents_in(&func.body, &mut i);
        let mut s = Vec::new();
        strings_in(&func.body, &mut s);
        Some((i, s))
    };
    let (_, prelude_strs) =
        fn_ids("old_prelude").ok_or_else(|| format!("old_consts {key}: no old_prelude fn"))?;
    if !prelude_strs
        .iter()
        .any(|s| s.starts_with("#define {s} {s}{OLD_SUFFIX}"))
    {
        return Err(format!(
            "old_consts {key}: old_prelude no longer renames every symbol with OLD_SUFFIX"
        ));
    }
    let load = f
        .fns
        .iter()
        .find(|x| x.name == "load_old")
        .ok_or_else(|| format!("old_consts {key}: no load_old fn"))?;
    let mut load_ids = Vec::new();
    idents_in(&load.body, &mut load_ids);
    for need in ["old_prelude", "old_source"] {
        if !load_ids.iter().any(|i| i == need) {
            return Err(format!("old_consts {key}: load_old no longer uses {need}"));
        }
    }
    // The kernels it compiles must be NAMED with the suffix, from the prelude'd source.
    let idx = BindIndex::build(&load.body);
    let load_params = fn_params(load);
    let mut compiles: Vec<Vec<Vec<Tt>>> = Vec::new();
    calls_of(&load.body, "ensure_kernel_public", &mut compiles);
    let flows = !compiles.is_empty()
        && compiles.iter().all(|a| {
            a.len() == 3
                && name_ends_with_const(&a[2], "OLD_SUFFIX", &idx, &load_params, 0)
                && expr_uses_fn(&a[1], "old_prelude", &idx, 0)
        });
    if !flows {
        return Err(format!(
            "old_consts {key}: load_old no longer compiles its kernels under `<sym>{{OLD_SUFFIX}}` from old_prelude'd source"
        ));
    }
    let mut uses = 0usize;
    for func in &f.fns {
        check_old_uses(&func.body, name, None, '{', &mut uses)
            .map_err(|e| format!("old_consts {key}: {} in `{}`", e, func.name))?;
    }
    if uses == 0 {
        return Err(format!("old_consts {key}: never used (stale entry)"));
    }
    Ok(())
}

fn check_old_uses(
    ts: &[Tt],
    name: &str,
    call: Option<&str>,
    open: char,
    n: &mut usize,
) -> Result<(), String> {
    for k in 0..ts.len() {
        match &ts[k] {
            Tt::Id(s, _) if s == name => {
                *n += 1;
                let by_ref = k > 0 && ts[k - 1].is_p('&');
                let syms_field = ts.get(k + 1).map_or(false, |t| t.is_p('.'))
                    && ts.get(k + 2).map_or(false, |t| t.is_id("syms"));
                if !((by_ref && (call == Some("load_old") || open == '[')) || syms_field) {
                    return Err(format!("use of {name} that is neither load_old(&{name}), a `&{name}` list entry nor {name}.syms"));
                }
            }
            Tt::G(d, c, _) => {
                let (callee, o) = if *d == '(' {
                    (if k > 0 { ts[k - 1].id() } else { None }, '(')
                } else if *d == '{' {
                    (call, open)
                } else {
                    (None, *d)
                };
                check_old_uses(c, name, callee, o, n)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// A const table (e.g. `VARIANTS`) is not source text: it carries kernel names
/// and references the source consts. Verified: no source-building macro, and
/// every changed source const it references (directly or through same-file
/// const fns it calls) is itself text-evaluable and therefore indexed.
fn verify_table_const(world: &World, syms: &Symbols, name: &str, file: &str) -> Result<(), String> {
    let key = format!("{file}::{name}");
    let f = world
        .file(file)
        .ok_or_else(|| format!("source_tables {key}: file not scanned"))?;
    let defs = f
        .consts
        .get(name)
        .filter(|d| d.len() == 1)
        .ok_or_else(|| format!("source_tables {key}: not a unique const"))?;
    let mut ids = Vec::new();
    idents_in(&defs[0].init(), &mut ids);
    let mut all: Vec<String> = ids.clone();
    for id in &ids {
        for func in f.fns.iter().filter(|x| &x.name == id) {
            idents_in(&func.body, &mut all);
        }
    }
    if all
        .iter()
        .any(|i| matches!(i.as_str(), "include_str" | "concat" | "format" | "env"))
    {
        return Err(format!(
            "source_tables {key}: builds source text itself; it must only reference source consts"
        ));
    }
    let mut referenced = 0usize;
    for id in &all {
        if id != name && syms.is_changed_const(id, file) {
            if world.const_text(id, file).is_none() {
                return Err(format!("source_tables {key}: references changed const `{id}` whose text cannot be evaluated"));
            }
            referenced += 1;
        }
    }
    if referenced == 0 {
        return Err(format!(
            "source_tables {key}: references no changed source const (stale entry)"
        ));
    }
    Ok(())
}

fn arch_of(path: &str) -> Result<Arch, String> {
    for (suffix, arch) in [
        ("gfx1151", Arch::Gfx1151),
        ("gfx1201", Arch::Gfx1201),
        ("gfx1100", Arch::Gfx1100),
        ("gfx1030", Arch::Gfx1030),
    ] {
        if path.contains(suffix) {
            return Ok(arch);
        }
    }
    Err(format!(
        "{path}: cannot infer the image architecture from its name"
    ))
}

/// Index PeaceMaker images by their embedded HSA metadata.
pub fn build_pm(root: &Path, images: &[String], syms: &mut Symbols) -> Result<(), String> {
    for path in images {
        let bytes = std::fs::read(root.join(path)).map_err(|e| format!("{path}: {e}"))?;
        let arch = arch_of(path)?;
        let (_, kernels) = Source::read(&bytes, arch).map_err(|e| format!("{path}: {e:?}"))?;
        let origin = format!("image:{path}");
        for k in kernels {
            let m = &k.metadata.parsed;
            let mut args: Vec<Arg> = Vec::new();
            for a in m
                .args
                .iter()
                .filter(|a| !a.value_kind.starts_with("hidden_"))
            {
                args.push(Arg {
                    offset: a.offset as u32,
                    size: a.size as u32,
                    kind: if a.value_kind == "global_buffer" {
                        'p'
                    } else {
                        'v'
                    },
                    text: format!("{}:{}", a.name, a.value_kind),
                });
            }
            let end = args.iter().map(|a| a.offset + a.size).max().unwrap_or(0);
            syms.pm_segment.insert(
                (path.clone(), k.name.clone()),
                m.kernarg_segment_size as u32,
            );
            syms.origin_symbols
                .entry(origin.clone())
                .or_default()
                .push(k.name.clone());
            syms.pm
                .entry(k.name.clone())
                .or_default()
                .push((origin.clone(), Decl { args, end }));
        }
    }
    Ok(())
}

/// Argument lists of every call `name(..)` in `ts` (nested groups included).
pub fn calls_of(ts: &[Tt], name: &str, out: &mut Vec<Vec<Vec<Tt>>>) {
    for k in 0..ts.len() {
        if let Tt::G(d, c, _) = &ts[k] {
            if *d == '(' && k > 0 && ts[k - 1].is_id(name) {
                out.push(split_top(c, ','));
            }
            calls_of(c, name, out);
        }
    }
}

fn strip_refs(ts: &[Tt]) -> Vec<Tt> {
    let mut t: Vec<Tt> = ts.to_vec();
    while t.first().map_or(false, |x| x.is_p('&')) {
        t.remove(0);
    }
    t
}

/// Does this expression build a name that ENDS with the value of the const `konst`?
/// Only `format!("..{KONST}")` / `format!("{}{}", .., KONST)` count (a plain mention of
/// the const elsewhere in the function does not), following a single local binding.
/// Anything that could change which value the placeholder prints is refused: a named
/// format argument `KONST = ..`, or a local/parameter that shadows the const.
pub fn name_ends_with_const(
    ts: &[Tt],
    konst: &str,
    idx: &BindIndex,
    params: &std::collections::HashSet<String>,
    depth: u32,
) -> bool {
    if params.contains(konst) || idx.map.contains_key(konst) {
        return false;
    }
    let t = strip_refs(ts);
    if t.len() == 1 {
        if let Some(id) = t[0].id() {
            if depth < 3 {
                if let Some(bs) = idx.map.get(id) {
                    return bs.len() == 1
                        && name_ends_with_const(&bs[0].init, konst, idx, params, depth + 1);
                }
            }
            return false;
        }
    }
    if t.len() == 3 && t[0].is_id("format") && t[1].is_p('!') {
        if let Some(g) = t[2].group() {
            let parts = split_top(g, ',');
            let overridden = parts.iter().skip(1).any(|p| {
                p.len() >= 2
                    && p[0].is_id(konst)
                    && p[1].is_p('=')
                    && !p.get(2).map_or(false, |x| x.is_p('='))
            });
            if overridden {
                return false;
            }
            if let Some([Tt::S(tpl, _)]) = parts.first().map(|p| p.as_slice()) {
                if tpl.ends_with(&format!("{{{konst}}}")) {
                    return true;
                }
                if tpl.ends_with("{}") && parts.len() >= 2 {
                    let last = parts.last().unwrap();
                    return last.len() == 1 && last[0].is_id(konst);
                }
            }
        }
    }
    false
}

/// Does this expression (or the single local it names) call `func`?
pub fn expr_uses_fn(ts: &[Tt], func: &str, idx: &BindIndex, depth: u32) -> bool {
    let t = strip_refs(ts);
    let mut ids = Vec::new();
    idents_in(&t, &mut ids);
    if ids.iter().any(|i| i == func) {
        return true;
    }
    if t.len() == 1 && depth < 3 {
        if let Some(id) = t[0].id() {
            if let Some(bs) = idx.map.get(id) {
                return bs.len() == 1 && expr_uses_fn(&bs[0].init, func, idx, depth + 1);
            }
        }
    }
    false
}

/// The value the OLD helpers' `OLD_SUFFIX` placeholder really prints: one const in
/// the file whose only string is `__beta` (an empty or changed suffix would make the
/// "OLD" launch name the current kernel).
pub fn old_suffix_is_beta(world: &World, file: &str) -> Result<(), String> {
    let f = world
        .file(file)
        .ok_or_else(|| format!("{file}: file not scanned"))?;
    let defs = f
        .consts
        .get("OLD_SUFFIX")
        .filter(|d| d.len() == 1)
        .ok_or("no unique OLD_SUFFIX const")?;
    let mut lits = Vec::new();
    strings_in(&defs[0].init(), &mut lits);
    if lits != ["__beta".to_string()] {
        return Err(format!("OLD_SUFFIX is {lits:?}, expected \"__beta\""));
    }
    Ok(())
}
