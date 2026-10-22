//! The `qwen4_moe_sym` down entries as natives (a child module of [`super`]).
//!
//! Two layers over `peacemaker_isa::kernels::qwen4_moe_sym`, adding no typestate surface:
//!
//! - Coarse: `MoeSym::emit` / `MoeSym::emit_module` live in `native_gen` (the Rust functions'
//!   values).
//! - Blocks: `QDownGen::new(nt, gp, arch)` and the `qdown::*` natives of
//!   `qwen4_moe_sym::down_blocks`, for the gfx1201 MQ4 down NT4/NT8 entries, flat-slot
//!   (`gp = false`, `down_ntN`) and grouped-position (`gp = true`, `down_gp_ntN`). The down entries
//!   have no LDS and no ring, so there is nothing to lend: each block that needs the typed core
//!   (`prologue`, `compute`, `tail_pair`, `epilogue`, `trips_cmp`) runs in one `Driver::typed`
//!   call, and the rest (`load_w`, `advance`, `trips_dec`) emit straight into `wg.isa()`. The
//!   K-loop's trips, the order of weight loads and epoch folds in a trip, the tail and the exit are
//!   script statements.

use super::native_gen::{exit_of, typed, wg_arg};
use super::{
    arch_of, check_args, done, done_with, driver, nat, nat_of, spec_value, str_of, u8_of, usize_of,
    Nat,
};
use crate::interp::{Cx, Interp, R};
use crate::value::Value;
use peacemaker_isa::kernels::qwen4_moe_sym::{down_blocks as blocks, Kind, Spec, WeightCodec};
use std::cell::RefCell;
use std::rc::Rc;

type Res<T> = Result<T, String>;

const FNS: &[&str] = &[
    "QDownGen::new",
    "qdown::srd_x",
    "qdown::prologue",
    "qdown::trips_cmp",
    "qdown::trips_dec",
    "qdown::load_w",
    "qdown::compute",
    "qdown::compute_reload",
    "qdown::advance",
    "qdown::tail",
    "qdown::epilogue",
];

pub(super) fn is_fn(name: &str) -> bool {
    FNS.contains(&name)
}

pub(super) fn is_type(n: &str) -> bool {
    n == "QDownGen"
}

fn gen_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Rc<blocks::Gen>> {
    match nat_of(it, v) {
        Some(Nat::QDownGen(g)) => Ok(g),
        _ => Err("expected a QDownGen (`QDownGen::new(nt, gp, arch)?`)".into()),
    }
}

pub(super) fn call_fn<'a>(
    it: &Interp<'a>,
    name: &str,
    a: &[Value<'a>],
    cx: &mut Cx<'_>,
) -> Option<R<'a>> {
    is_fn(name).then(|| call(it, name, a, cx))
}

fn call<'a>(it: &Interp<'a>, name: &str, a: &[Value<'a>], cx: &mut Cx<'_>) -> R<'a> {
    let need = |n: usize| check_args(name, a, n);
    match name {
        "QDownGen::new" => {
            need(3)?;
            let nt = u8_of(it, &a[0], "tile width nt")?;
            let gp = match it.deref(a[1].clone()) {
                Value::Bool(b) => b,
                o => {
                    return Err(format!(
                        "grouped-position flag: expected a bool, found {}",
                        o.type_name()
                    )
                    .into())
                }
            };
            let arch = arch_of(it, &a[2])?;
            let spec = Spec {
                arch,
                kind: Kind::Down,
                nt,
                rr: 1,
                codec: WeightCodec::Mq4Sym,
            };
            Ok(done_with(blocks::Gen::new(spec, gp), |g| {
                nat(Nat::QDownGen(Rc::new(g)))
            }))
        }
        "qdown::srd_x" => {
            need(1)?;
            Ok(done_with(
                blocks::srd_x(usize_of(it, &a[0], "parity")?),
                |r| Value::u8(r),
            ))
        }
        "qdown::prologue" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let (g, exit) = (gen_of(it, &a[1])?, exit_of(it, &a[2])?);
            Ok(done(typed(cx, |wg, lend| {
                let end = lend.end(&exit)?;
                blocks::prologue(wg, &g, &end)
            })?))
        }
        "qdown::trips_cmp" => {
            need(2)?;
            wg_arg(it, &a[0])?;
            let cmp = str_of(it, &a[1], "trips_cmp")?;
            let c = typed(
                cx,
                |wg, lend| Ok(lend.wg_cond(blocks::trips_cmp(wg, &cmp)?)),
            )?;
            Ok(done_with(c, |c| nat(Nat::WgCond(c))))
        }
        "qdown::trips_dec" => {
            need(1)?;
            wg_arg(it, &a[0])?;
            Ok(done(blocks::trips_dec(driver(cx)?.isa())))
        }
        "qdown::load_w" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let (g, p) = (gen_of(it, &a[1])?, usize_of(it, &a[2], "epoch parity")?);
            Ok(done(blocks::load_w(driver(cx)?.isa(), &g, p)))
        }
        "qdown::advance" => {
            need(1)?;
            wg_arg(it, &a[0])?;
            Ok(done(blocks::advance(driver(cx)?.isa())))
        }
        "qdown::compute" | "qdown::compute_reload" => {
            let reload = name == "qdown::compute_reload";
            need(if reload { 5 } else { 4 })?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let (p, site) = (
                usize_of(it, &a[2], "epoch parity")?,
                str_of(it, &a[3], "site label")?,
            );
            let srd = if reload {
                Some(u8_of(it, &a[4], "reload descriptor")?)
            } else {
                None
            };
            Ok(done(typed(cx, |wg, _| {
                blocks::compute(wg, &g, p, &site, srd)
            })?))
        }
        "qdown::tail" => {
            need(2)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            Ok(done(typed(cx, |wg, _| blocks::tail(wg, &g))?))
        }
        "qdown::epilogue" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let (g, exit) = (gen_of(it, &a[1])?, exit_of(it, &a[2])?);
            Ok(done(typed(cx, |wg, lend| {
                let end = lend.end(&exit)?;
                blocks::epilogue(wg, &g, &end)
            })?))
        }
        n => Err(format!("native function `{n}` is not implemented").into()),
    }
}

/// Methods of a `QDownGen`: `plan()`, `kernel_spec()`, `nt()`, `gp()`, `label(name)`,
/// `exit_label()`.
pub(super) fn method<'a>(
    it: &Interp<'a>,
    g: &Rc<blocks::Gen>,
    name: &str,
    a: &[Value<'a>],
) -> R<'a, Option<Value<'a>>> {
    let need = |n: usize| check_args(name, a, n);
    Ok(match name {
        "plan" => {
            need(0)?;
            Some(done_with(g.plan(), |p| {
                nat(Nat::Plan(Rc::new(RefCell::new(p))))
            }))
        }
        "kernel_spec" => {
            need(0)?;
            Some(spec_value(&g.kernel_spec()))
        }
        "nt" => {
            need(0)?;
            Some(Value::u8(g.spec().nt))
        }
        "gp" => {
            need(0)?;
            Some(Value::Bool(g.gp()))
        }
        "label" => {
            need(1)?;
            Some(Value::str(g.label(&str_of(it, &a[0], "label")?)))
        }
        "exit_label" => {
            need(0)?;
            Some(Value::str(g.exit_label()))
        }
        _ => None,
    })
}
