// SPDX-License-Identifier: Apache-2.0
//! The admission barrier keeps all weight/device work behind cheap checks.
pub(crate) fn prepare_after_checks<D, C, T, E>(
    eligible: impl FnOnce() -> Result<(), E>,
    device: impl FnOnce() -> Result<D, E>,
    calibration: impl FnOnce(D) -> Result<C, E>,
    prepare: impl FnOnce(C) -> Result<T, E>,
) -> Result<T, E> {
    eligible()?;
    let device = device()?;
    let calibration = calibration(device)?;
    prepare(calibration)
}

/// Auto refusals are GPU-only, including preparation/open errors. Forced modes fail.
pub(crate) fn finish<T, E: std::fmt::Display>(mode: super::NpuTrunkMode, result: Result<Option<T>, E>) -> Result<Option<T>, E> {
    match result {
        Err(reason) if mode == super::NpuTrunkMode::Auto => {
            let rendered = reason.to_string();
            let reason = rendered.split_once("npu-trunk: ").map_or(rendered.as_str(), |(_, reason)| reason);
            eprintln!("npu-trunk: off ({reason})");
            Ok(None)
        }
        result => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    #[test]
    fn checks_precede_download_pack_alloc_and_open() {
        for failed in 0..=4 {
            let calls = RefCell::new(Vec::new());
            let stage = |n| { calls.borrow_mut().push(n); if n == failed { Err("refused") } else { Ok(()) } };
            let result = prepare_after_checks(|| stage(0), || stage(1), |_| stage(2), |_| {
                stage(3)?; // download, pack and allocate only beyond the barrier
                stage(4)  // runtime open
            });
            assert!(result.is_err());
            assert_eq!(*calls.borrow(), (0..=failed).collect::<Vec<_>>());
            assert!(finish(super::super::NpuTrunkMode::Auto, result.map(Some)).unwrap().is_none());
        }
        assert!(finish::<(), _>(super::super::NpuTrunkMode::On, Err("open failed")).is_err());
    }
}
