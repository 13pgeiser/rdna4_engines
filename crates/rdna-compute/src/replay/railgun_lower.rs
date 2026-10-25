// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! railgun's gfx12 lowering transport, extracted from the M1/M2 shadow
//! (`railgun_shadow::lower`) so a *directly authored* program can use it
//! without any Redline recording (design A §1 "Its emission transport can be
//! reused; its recorder/planner must not be used for this feature", slice
//! A1).
//!
//! * [`emit_segment`] is the one Pm4Op→command emission loop. The legacy M1
//!   shadow (`railgun_shadow::lower`) now calls it with the closure that does
//!   what its inline loop did, so the legacy AR/M2 path emits the same
//!   dwords in the same order.
//! * [`lower_owned`] lowers a single-segment, all-PM4 [`rk::KernelProgram`]
//!   with railgun's own kernarg images and owns everything the queue reads:
//!   loaded kernels, kernarg buffers and the retained indirect buffer. It
//!   needs no `RecordedHipLaunch`, `ReplayBindings` or Redline kernarg
//!   buffer. [`OwnedWindowLowering`] is the GPU-facing half of
//!   `railgun::window::PreparedWindow`: the integration owner implements
//!   `railgun::window::WindowQueue` over it (write images, patch grids,
//!   publish, replay).

use railgun::kernel as rk;
use railgun::plan::{self as rp, Pm4Op, Rung};

use super::railgun_shadow::Transport;
use super::{
    Gfx12DispatchPacing, GpuDevice, KernargBuffer, Kernel, LaunchGeometry, Pm4Commands,
    ReplayQuiescence, SingleQueuePm4Ib,
};

/// A fresh command buffer from Redline's builder, paced per `lowering`.
pub(super) fn paced_commands(transport: &Transport<'_>, lowering: rp::ArchLowering) -> Pm4Commands {
    let mut commands = (transport.new_commands)();
    if let Pm4Commands::Gfx12(gfx12) = &mut commands {
        gfx12.set_dispatch_pacing(match lowering.post_dispatch_nop {
            0 => Gfx12DispatchPacing::None,
            n => Gfx12DispatchPacing::PostDispatchNop(n),
        });
    }
    commands
}

/// Emit one segment's commands: entry acquire, the planned boundaries,
/// `dispatch` for every node, and the trailing release. `dispatch` emits the
/// node's dispatch packet (and owns its kernarg buffer).
pub(super) fn emit_segment(
    plan: &rp::Pm4Plan,
    segment: &rp::Segment,
    arch_lowering: rp::ArchLowering,
    transport: &Transport<'_>,
    mut dispatch: impl FnMut(&mut Pm4Commands, usize) -> Result<(), String>,
) -> Result<Pm4Commands, String> {
    let mut commands = paced_commands(transport, arch_lowering);
    for op in plan.segment_ops(segment) {
        match op {
            Pm4Op::Entry => {
                commands.emit_entry_sentinel_reset()?;
                commands.acquire_entry(transport.gcr_trim, transport.entry_policy);
            }
            Pm4Op::Boundary(v) => {
                if v.wait_idle {
                    commands.wait_compute_idle()?;
                }
                match v.acquire {
                    Rung::None => {}
                    Rung::InterNode => commands.acquire_inter_node(transport.gcr_trim, false),
                    Rung::System => commands.gfx12_system_acquire()?,
                }
            }
            Pm4Op::Dispatch(i) => dispatch(&mut commands, i)?,
            Pm4Op::Exit => commands.trailing_release()?,
        }
    }
    Ok(commands)
}

/// A failed replay and what is known of the queue afterwards.
#[allow(dead_code)] // consumed by the WindowQueue implementation (slice A6)
#[derive(Clone, Debug)]
pub(crate) struct OwnedReplayFailure {
    pub error: String,
    pub quiescence: ReplayQuiescence,
}

/// railgun's lowering of a directly authored program, with every object the
/// queue reads owned. Field order is drop order: the indirect buffer first,
/// then the kernels it dispatches, then the kernarg buffers it points at
/// (the order `PreparedPm4Replay` uses).
#[allow(dead_code)] // consumed by the WindowQueue implementation (slice A6)
pub(crate) struct OwnedWindowLowering {
    graph: SingleQueuePm4Ib,
    _kernels: Vec<Kernel>,
    kernargs: Vec<KernargBuffer>,
    blocks: Vec<[u32; 3]>,
    publish: Option<usize>,
    command_dwords: u32,
}

/// Lower `program` (one segment spanning every node, every node PM4) through
/// railgun's own plan with railgun's kernarg images. `kernels[i]` is the
/// loaded object of node `i`. Nothing is submitted.
#[allow(dead_code)] // consumed by the WindowQueue implementation (slice A6)
pub(crate) fn lower_owned(
    program: &rk::KernelProgram,
    plan: &rp::Pm4Plan,
    arch_lowering: rp::ArchLowering,
    kernels: Vec<Kernel>,
    transport: &Transport<'_>,
    device: &GpuDevice,
) -> Result<OwnedWindowLowering, String> {
    let nodes = program.nodes.len();
    if !plan.is_single_segment(nodes) {
        return Err(format!(
            "an owned window lowers exactly one segment over {nodes} nodes ({} segments)",
            plan.segments.len()
        ));
    }
    if let Some((i, node)) = program
        .nodes
        .iter()
        .enumerate()
        .find(|(_, n)| !n.lowering.is_pm4())
    {
        return Err(format!(
            "node {i} {} is not PM4-lowered: {:?}",
            node.symbol, node.lowering
        ));
    }
    if kernels.len() != nodes {
        return Err(format!("{} loaded kernels for {nodes} nodes", kernels.len()));
    }
    let words = program.authored_words();
    let mut buffers: Vec<Option<KernargBuffer>> = (0..nodes).map(|_| None).collect();
    let commands = emit_segment(plan, &plan.segments[0], arch_lowering, transport, |commands, i| {
        let node = &program.nodes[i];
        let metadata = kernels[i].metadata();
        let image = program.kernarg_image(i, metadata.kernarg_segment_size as usize, &words)?;
        let mut buffer = transport
            .pool
            .allocate_for(metadata)
            .map_err(|e| format!("{}: kernarg: {e}", node.symbol))?;
        buffer
            .write_exact(&image)
            .map_err(|e| format!("{}: kernarg: {e}", node.symbol))?;
        let mut workgroup = [0u16; 3];
        for axis in 0..3 {
            workgroup[axis] = u16::try_from(node.block[axis])
                .map_err(|_| format!("{}: block exceeds u16", node.symbol))?;
        }
        let geometry = LaunchGeometry::from_hip_workgroups(node.grid, workgroup)
            .map_err(|e| format!("{}: {e}", node.symbol))?;
        device
            .validate_geometry(geometry)
            .map_err(|e| format!("{}: {e}", node.symbol))?;
        commands
            .dispatch(&kernels[i], geometry, node.dynamic_lds, buffer.address())
            .map_err(|e| format!("{}: {e}", node.symbol))?;
        buffers[i] = Some(buffer);
        Ok(())
    })?;
    if commands.gfx12_dwords().is_none() {
        return Err("an owned window lowers gfx12 command buffers only".to_owned());
    }
    let kernargs: Vec<KernargBuffer> = buffers
        .into_iter()
        .enumerate()
        .map(|(i, b)| b.ok_or_else(|| format!("node {i} was never dispatched")))
        .collect::<Result<_, _>>()?;
    let command_dwords = commands.len_dwords();
    let graph = commands.create_graph(device, transport.pool, None, false)?;
    let publish = kernargs
        .iter()
        .rposition(|kernarg| kernarg.is_device_local() && !kernarg.is_empty());
    Ok(OwnedWindowLowering {
        graph,
        _kernels: kernels,
        kernargs,
        blocks: program.nodes.iter().map(|n| n.block).collect(),
        publish,
        command_dwords,
    })
}

#[allow(dead_code)] // consumed by the WindowQueue implementation (slice A6)
impl OwnedWindowLowering {
    /// Command dwords of the retained indirect buffer.
    pub(crate) fn command_dwords(&self) -> u32 {
        self.command_dwords
    }

    /// Replace node `node`'s whole kernarg segment (one
    /// `KernelProgram::encode_kernarg_into` image). Only valid while the
    /// queue is quiescent.
    pub(crate) fn write_kernarg(&mut self, node: usize, image: &[u8]) -> Result<(), String> {
        self.kernargs
            .get_mut(node)
            .ok_or_else(|| format!("no node {node}"))?
            .write_exact(image)
            .map_err(|e| format!("node {node}: kernarg: {e}"))
    }

    /// Patch node `node`'s dispatch geometry in the indirect buffer to the
    /// HIP grid `grid` (workgroups). gfx12 dispatch packets count work-items,
    /// so the grid is scaled by the node's block, checked. Only valid while
    /// the queue is quiescent.
    pub(crate) fn patch_grid(&mut self, node: usize, grid: [u32; 3]) -> Result<(), String> {
        let block = *self
            .blocks
            .get(node)
            .ok_or_else(|| format!("no node {node}"))?;
        let mut workitems = [0u32; 3];
        for axis in 0..3 {
            workitems[axis] = grid[axis].checked_mul(block[axis]).ok_or_else(|| {
                format!(
                    "node {node}: grid overflow axis={axis} workgroups={} workgroup={}",
                    grid[axis], block[axis]
                )
            })?;
        }
        self.graph
            .patch_dispatch_dimensions(node, workitems)
            .map_err(|e| e.to_string())
    }

    /// Make every host kernarg write since the last call GPU-visible.
    pub(crate) fn publish_host_writes(&self) {
        if let Some(index) = self.publish {
            self.kernargs[index].publish_host_writes();
        }
    }

    /// Submit the retained indirect buffer once and wait for it.
    ///
    /// # Safety
    ///
    /// Every pointer in the written kernarg images must still refer to a live
    /// allocation of the owner (the window binding key proves it).
    pub(crate) unsafe fn replay_and_wait(&mut self) -> Result<(), OwnedReplayFailure> {
        // SAFETY: forwarded from this method's caller.
        unsafe { self.graph.replay_and_wait_checked() }.map_err(|(error, quiescence)| {
            OwnedReplayFailure {
                error: error.to_string(),
                quiescence: match quiescence {
                    redline_dispatch::aql::Quiescence::Proven => ReplayQuiescence::Proven,
                    redline_dispatch::aql::Quiescence::Unknown => ReplayQuiescence::Unknown,
                },
            }
        })
    }
}
