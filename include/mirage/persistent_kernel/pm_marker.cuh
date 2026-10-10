// pm_marker.cuh — task-boundary markers that Nsight Compute PM sampling can see.
//
// Why: MPK's trace (%globaltimer stamps) and ncu's PM samples live on different
// clocks, and range replay may give them different executions. A marker that
// both record lines them up exactly: the trace already has the stamp, and the
// marker's instructions show up in a PM counter at the same moment.
//
// Channel: the FP64 pipe. A full demo run executes no FP64 SASS at all
// (profile2.ncu-rep, sass__inst_executed_per_opcode_pipeline has no FP64
// entry), so every FP64 cycle in a PM sample is a marker. The default
// PmSampling section on CC 9.0 already records it as "SM FP64 Pipe Throughput"
// (sm__pipe_fp64_cycles_active_realtime..., prefixed TPC.TriageCompute. or
// pmsampling: depending on the ncu version). The texture pipe is
// not clean (~5e7 TEX instructions per run), so don't use it.
//
// Cost: a short chain of register-only DFMAs from one thread. No memory
// traffic, no shared state, no change to task code.
//
// Build flag: -DBH_MPK_PM_MARKERS (see "Plumbing" below). Without it every
// function here compiles to nothing.
//
// ---------------------------------------------------------------- placement
// Called in execute_worker (persistent_kernel.cuh) right after the
// PROFILER_EVENT_START stamp, in the region every thread passes through.
// It must sit next to PROFILER_EVENT_START, because mpk_ncu_align.py predicts
// the marker signal from each trace row's start stamp (t0). Thread 0 fires it,
// as the profiler stamps are thread 0 too. TASK_BEGIN_TASK_GRAPH (one per
// iteration) fires a longer burst, so iterations stand out and the aligner
// cannot lock onto the wrong layer (~170 us period).
//
// ----------------------------------------------------------------- plumbing
// persistent_kernel.py adds -DBH_MPK_PM_MARKERS and -DBH_MPK_PM_MARKER_BURST
// when the env vars BH_MPK_PM_MARKERS=1 / BH_MPK_PM_MARKER_BURST are set;
// persistent_kernel.cuh includes this file after task_header.cuh.
// profile_runs.sh sets both with PM_MARKERS=1 (and PM_BURST).
//
// ------------------------------------------------------------------- checks
// 1. SASS: `cuobjdump -sass` the build with and without the flag. Only the
//    worker loop should differ (DFMA/DSETP plus one predicated store), not the
//    task bodies.
// 2. Overhead: R0 latency with markers on vs off (expect noise-level).
// 3. Calibration: in a PM report, total FP64 cycles / number of markers is a
//    constant. mpk_ncu_align.py fits a scale factor, so its value doesn't
//    matter, only that it is constant.
#pragma once
#include <cstdint>

#ifndef BH_MPK_PM_MARKER_BURST
#define BH_MPK_PM_MARKER_BURST 32  // DFMAs per iteration marker; keep in sync
#endif                             // with mpk_ncu_align.py --marker-burst

namespace bh {

#ifdef BH_MPK_PM_MARKERS
// Written only if a DFMA chain lands on a value it never takes. ptxas can't
// prove that, so the chain survives dead-code elimination.
__device__ double pm_marker_sink;

// N dependent DFMAs on a seed ptxas can't constant-fold (%clock).
template <int N>
__device__ __forceinline__ void pm_mark_fp64() {
  double x;
  asm volatile("{\n\t.reg .u32 c;\n\tmov.u32 c, %%clock;\n\t"
               "cvt.rn.f64.u32 %0, c;\n\t}" : "=d"(x));
#pragma unroll
  for (int i = 0; i < N; i++) {
    asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(x) : "d"(1.0000001), "d"(0.5));
  }
  if (x == -1.0) pm_marker_sink = x;
}

__device__ __forceinline__ void bh_pm_mark_task_start(int task_type) {
  // TASK_TERMINATE has no trace row, so it gets no marker either.
  if (threadIdx.x != 0 || task_type == mirage::runtime::TASK_TERMINATE) return;
  if (task_type == mirage::runtime::TASK_BEGIN_TASK_GRAPH) {
    pm_mark_fp64<BH_MPK_PM_MARKER_BURST>();
  } else {
    pm_mark_fp64<1>();
  }
}
#else
__device__ __forceinline__ void bh_pm_mark_task_start(int) {}
#endif

}  // namespace bh

using bh::bh_pm_mark_task_start;
