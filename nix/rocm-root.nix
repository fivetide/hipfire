# One coherent ROCm prefix for HIPFIRE_ROCM_PATH.
#
# hipfire's resolver (crates/hipfire-config/src/rocm.rs) treats
# HIPFIRE_ROCM_PATH as authoritative and requires libamdhip64,
# libhsa-runtime64, include/hip/hip_runtime.h and bin/hipcc under that one
# root. nixpkgs splits ROCm across store paths (libhsa-runtime64 lives in
# rocm-runtime, not clr), so no single package satisfies it.
{ symlinkJoin, rocmPackages }:
symlinkJoin {
  name = "hipfire-rocm-root";
  paths = [
    rocmPackages.clr
    rocmPackages.rocm-runtime
    rocmPackages.rocm-comgr
    rocmPackages.rocprofiler-register
  ];
}
