{ lib
, callPackage
, rustPlatform
, rocmPackages
, makeWrapper
, rocmSupport ? true
, src ? lib.cleanSource ./..
, cargoLockFile ? ../Cargo.lock
, kernels
}:

let
  cargoToml = builtins.fromTOML (builtins.readFile (src + "/Cargo.toml"));
  rocmRoot = callPackage ./rocm-root.nix { };
  # Both wrappers need the same ROCm env: a CLI-launched daemon takes its
  # policy (incl. diagnostic.compiler.hipcc_extra_flags) from the CLI's
  # startup config, not its own environment. The device-lib flag must match
  # the one hipfire-kernels was packed with, or every packaged object is
  # rejected and JIT-compiled.
  rocmWrapArgs = lib.optionalString rocmSupport ''
    --set HIPFIRE_ROCM_PATH ${rocmRoot} \
    --set HIPFIRE_HIPCC_EXTRA_FLAGS --rocm-device-lib-path=${rocmPackages.rocm-device-libs}/amdgcn/bitcode \
    --prefix LD_LIBRARY_PATH : ${rocmRoot}/lib
  '';
in
rustPlatform.buildRustPackage {
  pname = "hipfire";
  version = cargoToml.workspace.package.version or cargoToml.package.version;

  inherit src;
  cargoLock.lockFile = cargoLockFile;
  doCheck = false;  # tests require GPU

  # The deliverables are standalone binary crates (mirrors Containerfile).
  buildPhase = ''
    runHook preBuild
    cargo build --release -p hipfire-daemon
    cargo build --release -p hipfire-cli
    runHook postBuild
  '';

  dontCargoInstall = true;

  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin
    # current_exe resolves to hipfire-daemon-unwrapped in this directory.
    cp -r ${kernels}/kernels $out/bin/kernels

    # Install and wrap daemon binary with LD_LIBRARY_PATH for libamdhip64.so dlopen.
    # `hipfire-daemon`'s [[bin]] name is `daemon`, so the artifact is
    # target/release/daemon (mirrors Containerfile's COPY to /opt/hipfire/bin/daemon).
    cp target/release/daemon $out/bin/hipfire-daemon-unwrapped
    makeWrapper $out/bin/hipfire-daemon-unwrapped $out/bin/hipfire-daemon ${rocmWrapArgs}

    # Install the native Rust control plane. HIPFIRE_DAEMON_BIN points it at
    # the ROCm-wrapped daemon rather than relying on a source-tree layout.
    cp target/release/hipfire $out/bin/hipfire-unwrapped
    makeWrapper $out/bin/hipfire-unwrapped $out/bin/hipfire \
      --set HIPFIRE_DAEMON_BIN $out/bin/hipfire-daemon ${rocmWrapArgs}

    runHook postInstall
  '';

  meta = with lib; {
    description = "LLM inference for AMD RDNA GPUs";
    homepage = "https://github.com/warpfront/hipfire";
    license = [ licenses.asl20 licenses.mit ];
    platforms = [ "x86_64-linux" ];
    mainProgram = "hipfire";
  };
}
