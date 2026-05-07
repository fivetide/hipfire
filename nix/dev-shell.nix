{ lib
, mkShell
, rust-bin
, rocmPackages
, bun
, pkg-config
, rocmSupport ? true
}:

mkShell {
  name = "hipfire-dev";

  nativeBuildInputs = [
    (rust-bin.stable.latest.default.override {
      extensions = [ "rust-src" "rust-analyzer" ];
    })
    bun
    pkg-config
  ] ++ lib.optionals rocmSupport [
    rocmPackages.clr
    rocmPackages.rocm-smi
    rocmPackages.rocminfo
  ];

  LD_LIBRARY_PATH = lib.optionalString rocmSupport
    "${rocmPackages.clr}/lib";

  shellHook = ''
    echo "hipfire dev shell"
    echo "  rust: $(rustc --version)"
    echo "  bun:  $(bun --version)"
    ${lib.optionalString rocmSupport ''
      echo "  hip:  $(hipcc --version 2>&1 | head -1)"
    ''}
  '';
}
