{ lib
, stdenv
, rocmPackages
, gpuTargets ? [ "gfx1100" ]
}:

stdenv.mkDerivation {
  pname = "hipfire-kernels";
  version = "0.1.20";

  src = lib.cleanSource ./..;

  nativeBuildInputs = [
    rocmPackages.clr
    rocmPackages.llvm.clang
  ];

  buildPhase = ''
    runHook preBuild
    export HOME=$TMPDIR
    bash scripts/compile-kernels.sh ${lib.concatStringsSep " " gpuTargets}
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/kernels/compiled
    for arch in ${lib.concatStringsSep " " gpuTargets}; do
      if [ -d "kernels/compiled/$arch" ]; then
        cp -r "kernels/compiled/$arch" "$out/kernels/compiled/"
      fi
    done
    runHook postInstall
  '';

  meta = with lib; {
    description = "Pre-compiled GPU kernels for hipfire";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
