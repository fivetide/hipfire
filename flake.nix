{
  description = "hipfire — LLM inference for AMD RDNA GPUs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    let lib = nixpkgs.lib; in
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        hipfire = pkgs.callPackage ./nix/package.nix {
          rocmSupport = true;
          src = lib.cleanSource ./.;
          cargoLockFile = ./Cargo.lock;
        };

        # Default target; override via services.hipfire.gpuTargets in NixOS module
        hipfire-kernels = pkgs.callPackage ./nix/kernels.nix {
          gpuTargets = [ "gfx1100" ];
        };
      in
      {
        packages = {
          default = hipfire;
          inherit hipfire hipfire-kernels;
        };

        devShells.default = pkgs.callPackage ./nix/dev-shell.nix {
          rust-bin = pkgs.rust-bin;
          rocmSupport = true;
        };
      }
    ) // {
      nixosModules.default = import ./nix/module.nix;

      overlays.default = final: prev: {
        hipfire = final.callPackage ./nix/package.nix {
          rocmSupport = true;
          src = lib.cleanSource ./.;
          cargoLockFile = ./Cargo.lock;
        };
        hipfire-kernels = final.callPackage ./nix/kernels.nix {
          gpuTargets = [ "gfx1100" ];
        };
      };
    };
}
