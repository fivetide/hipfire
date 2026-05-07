# NixOS

hipfire has first-class NixOS support via a Nix flake.

## Prerequisites

- NixOS 24.05+ or nixos-unstable
- AMD GPU with the `amdgpu` kernel module loaded
- User in `video` and `render` groups (for non-service usage)

Verify your GPU is visible:

    ls /dev/kfd         # should exist
    ls /dev/dri/        # should show renderD128+

Detect your GPU architecture:

    grep gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties

## Quick Start

### Development shell

Enter a shell with Rust, bun, hipcc, and ROCm tools:

    nix develop github:Kaden-Schutt/hipfire

This works with any hipfire checkout — the shell provides tools only,
not the source tree:

    nix develop github:Kaden-Schutt/hipfire
    cd ~/my-hipfire-fork
    cargo build --release --features deltanet --example daemon -p hipfire-runtime

### Build from source

    nix build github:Kaden-Schutt/hipfire
    ./result/bin/hipfire run qwen3.5:9b "Hello"

### Build kernels

    nix build github:Kaden-Schutt/hipfire#hipfire-kernels

## NixOS Module

Add hipfire to your flake inputs and enable the service.

### Minimal example

```nix
# flake.nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    hipfire.url = "github:Kaden-Schutt/hipfire";
  };

  outputs = { nixpkgs, hipfire, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        hipfire.nixosModules.default
        {
          nixpkgs.overlays = [ hipfire.overlays.default ];
          services.hipfire.enable = true;
          services.hipfire.gpuTargets = [ "gfx1100" ];
        }
      ];
    };
  };
}
```

### Full example

```nix
services.hipfire = {
  enable = true;
  gpuTargets = [ "gfx1100" ];

  # Global config (written to config.json)
  settings = {
    temperature = 0.3;
    top_p = 0.8;
    max_tokens = 512;
    kv_cache = "asym3";
    dflash_mode = "auto";
    idle_timeout = 300;
    default_model = "qwen3.5:9b";
  };

  # Per-model overrides (written to per_model_config.json)
  perModelSettings = {
    "qwen3.5:27b" = {
      max_seq = 16384;
      kv_cache = "q8";
      dflash_mode = "on";
    };
  };

  # Extra env vars (highest precedence)
  environment = {
    HIPFIRE_NORMALIZE_PROMPT = "1";
  };

  modelDir = "/var/lib/hipfire/models";
};
```

### Desktop / user-service mode

For single-user desktop setups, use a user-level systemd service:

```nix
services.hipfire = {
  enable = true;
  userService = true;
  gpuTargets = [ "gfx1100" ];
};
```

Manage with `systemctl --user start hipfire`, `systemctl --user status hipfire`.

Your user must be in `video` and `render` groups:

```nix
users.users.yourname.extraGroups = [ "video" "render" ];
```

## GPU Targets

| Arch | Card | Generation |
|------|------|-----------|
| gfx906 | Vega 20 / MI50 | GCN5 |
| gfx908 | MI100 | CDNA |
| gfx1010 | RX 5700 XT | RDNA1 |
| gfx1030 | RX 6800 XT | RDNA2 |
| gfx1100 | RX 7900 XTX | RDNA3 |
| gfx1151 | Strix Halo | RDNA3.5 |
| gfx1200 | Radeon R9700 | RDNA4 |
| gfx1201 | RX 9070 XT | RDNA4 |

Build kernels for multiple arches:

```nix
services.hipfire.gpuTargets = [ "gfx1100" "gfx1030" ];
```

## ROCm Configuration

### Default: bundled nixpkgs ROCm

hipfire uses `rocmPackages.clr` from nixpkgs by default.
`LD_LIBRARY_PATH` is injected automatically via wrapper scripts.

### Bring your own ROCm

```nix
services.hipfire = {
  rocmSupport = false;
  environment = {
    LD_LIBRARY_PATH = "/opt/rocm/lib";
  };
};
```

### Custom ROCm version via overlay

Override `rocmPackages` in your nixpkgs overlay. The hipfire package
and module reference `rocmPackages.clr` generically, so overlays
apply transparently.

## Configuration

hipfire uses a layered config system. Precedence (lowest to highest):

1. Engine defaults (hardcoded)
2. `config.json` — set via `services.hipfire.settings`
3. `per_model_config.json` — set via `services.hipfire.perModelSettings`
4. Environment variables — set via `services.hipfire.environment`

For the full list of config keys, see [CONFIG.md](CONFIG.md).

### Environment variables (interactive use)

For interactive usage outside the systemd service (`hipfire run`, `hipfire chat`),
set `HIPFIRE_*` variables in your shell profile:

```bash
# ~/.bashrc or equivalent
export HIPFIRE_KV_MODE=asym3
export HIPFIRE_NORMALIZE_PROMPT=1
```

These only affect the current user's shell sessions, not the systemd service.

## Troubleshooting

### "libamdhip64.so not found"

The HIP runtime library is missing. If using bundled ROCm (`rocmSupport = true`),
verify `rocmPackages.clr` is available:

    nix build nixpkgs#rocmPackages.clr

If using bring-your-own, check your `LD_LIBRARY_PATH`:

    ls -la /opt/rocm/lib/libamdhip64.so*

### "Permission denied" on /dev/kfd

Your user needs `video` and `render` group membership:

```nix
users.users.yourname.extraGroups = [ "video" "render" ];
```

Then rebuild and relogin.

### "No AMD GPU detected"

Check that the amdgpu kernel module is loaded:

    lsmod | grep amdgpu

On NixOS, ensure firmware is available:

```nix
hardware.firmware = [ pkgs.linux-firmware ];
# or for AMD specifically:
hardware.amdgpu.initrd.enable = true;  # NixOS 24.05+
```

### Kernel pre-compilation fails

Ensure hipcc version matches your GPU target. Check with:

    hipcc --version

gfx1200/gfx1201 requires ROCm 6.4+, gfx1151 requires ROCm 7.2+.
