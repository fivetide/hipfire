{ config, lib, pkgs, ... }:

let
  cfg = config.services.hipfire;

  settingsWithPort = cfg.settings // { port = cfg.port; };
  configJson = pkgs.writeText "hipfire-config.json"
    (builtins.toJSON settingsWithPort);
  perModelConfigJson = pkgs.writeText "hipfire-per-model-config.json"
    (builtins.toJSON cfg.perModelSettings);

  hipfirePkg = cfg.package;
  hipfireKernelsPkg =
    if cfg.kernelsPackage == pkgs.hipfire-kernels
    then cfg.kernelsPackage.override { gpuTargets = cfg.gpuTargets; }
    else cfg.kernelsPackage;

  envList =
    (lib.mapAttrsToList (k: v: "${k}=${v}") cfg.environment)
    ++ lib.optionals cfg.rocmSupport [
      "LD_LIBRARY_PATH=${pkgs.rocmPackages.clr}/lib"
    ];
in
{
  options.services.hipfire = {

    enable = lib.mkEnableOption "hipfire inference daemon";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.hipfire;
      defaultText = lib.literalExpression "pkgs.hipfire";
      description = "The hipfire package to use.";
    };

    kernelsPackage = lib.mkOption {
      type = lib.types.package;
      default = pkgs.hipfire-kernels;
      defaultText = lib.literalExpression "pkgs.hipfire-kernels";
      description = "Pre-compiled GPU kernels package.";
    };

    gpuTargets = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ "gfx1100" ];
      example = [ "gfx1100" "gfx1030" ];
      description = ''
        GPU architectures to compile kernels for.
        Detect yours: grep gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties
      '';
    };

    rocmSupport = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Use nixpkgs ROCm libraries (rocmPackages.clr).
        Set to false to provide your own libamdhip64.so via environment.
      '';
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 11435;
      description = "Port for the OpenAI-compatible API server.";
    };

    modelDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/hipfire/models";
      description = "Directory containing model files.";
    };

    settings = lib.mkOption {
      type = lib.types.attrs;
      default = {
        temperature = 0.3;
        top_p = 0.8;
        max_tokens = 512;
      };
      description = ''
        Global configuration written to config.json.
        See docs/CONFIG.md for all available keys.
        Note: the 'port' key is managed by services.hipfire.port.
      '';
    };

    perModelSettings = lib.mkOption {
      type = lib.types.attrsOf lib.types.attrs;
      default = { };
      example = lib.literalExpression ''
        {
          "qwen3.5:27b" = {
            max_seq = 16384;
            kv_cache = "q8";
          };
        }
      '';
      description = ''
        Per-model config overrides written to per_model_config.json.
        Keys are model tags, values are config attrsets.
      '';
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      example = lib.literalExpression ''
        { HIPFIRE_KV_MODE = "asym3"; }
      '';
      description = "Extra environment variables (HIPFIRE_*) for the daemon.";
    };

    userService = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Run as a user-level systemd service (systemctl --user)
        instead of a system service. No dedicated user is created.
      '';
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "hipfire";
      description = "User to run the daemon as (system mode only).";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "hipfire";
      description = "Group to run the daemon as (system mode only).";
    };
  };

  config = lib.mkIf cfg.enable (lib.mkMerge [

    # ── System service mode ──────────────────────────────────
    (lib.mkIf (!cfg.userService) {

      users.users.${cfg.user} = {
        isSystemUser = true;
        group = cfg.group;
        extraGroups = [ "video" "render" ];
        home = "/var/lib/hipfire";
        createHome = true;
      };
      users.groups.${cfg.group} = { };

      systemd.services.hipfire-setup = {
        description = "hipfire config setup";
        wantedBy = [ "multi-user.target" ];
        before = [ "hipfire-precompile.service" "hipfire.service" ];
        serviceConfig = {
          Type = "oneshot";
          RemainAfterExit = true;
          User = cfg.user;
          Group = cfg.group;
        };
        script = ''
          mkdir -p /var/lib/hipfire/.hipfire/bin
          mkdir -p ${lib.escapeShellArg cfg.modelDir}
          cp -f ${configJson} /var/lib/hipfire/.hipfire/config.json
          cp -f ${perModelConfigJson} /var/lib/hipfire/.hipfire/per_model_config.json
          ln -sf ${hipfirePkg}/bin/hipfire-daemon /var/lib/hipfire/.hipfire/bin/daemon
          ln -sfn ${hipfireKernelsPkg}/kernels /var/lib/hipfire/.hipfire/bin/kernels
        '';
      };

      systemd.services.hipfire-precompile = {
        description = "hipfire GPU kernel pre-compilation";
        wantedBy = [ "multi-user.target" ];
        after = [ "hipfire-setup.service" ];
        before = [ "hipfire.service" ];
        serviceConfig = {
          Type = "oneshot";
          RemainAfterExit = true;
          User = cfg.user;
          Group = cfg.group;
          Environment = envList;
        };
        script = ''
          export HOME=/var/lib/hipfire
          if ! ${hipfirePkg}/bin/hipfire-daemon --precompile; then
            echo "WARNING: kernel pre-compilation failed (exit $?). Daemon will JIT-compile on first request." >&2
          fi
        '';
      };

      systemd.services.hipfire = {
        description = "hipfire inference daemon";
        after = [ "network.target" "hipfire-precompile.service" ];
        wantedBy = [ "multi-user.target" ];
        serviceConfig = {
          ExecStart = "${hipfirePkg}/bin/hipfire serve";
          Restart = "on-failure";
          RestartSec = 5;
          User = cfg.user;
          Group = cfg.group;
          Environment = envList ++ [
            "HOME=/var/lib/hipfire"
          ];
          ProtectSystem = "strict";
          ReadWritePaths = [ cfg.modelDir "/var/lib/hipfire" ];
          NoNewPrivileges = true;
          DevicePolicy = "closed";
          DeviceAllow = [ "/dev/kfd rw" "/dev/dri rw" "/dev/dri/* rw" ];
        };
      };
    })

    # ── User service mode ────────────────────────────────────
    (lib.mkIf cfg.userService {

      systemd.user.services.hipfire-setup = {
        description = "hipfire config setup (user)";
        wantedBy = [ "default.target" ];
        before = [ "hipfire.service" ];
        serviceConfig = {
          Type = "oneshot";
          RemainAfterExit = true;
        };
        script = ''
          mkdir -p $HOME/.hipfire/bin
          mkdir -p ${lib.escapeShellArg cfg.modelDir}
          cp -f ${configJson} $HOME/.hipfire/config.json
          cp -f ${perModelConfigJson} $HOME/.hipfire/per_model_config.json
          ln -sf ${hipfirePkg}/bin/hipfire-daemon $HOME/.hipfire/bin/daemon
          ln -sfn ${hipfireKernelsPkg}/kernels $HOME/.hipfire/bin/kernels
        '';
      };

      systemd.user.services.hipfire = {
        description = "hipfire inference daemon (user)";
        after = [ "hipfire-setup.service" ];
        wantedBy = [ "default.target" ];
        serviceConfig = {
          ExecStart = "${hipfirePkg}/bin/hipfire serve";
          Restart = "on-failure";
          RestartSec = 5;
          Environment = envList;
        };
      };
    })
  ]);
}
