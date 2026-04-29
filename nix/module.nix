self: {
  pkgs,
  lib,
  config,
  ...
}: let
  cfg = config.services.llama-mesh;
  wcfg = cfg.worker;
  ccfg = cfg.coordinator;

  defaultPkg = self.packages.${pkgs.system}.default;

  gpuPkgFor = gpu:
    {
      rocm = pkgs.llama-cpp.override {
        rocmSupport = true;
        rpcSupport = true;
      };
      cuda = pkgs.llama-cpp.override {
        cudaSupport = true;
        rpcSupport = true;
      };
      cpu = pkgs.llama-cpp.override {rpcSupport = true;};
    }
    .${gpu};

  gpuDeviceAllow = [
    "char-nvidiactl"
    "char-nvidia-caps"
    "char-nvidia-frontend"
    "char-nvidia-uvm"
    "char-drm"
    "char-fb"
    "char-kfd"
    "/dev/dxg"
  ];

  gpuEnv = gpu: visibleDevices: rocmVersion:
    lib.optionalAttrs (visibleDevices != null && gpu == "cuda") {
      CUDA_VISIBLE_DEVICES = visibleDevices;
    }
    // lib.optionalAttrs (visibleDevices != null && gpu == "rocm") {
      ROCR_VISIBLE_DEVICES = visibleDevices;
    }
    // lib.optionalAttrs (gpu == "rocm" && rocmVersion != "") {
      HSA_OVERRIDE_GFX_VERSION = rocmVersion;
    };

  # Coordinator TOML config generation
  coordHome = "/var/lib/llama-mesh";

  llamaPkg =
    if ccfg.gpu != null
    then gpuPkgFor ccfg.gpu
    else pkgs.llama-cpp.override {rpcSupport = true;};

  tomlFormat = pkgs.formats.toml {};
  coordConfigFile = tomlFormat.generate "llama-mesh-coordinator.toml" {
    swap_bin = "${pkgs.llama-swap}/bin/llama-swap";
    swap_listen = "${ccfg.swapHost}:${toString ccfg.swapPort}";
    llama_server_bin = "${llamaPkg}/bin/llama-server";
    swap_config_path = "${coordHome}/swap.yaml";
    local_vram_mb = ccfg.localVramMb;
    models = map (m: {
      inherit (m) name path args ttl;
    }) ccfg.models;
  };

  needsGpuWorker = wcfg.gpu != "cpu";
  needsGpuCoord = ccfg.gpu != null;
in {
  options.services.llama-mesh = {
    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    worker = {
      enable = lib.mkEnableOption "llama-mesh worker";

      package = lib.mkOption {
        type = lib.types.package;
        default = defaultPkg;
        defaultText = lib.literalExpression "inputs.llama-mesh.packages.\${pkgs.system}.default";
        description = "The llama-mesh package to use";
      };

      coordinator = lib.mkOption {
        type = lib.types.str;
        example = "ws://192.168.178.24:50050";
        description = "WebSocket URL of the coordinator";
      };

      gpu = lib.mkOption {
        type = lib.types.enum ["cuda" "rocm" "cpu"];
        description = "GPU backend for this worker";
      };

      rpcPort = lib.mkOption {
        type = lib.types.port;
        default = 50052;
        description = "Port for the local RPC server";
      };

      rpcHost = lib.mkOption {
        type = lib.types.str;
        default = "0.0.0.0";
        description = "Bind address for the local RPC server";
      };

      preemptTriggers = lib.mkOption {
        type = lib.types.listOf lib.types.str;
        default = [];
        example = ["gamescope" "steam"];
        description = "Process names that trigger GPU preemption (worker leaves the mesh while they run)";
      };

      nodeId = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "Node identifier (null = use hostname)";
      };

      rocmVersion = lib.mkOption {
        type = lib.types.str;
        default = "";
        description = "ROCm GFX version override (sets HSA_OVERRIDE_GFX_VERSION)";
      };

      visibleDevices = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "Device selection: CUDA_VISIBLE_DEVICES for cuda, ROCR_VISIBLE_DEVICES for rocm (null = all)";
      };
    };

    # ------------------------------------------------------------------
    # Coordinator
    # ------------------------------------------------------------------
    coordinator = {
      enable = lib.mkEnableOption "llama-mesh coordinator";

      package = lib.mkOption {
        type = lib.types.package;
        default = defaultPkg;
        defaultText = lib.literalExpression "inputs.llama-mesh.packages.\${pkgs.system}.default";
        description = "The llama-mesh package to use";
      };

      host = lib.mkOption {
        type = lib.types.str;
        default = "0.0.0.0";
        description = "Bind address for worker WebSocket connections";
      };

      port = lib.mkOption {
        type = lib.types.port;
        default = 50050;
        description = "Port for worker WebSocket connections";
      };

      gpu = lib.mkOption {
        type = lib.types.nullOr (lib.types.enum ["cuda" "rocm"]);
        default = null;
        description = "Local GPU backend (null = no local GPU, RPC-only)";
      };

      localVramMb = lib.mkOption {
        type = lib.types.int;
        default = 0;
        description = "Local VRAM in MB to include in tensor split (0 = coordinator has no GPU)";
      };

      swapHost = lib.mkOption {
        type = lib.types.str;
        default = "127.0.0.1";
        description = "Bind address for llama-swap's API listener";
      };

      swapPort = lib.mkOption {
        type = lib.types.port;
        default = 8080;
        description = "Port for llama-swap's API listener";
      };

      rocmVersion = lib.mkOption {
        type = lib.types.str;
        default = "";
        description = "ROCm GFX version override (sets HSA_OVERRIDE_GFX_VERSION)";
      };

      visibleDevices = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "Device selection for llama-server children: CUDA_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES";
      };

      models = lib.mkOption {
        type = lib.types.listOf (lib.types.submodule {
          options = {
            name = lib.mkOption {
              type = lib.types.str;
              description = "Model id exposed via the OpenAI-compatible API";
            };
            path = lib.mkOption {
              type = lib.types.str;
              description = "Absolute path to the GGUF model file";
            };
            args = lib.mkOption {
              type = lib.types.listOf lib.types.str;
              default = ["-ngl" "999" "-c" "16384" "--flash-attn" "on"];
              description = "Arguments passed to llama-server for this model";
            };
            ttl = lib.mkOption {
              type = lib.types.int;
              default = 300;
              description = "Seconds of idle before llama-swap unloads this model (0 = never)";
            };
          };
        });
        default = [];
        description = "GGUF models served through llama-swap";
      };

      openFirewall = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Open the coordinator WebSocket port in the firewall";
      };
    };
  };

  config = lib.mkMerge [
    # Shared user/group
    (lib.mkIf (wcfg.enable || ccfg.enable) {
      users.users.llama-mesh = {
        isSystemUser = true;
        group = "llama-mesh";
      };
      users.groups.llama-mesh = {};
    })

    # Worker service
    (lib.mkIf wcfg.enable {
      systemd.services.llama-mesh-worker = {
        description = "llama-mesh worker";
        wantedBy = ["multi-user.target"];
        after = ["network.target"];
        environment =
          {
            LD_LIBRARY_PATH = "${gpuPkgFor wcfg.gpu}/lib";
          }
          // gpuEnv wcfg.gpu wcfg.visibleDevices wcfg.rocmVersion;
        serviceConfig = {
          Type = "exec";
          ExecStart = lib.concatStringsSep " " (
            [
              "${wcfg.package}/bin/llama-mesh"
              "worker"
              "--coordinator"
              (lib.escapeShellArg wcfg.coordinator)
              "--gpu"
              wcfg.gpu
              "--rpc-port"
              (toString wcfg.rpcPort)
              "--rpc-host"
              wcfg.rpcHost
            ]
            ++ lib.optionals (wcfg.preemptTriggers != []) [
              "--preempt-triggers"
              (lib.concatStringsSep "," wcfg.preemptTriggers)
            ]
            ++ lib.optionals (wcfg.nodeId != null) [
              "--node-id"
              (lib.escapeShellArg wcfg.nodeId)
            ]
          );
          User = "llama-mesh";
          Group = "llama-mesh";
          PrivateDevices = !needsGpuWorker;
          DevicePolicy =
            if needsGpuWorker
            then "closed"
            else "strict";
          DeviceAllow = lib.optionals needsGpuWorker gpuDeviceAllow;
          SupplementaryGroups = lib.optionals needsGpuWorker ["render" "video"];
          Restart = "always";
          RestartSec = "3";
        };
      };
    })

    # Coordinator service
    (lib.mkIf ccfg.enable {
      networking.firewall.allowedTCPPorts = lib.optional ccfg.openFirewall ccfg.port;

      environment.systemPackages = [pkgs.llama-swap llamaPkg];

      systemd.tmpfiles.rules = [
        "d ${coordHome} 0755 llama-mesh llama-mesh -"
      ];

      systemd.services.llama-mesh-coordinator = {
        description = "llama-mesh coordinator";
        wantedBy = ["multi-user.target"];
        after = ["network.target"];
        environment =
          lib.optionalAttrs needsGpuCoord (
            {LD_LIBRARY_PATH = "${llamaPkg}/lib";}
            // gpuEnv ccfg.gpu ccfg.visibleDevices ccfg.rocmVersion
          );
        serviceConfig = {
          Type = "exec";
          ExecStart = lib.concatStringsSep " " [
            "${ccfg.package}/bin/llama-mesh"
            "coord"
            "--listen"
            "${ccfg.host}:${toString ccfg.port}"
            "--config"
            "${coordConfigFile}"
          ];
          User = "llama-mesh";
          Group = "llama-mesh";
          WorkingDirectory = coordHome;
          StateDirectory = ["llama-mesh"];
          ReadWritePaths = [coordHome];
          PrivateDevices = !needsGpuCoord;
          DevicePolicy =
            if needsGpuCoord
            then "closed"
            else "strict";
          DeviceAllow = lib.optionals needsGpuCoord gpuDeviceAllow;
          SupplementaryGroups = lib.optionals needsGpuCoord ["render" "video"];
          Restart = "always";
          RestartSec = "3";
        };
      };
    })
  ];
}
