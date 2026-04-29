{
  description = "llama-mesh — dynamic distributed LLM inference over llama-cpp RPC";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    crane.url = "github:ipetkov/crane";
  };

  outputs = {
    nixpkgs,
    crane,
    ...
  }: let
    systems = ["x86_64-linux" "aarch64-linux"];
    forEachSystem = f:
      nixpkgs.lib.genAttrs systems (system:
        f {
          pkgs = nixpkgs.legacyPackages.${system};
          craneLib = crane.mkLib nixpkgs.legacyPackages.${system};
        });
  in {
    packages = forEachSystem ({
      pkgs,
      craneLib,
    }: let
      # CPU-only RPC-enabled llama-cpp for linking at build time.
      # GPU backends (CUDA/ROCm) are loaded dynamically at runtime —
      # the NixOS module sets LD_LIBRARY_PATH to the right llama-cpp build.
      llama-cpp-rpc = pkgs.llama-cpp.override {rpcSupport = true;};

      src = craneLib.cleanCargoSource ./.;

      commonArgs = {
        inherit src;
        pname = "llama-mesh";
        version = "0.1.0";
        strictDeps = true;
        buildInputs = [llama-cpp-rpc];

        # Ensure the Rust linker can find libggml-rpc.so and friends
        LIBRARY_PATH = "${llama-cpp-rpc}/lib";
      };

      cargoArtifacts = craneLib.buildDepsOnly commonArgs;

      llama-mesh = craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
        });
    in {
      default = llama-mesh;
    });

    devShells = forEachSystem ({
      pkgs,
      craneLib,
    }: let
      llama-cpp-rpc = pkgs.llama-cpp.override {rpcSupport = true;};
    in {
      default = craneLib.devShell {
        packages = [pkgs.rust-analyzer];
        buildInputs = [llama-cpp-rpc];
        LIBRARY_PATH = "${llama-cpp-rpc}/lib";
        LD_LIBRARY_PATH = "${llama-cpp-rpc}/lib";
      };
    });
  };
}
