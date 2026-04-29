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
      src = craneLib.cleanCargoSource ./.;

      commonArgs = {
        inherit src;
        pname = "llama-mesh";
        version = "0.1.0";
        strictDeps = true;
      };

      # Build deps once, share across both binaries
      cargoArtifacts = craneLib.buildDepsOnly commonArgs;

      llama-mesh-worker = craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
          pname = "llama-mesh-worker";
          cargoExtraArgs = "-p llama-mesh-worker";
        });

      llama-mesh-coord = craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
          pname = "llama-mesh-coord";
          cargoExtraArgs = "-p llama-mesh-coord";
        });
    in {
      default = pkgs.symlinkJoin {
        name = "llama-mesh";
        paths = [llama-mesh-worker llama-mesh-coord];
      };
      inherit llama-mesh-worker llama-mesh-coord;
    });

    devShells = forEachSystem ({
      pkgs,
      craneLib,
    }: {
      default = craneLib.devShell {
        packages = [pkgs.rust-analyzer];
      };
    });
  };
}
