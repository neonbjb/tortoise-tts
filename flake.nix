{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication;
      in
      {
        packages = {
          myapp = mkPoetryApplication { projectDir = ./.; };
          default = self.packages.${system}.myapp;
        };

        # Shell for app dependencies.
        #
        #     nix develop
        #
        # Use this shell for developing your app.
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python311Packages.torch-bin
            python311Packages.torchaudio-bin
            python311Packages.progressbar
            python311Packages.einops
            python311Packages.librosa
            python311Packages.unidecode
            python311Packages.inflect
            python311Packages.rotary-embedding-torch
            python311Packages.safetensors
            python311Packages.transformers
            (python311Packages.buildPythonPackage rec {
              pname = "tortoise-tts";
              version = "3.0.0";
              doCheck = false;
              buildInputs = [
                python311Packages.pip
                python311Packages.tokenizers
                python311Packages.librosa
                python311Packages.unidecode
                python311Packages.einops
                python311Packages.progressbar
                python311Packages.inflect
                python311Packages.rotary-embedding-torch
                python311Packages.safetensors
                (python311Packages.buildPythonPackage rec {
                  doCheck = false;
                  pname = "transformers";
                  version = "4.31.0";
                  buildInputs = [
                    python311Packages.tokenizers # Version might be wrong
                    python311Packages.pip
                    python311Packages.tqdm
                    python311Packages.safetensors
                  ];
                  src = python311Packages.fetchPypi {
                    inherit pname version;
                    sha256 = "4302fba920a1c24d3a429a29efff6a63eac03f3f3cf55b55927fc795d01cb273";
                  };
                })
              ];
              src = python311Packages.fetchPypi {
                inherit pname version;
                sha256 = "8684aac23976ffa9813a4ec074b6bff0a348ce82a1923483e57d5d4e27dd21d6";
              };
            })
          ];
          inputsFrom = [ 
            pkgs.cudaPackages.libcusparse
            self.packages.${system}.myapp
          ];
        };

        # Shell for poetry.
        #
        #     nix develop .#poetry
        #
        # Use this shell for changes to pyproject.toml and poetry.lock.
        devShells.poetry = pkgs.mkShell {
          packages = [ pkgs.poetry ];
        };
      });
}
