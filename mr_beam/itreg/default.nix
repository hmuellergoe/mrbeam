{
  nixpkgs ? import nix/fetch.nix {
    name = "nixpkgs";
    url = https://github.com/NixOS/nixpkgs/archive/9bbad4c6254513fa62684da57886c4f988a92092.tar.gz;
    sha256 = "1qw1yygcwf42wpf76v9cinsdkamgm5wq5ljpqrqanz69fb0xk86j";
  }
, pkgs ? import nixpkgs {}
, ngsolve ? true
, devel ? true
}:

with pkgs.lib;
with pkgs.python3.pkgs;

buildPythonPackage {
  name = "regpy";
  src = ./.;
  buildInputs = [
    numpy scipy matplotlib
  ]
  ++ optionals ngsolve [
    (callPackage nix/ngsolve.nix {})
  ]
  ++ optionals devel [
    pytest
    (callPackage nix/pdoc3.nix { mako = callPackage nix/mako.nix {}; })
  ];
}
