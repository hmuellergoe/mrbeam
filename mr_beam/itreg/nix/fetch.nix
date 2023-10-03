{ name, url, sha256 }: with import <nix/config.nix>;

derivation {
  inherit name chrootDeps;
  system = builtins.currentSystem;
  src = import <nix/fetchurl.nix> { inherit url sha256; };
  builder = shell; args = [ "-e" (builtins.toFile "unpack" ''
    ${gzip} -d < $src | ${tar} x ${tarFlags}
    ${coreutils}/mv * $out
  '') ];
}
