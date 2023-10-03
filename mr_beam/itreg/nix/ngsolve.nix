{
  lib, buildPythonPackage, fetchFromGitHub, python, cmake, tk, tix,
  tkinter, openblasCompat, liblapackWithoutAtlas, mesa_glu, suitesparse, xorg,
  makeWrapper, writeText, glibcLocales, scalapack
}:

with lib;

buildPythonPackage rec {
  name = "ngsolve-${version}";
  namePrefix = "";
  version = "6.2.1908";
  format = "other";

  src = fetchFromGitHub {
    owner = "NGSolve";
    repo = "ngsolve";
    rev = "v${version}";
    sha256 = "173nsyq82zah0phyrvha31mc3dwsmdlygqfz1da6w2mqidmzr7cf";
    fetchSubmodules = true;
  };

  __noChroot = false;

  nativeBuildInputs = [ cmake makeWrapper ];

  buildInputs = [
    tk tix openblasCompat liblapackWithoutAtlas suitesparse
    mesa_glu xorg.libXmu xorg.libX11 scalapack
  ];

  # propagatedBuildInputs = [ tkinter ];

  patches = [ ./ngsolve.patch ];

  cmakeFlags = [
    "-DUSE_UMFPACK=ON"
    "-DBUILD_UMFPACK=OFF"
    "-DSuiteSparse=${suitesparse}"
    "-DSCALAPACK_LIBRARY=${scalapack}/lib/libscalapack.so"
    "-Dgit_version_string=v${version}-0-g${substring 0 8 src.rev}"
  ];

  postInstall = ''
    chmod +x $out/lib/*.so
    wrapProgram $out/bin/netgen \
      --prefix PYTHONPATH : ".:$out/${python.sitePackages}:$out/lib" \
      --prefix TCLLIBPATH " " "${tix}/lib" \
      --set NETGENDIR "$out/bin" \
      --set LOCALE_ARCHIVE "${glibcLocales}/lib/locale/locale-archive"
  '';

  setupHook = writeText "ngsolve-setup-hook" ''
    export PYTHONPATH="@out@/lib''${PYTHONPATH:+:}$PYTHONPATH"
    export TCLLIBPATH="${tix}/lib''${TCLLIBPATH:+ }$TCLLIBPATH"
    export NETGENDIR="@out@/bin"
    export LOCALE_ARCHIVE="${glibcLocales}/lib/locale/locale-archive"
  '';

}
