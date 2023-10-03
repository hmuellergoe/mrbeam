{ buildPythonPackage, fetchPypi, markupsafe, mock, pytest }:

buildPythonPackage rec {
  pname = "Mako";
  version = "1.0.14";
  src = fetchPypi {
    inherit pname version;
    sha256 = "0jvznnyidyva7n7jw7pm42qpwxlhz5pjk2x6camnk4k9qpc459pm";
  };
  buildInputs = [ markupsafe mock pytest ];
}
