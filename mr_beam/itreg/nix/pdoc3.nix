{
  buildPythonPackage, fetchPypi, setuptools-git, setuptools_scm, markupsafe,
  mako, markdown
}:

buildPythonPackage rec {
  pname = "pdoc3";
  version = "0.7.1";
  src = fetchPypi {
    inherit pname version;
    sha256 = "1jaq0x31v2gxihryc8mvqpkrx0jvbayv0qb5wzk5pd1ngbyg3ky2";
  };
  buildInputs = [ setuptools-git setuptools_scm ];
  propagatedBuildInputs = [ mako markdown markupsafe ];
}
