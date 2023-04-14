"""Configuration file for nox."""

import nox

PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10", "3.11"]


@nox.session(python=PYTHON_VERSIONS, tags=["ci"])
def test(session):
    """Run CI tests."""
    session.env["QISKIT_SUPPRESS_PACKAGING_WARNINGS"] = "Y"
    session.install("-r", "requirements-dev.txt")
    session.install("-e", ".")
    posargs = {}
    if session.posargs:
        posargs = session.posargs
    session.run("stestr", "run", *posargs)


@nox.session(python=PYTHON_VERSIONS, tags=["cron"])
def test_terra_main(session):
    """Run CI tests against terra main branch."""
    session.env["QISKIT_SUPPRESS_PACKAGING_WARNINGS"] = "Y"
    session.install("git+https://github.com/Qiskit/qiskit-terra", "-r", "requirements-dev.txt")
    session.install("-e", ".")
    posargs = {}
    if session.posargs:
        posargs = session.posargs
    session.run("stestr", "run", *posargs)


@nox.session(tags=["style"])
def black(session):
    """Runs black."""
    session.env["QISKIT_SUPPRESS_PACKAGING_WARNINGS"] = "Y"
    session.install("-r", "requirements-dev.txt")
    session.install("-e", ".")
    session.run("black", "qiskit_experiments", "test", "tools", "setup.py", "docs/conf.py")


@nox.session(tags=["docs", "ci"])
def docs(session):
    """Build the full docs."""
    session.env["QISKIT_SUPPRESS_PACKAGING_WARNINGS"] = "Y"
    session.install("-r", "requirements-dev.txt")
    session.install(".")
    session.run(
        "sphinx-build", "-T", "-W", "--keep-going", "-b", "html", "docs/", "docs/_build/html"
    )


@nox.session(tags=["docs"])
def docs_minimal(session):
    """Build the docs without executing code in Jupyter Sphinx cells."""
    session.env["QISKIT_DOCS_SKIP_EXECUTE"] = "1"
    session.install("-r", "requirements-dev.txt")
    session.install(".")
    session.run(
        "sphinx-build", "-T", "-W", "--keep-going", "-b", "html", "docs/", "docs/_build/html"
    )


@nox.session(tags=["docs"])
def docs_parallel(session):
    """Build the full docs in parallel."""
    session.install("-r", "requirements-dev.txt")
    session.install(".")
    session.run(
        "sphinx-build",
        "-j",
        "auto",
        "-T",
        "-W",
        "--keep-going",
        "-b",
        "html",
        "docs/",
        "docs/_build/html",
    )


@nox.session(tags=["style", "lint", "ci"])
def lint(session):
    """Run black and pylint."""
    session.env["QISKIT_SUPPRESS_PACKAGING_WARNINGS"] = "Y"
    session.install("-r", "requirements-dev.txt")
    session.install("-e", ".")
    session.run(
        "black", "--check", "qiskit_experiments", "test", "tools", "setup.py", "docs/conf.py"
    )
    session.run(
        "pylint",
        "-rn",
        "-j",
        "0",
        "--rcfile=.pylintrc",
        "qiskit_experiments/",
        "test/",
        "tools/",
        "docs/conf.py",
    )
    session.run(
        "python",
        "tools/verify_headers.py",
    )


@nox.session(tags=["lint"])
def lint_incr(session):
    """Runs lint only on changes compared against the main branch."""
    session.env["QISKIT_SUPPRESS_PACKAGING_WARNINGS"] = "Y"
    session.install("-r", "requirements-dev.txt")
    session.install("-e", ".")
    session.run(
        "black", "--check", "qiskit_experiments", "test", "tools", "setup.py", "docs/conf.py"
    )
    session.run(
        "git",
        "fetch",
        "-q",
        "https://github.com/Qiskit/qiskit-experiments",
        ":lint_incr_latest",
        external=True,
    )
    session.run(
        "python",
        "tools/pylint_incr.py",
        "-rn",
        "-j4",
        "-sn",
        "--paths",
        ":/qiskit_experiments/*.py",
        ":/test/*.py",
        ":/tools/*.py",
    )
    session.run(
        "python",
        "tools/verify_headers.py",
        "qiskit_experiments",
        "test",
        "tools",
    )
