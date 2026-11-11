"""
ATU source package.

Intentionally free of imports: eagerly importing `src.train` here made
`python -m src.train` emit a RuntimeWarning (module already in sys.modules before
execution). Import submodules directly, e.g. `from src.train import run_pipeline`.
"""
