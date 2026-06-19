"""
Shared pytest fixtures for femo tests.

Unit tests (tests/unit/) use lightweight mock fixtures that avoid importing
dolfinx/petsc4py so they can run without a FEM installation.

Integration tests (tests/integration/) import dolfinx directly — they are
skipped automatically when dolfinx is not installed.
"""
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Minimal stubs used only by unit tests
# ---------------------------------------------------------------------------

class _MockFunctionSpace:
    def __init__(self, n=10):
        self.n = n


class _MockFunction:
    def __init__(self, name="u", n=10):
        self.name = name
        self.n = n
        self.function_space = _MockFunctionSpace(n)

        class _X:
            array = np.zeros(n)
        self.x = _X()

    def rename(self, label, _):
        self.name = label


@pytest.fixture
def mock_mesh():
    """A minimal mesh stand-in for unit tests."""
    class _Mesh:
        pass
    return _Mesh()


@pytest.fixture
def mock_function():
    return _MockFunction()


@pytest.fixture
def mock_function_space():
    return _MockFunctionSpace()
