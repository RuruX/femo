"""Unit tests for mesh import helper contracts."""

import types


def test_import_mesh_parses_association_table_and_returns_expected_tuple(tmp_path, fresh_import):
    """Verify import_mesh parses associations and return arity for both modes."""
    utils = fresh_import("femo.fea.utils_dolfinx")

    assoc = tmp_path / "mesh_association_table.ini"
    assoc.write_text("[ASSOCIATION TABLE]\nleft=1\nright=2\n", encoding="utf-8")

    class FakeXDMF:
        def __init__(self, comm, path, mode):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read_mesh(self, name="Grid"):
            return types.SimpleNamespace(topology=types.SimpleNamespace(dim=2, create_connectivity=lambda *a, **k: None))

        def read_meshtags(self, mesh, name="Grid"):
            return "boundaries" if "boundaries" in self.path else "subdomains"

    utils.XDMFFile = FakeXDMF

    mesh, boundaries, association = utils.import_mesh(prefix="mesh", directory=str(tmp_path), subdomains=False)
    assert boundaries == "boundaries"
    assert association == {"left": 1, "right": 2}

    mesh, boundaries, subdomains, association = utils.import_mesh(prefix="mesh", directory=str(tmp_path), subdomains=True)
    assert boundaries == "boundaries"
    assert subdomains == "subdomains"
    assert association["left"] == 1
