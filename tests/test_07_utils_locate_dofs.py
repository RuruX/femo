import numpy as np


def test_locate_dofs_converts_polar_input(fresh_import):
    utils = fresh_import("femo.fea.utils_dolfinx")

    captured = {}

    def fake_find(coords, all_coords):
        captured["coords"] = coords.copy()
        return np.array([0, 1])

    utils.findNodeIndices = fake_find

    class DummyV:
        def tabulate_dof_coordinates(self):
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    polar = np.array([0.0, 1.0, np.pi / 2, 1.0])
    out = utils.locateDOFs(polar, DummyV(), input="polar")
    assert np.allclose(captured["coords"], np.array([[1.0, 0.0], [0.0, 1.0]]), atol=1e-12)
    assert out.tolist() == [0, 1, 2, 3]


def test_locate_dofs_keeps_cartesian_input(fresh_import):
    utils = fresh_import("femo.fea.utils_dolfinx")

    captured = {}

    def fake_find(coords, all_coords):
        captured["coords"] = coords.copy()
        return np.array([1])

    utils.findNodeIndices = fake_find

    class DummyV:
        def tabulate_dof_coordinates(self):
            return np.array([[0.0, 0.0, 0.0], [2.0, 3.0, 0.0]])

    cart = np.array([2.0, 3.0])
    out = utils.locateDOFs(cart, DummyV(), input="cartesian")
    assert np.allclose(captured["coords"], np.array([[2.0, 3.0]]))
    assert out.tolist() == [2, 3]
