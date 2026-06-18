"""Unit tests for node index lookup utility."""

import numpy as np


def test_find_node_indices_returns_nearest(fresh_import):
    """Verify nearest-node indices are returned for requested coordinates."""
    utils = fresh_import("femo.fea.utils_dolfinx")

    class SimpleKDTree:
        def __init__(self, points):
            self.points = np.asarray(points)

        def query(self, nodes):
            nodes = np.asarray(nodes)
            idx = []
            dist = []
            for node in nodes:
                d = np.sqrt(((self.points - node) ** 2).sum(axis=1))
                i = int(np.argmin(d))
                idx.append(i)
                dist.append(d[i])
            return np.array(dist), np.array(idx)

    utils.KDTree = SimpleKDTree

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    nodes = np.array([[1.1, 0.0], [0.2, 0.0]])
    got = utils.findNodeIndices(nodes, coords)
    assert got.tolist() == [1, 0]
