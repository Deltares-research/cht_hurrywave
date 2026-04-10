"""
Utility to convert a regular HurryWave grid to an xugrid Ugrid2d object.

Provides :func:`xug`, which builds an ``xu.Ugrid2d`` from the grid metadata
of a HurryWave domain object (reprojected to Web Mercator).
"""

import time

import numpy as np
import xugrid as xu
from pyproj import Transformer


def xug(grid) -> xu.Ugrid2d:
    """
    Build an ``xu.Ugrid2d`` from a regular HurryWave grid.

    Nodes are projected to EPSG:3857 (Web Mercator) before the unstructured
    grid object is constructed.

    Parameters
    ----------
    grid : object
        Object exposing ``x0``, ``y0``, ``nmax``, ``mmax``, ``dx``, ``dy``,
        and ``rotation`` attributes (typically a :class:`~.quadtree.QuadtreeMesh`
        or a plain namespace).

    Returns
    -------
    xu.Ugrid2d
        Unstructured 2-D grid in Web Mercator coordinates.
    """
    x0 = grid.x0
    y0 = grid.y0
    nmax = grid.nmax
    mmax = grid.mmax
    dx = grid.dx
    dy = grid.dy
    rotation = grid.rotation

    nr_cells = nmax * mmax
    cosrot = np.cos(rotation * np.pi / 180)
    sinrot = np.sin(rotation * np.pi / 180)

    cell_nm_indices = np.full(nr_cells, 0, dtype=int)
    nm_nodes = np.full(4 * nr_cells, 1e9, dtype=int)
    face_nodes = np.full((4, nr_cells), -1, dtype=int)
    node_x = np.full(4 * nr_cells, 1e9, dtype=float)
    node_y = np.full(4 * nr_cells, 1e9, dtype=float)
    nnodes = 0
    icel = 0

    tic = time.perf_counter()

    for m in range(mmax):
        for n in range(nmax):
            # Lower left
            nmind = m * (nmax + 1) + n
            nm_nodes[nnodes] = nmind
            face_nodes[0, icel] = nnodes
            node_x[nnodes] = x0 + cosrot * (m * dx) - sinrot * (n * dy)
            node_y[nnodes] = y0 + sinrot * (m * dx) + cosrot * (n * dy)
            nnodes += 1
            # Lower right
            nmind = (m + 1) * (nmax + 1) + n
            nm_nodes[nnodes] = nmind
            face_nodes[1, icel] = nnodes
            node_x[nnodes] = x0 + cosrot * ((m + 1) * dx) - sinrot * (n * dy)
            node_y[nnodes] = y0 + sinrot * ((m + 1) * dx) + cosrot * (n * dy)
            nnodes += 1
            # Upper right
            nmind = (m + 1) * (nmax + 1) + (n + 1)
            nm_nodes[nnodes] = nmind
            face_nodes[2, icel] = nnodes
            node_x[nnodes] = x0 + cosrot * ((m + 1) * dx) - sinrot * ((n + 1) * dy)
            node_y[nnodes] = y0 + sinrot * ((m + 1) * dx) + cosrot * ((n + 1) * dy)
            nnodes += 1
            # Upper left
            nmind = m * (nmax + 1) + (n + 1)
            nm_nodes[nnodes] = nmind
            face_nodes[3, icel] = nnodes
            node_x[nnodes] = x0 + cosrot * (m * dx) - sinrot * ((n + 1) * dy)
            node_y[nnodes] = y0 + sinrot * (m * dx) + cosrot * ((n + 1) * dy)
            nnodes += 1

            icel += 1

    toc = time.perf_counter()
    print(f"Found nodes in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()

    xxx, indx, irev = np.unique(nm_nodes, return_index=True, return_inverse=True)
    node_x = node_x[indx]
    node_y = node_y[indx]

    transformer = Transformer.from_crs(4326, 3857, always_xy=True)
    node_x, node_y = transformer.transform(node_x, node_y)

    for icel in range(nr_cells):
        for j in range(4):
            face_nodes[j, icel] = irev[face_nodes[j, icel]]

    toc = time.perf_counter()
    print(f"Get rid of duplicates {toc - tic:0.4f} seconds")

    nodes = np.transpose(np.vstack((node_x, node_y)))
    faces = np.transpose(face_nodes)
    fill_value = -1

    grid = xu.Ugrid2d(nodes[:, 0], nodes[:, 1], fill_value, faces)
    return grid
