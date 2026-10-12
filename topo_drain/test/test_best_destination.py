"""Regression tests for selecting the best reachable destination cell."""

import os
import tempfile
import unittest

import numpy as np
from osgeo import gdal

from topo_drain.core.topo_drain_core import TopoDrainCore


def write_raster(path, data, gdal_type, nodata=None):
    ds = gdal.GetDriverByName('GTiff').Create(path, data.shape[1], data.shape[0], 1, gdal_type)
    ds.SetGeoTransform((0, 1, 0, data.shape[0], 0, -1))
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    ds = None


class TestSelectBestDestinationCell(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.acc_path = os.path.join(self.temp.name, 'acc.tif')
        self.dest_path = os.path.join(self.temp.name, 'dest.tif')
        self.best_path = os.path.join(self.temp.name, 'best.tif')

    def select(self, acc, dest, nodata=-32768.0):
        write_raster(self.acc_path, acc, gdal.GDT_Float64, nodata)
        write_raster(self.dest_path, dest, gdal.GDT_Byte)
        TopoDrainCore._select_best_destination_cell(self.acc_path, self.dest_path, self.best_path)
        best = gdal.Open(self.best_path).ReadAsArray()
        return [tuple(int(v) for v in cell) for cell in np.argwhere(best == 1)]

    def test_ignores_nodata_destination_cells(self):
        # Destination cell overlapping a barrier gets NoData accumulated cost from whitebox_workflows.
        acc = np.full((3, 3), 5.0)
        acc[0, 0] = -32768.0
        acc[2, 2] = 3.0
        dest = np.zeros((3, 3), dtype=np.uint8)
        dest[0, 0] = 1
        dest[2, 2] = 1
        self.assertEqual(self.select(acc, dest), [(2, 2)])

    def test_ignores_negative_and_nan_without_declared_nodata(self):
        acc = np.full((3, 3), 5.0)
        acc[0, 0] = -1.0
        acc[0, 1] = np.nan
        acc[1, 1] = 4.0
        dest = np.zeros((3, 3), dtype=np.uint8)
        dest[0, 0] = dest[0, 1] = dest[1, 1] = 1
        self.assertEqual(self.select(acc, dest, nodata=None), [(1, 1)])

    def test_raises_when_only_nodata_destinations(self):
        acc = np.full((3, 3), -32768.0)
        dest = np.zeros((3, 3), dtype=np.uint8)
        dest[1, 1] = 1
        with self.assertRaises(RuntimeError):
            self.select(acc, dest)


if __name__ == '__main__':
    unittest.main()
