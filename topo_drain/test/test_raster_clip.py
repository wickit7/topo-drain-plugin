"""GDAL regression tests for the shared valleys/ridges perimeter clip."""

import math
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from osgeo import gdal, ogr, osr

from topo_drain.core.topo_drain_core import DTM_NODATA, TopoDrainCore


class Feedback:
    def __init__(self):
        self.messages = []
        self.progress = []
        self.cancelled = False

    def pushInfo(self, message):
        self.messages.append(message)

    def setProgress(self, progress):
        self.progress.append(progress)

    def isCanceled(self):
        return self.cancelled


class TestRasterClip(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source_path = os.path.join(self.temp.name, 'source.tif')
        self.mask_path = os.path.join(self.temp.name, 'mask.gpkg')
        self.output_path = os.path.join(self.temp.name, 'clipped.tif')
        self.crs = osr.SpatialReference()
        self.crs.ImportFromEPSG(3857)
        self.feedback = Feedback()
        # Clipping does not need runtime initialization.
        self.core = TopoDrainCore.__new__(TopoDrainCore)
        self.core.gdal_driver_mapping = {'.tif': 'GTiff'}
        self.create_mask('POLYGON ((0 0, 10 0, 0 10, 0 0))')

    def create_mask(self, wkt):
        if os.path.exists(self.mask_path):
            os.unlink(self.mask_path)
        ds = ogr.GetDriverByName('GPKG').CreateDataSource(self.mask_path)
        layer = ds.CreateLayer('mask', self.crs, ogr.wkbPolygon)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
        layer.CreateFeature(feature)
        feature = layer = ds = None

    def create_source(self, data_type=gdal.GDT_Float32, nodata=None):
        ds = gdal.GetDriverByName('GTiff').Create(
            self.source_path, 10, 10, 1, data_type
        )
        ds.SetGeoTransform((0, 1, 0, 10, 0, -1))
        ds.SetProjection(self.crs.ExportToWkt())
        values = np.full((10, 10), 100.0)
        values[9, 0] = 0.0
        if nodata is not None:
            values[8, 0] = nodata
            ds.GetRasterBand(1).SetNoDataValue(nodata)
        ds.GetRasterBand(1).WriteArray(values)
        ds = None

    def read_clip(self):
        result = self.core._clip_raster(
            self.source_path, self.mask_path, self.output_path,
            'EPSG:3857', self.feedback
        )
        self.assertEqual(result, self.output_path)
        ds = gdal.Open(result)
        try:
            band = ds.GetRasterBand(1)
            self.assertEqual(ds.GetGeoTransform(), (0, 1, 0, 10, 0, -1))
            self.assertTrue(ds.GetSpatialRef().IsSame(self.crs))
            return band.ReadAsArray(), band.GetNoDataValue(), band.DataType
        finally:
            band = ds = None

    def test_missing_nodata_preserves_zero_and_masks_exterior(self):
        self.create_source()
        data, nodata, data_type = self.read_clip()
        self.assertEqual(nodata, DTM_NODATA)
        self.assertEqual(data_type, gdal.GDT_Float32)
        self.assertEqual(int((data == nodata).sum()), 45)
        self.assertEqual(int((data == 0).sum()), 1)
        self.assertEqual(int((data == 100).sum()), 54)
        self.assertEqual(self.feedback.progress[-1], 10)

    def test_existing_nodata_is_preserved(self):
        for value in (-9999, -999999):
            with self.subTest(nodata=value):
                self.create_source(nodata=value)
                data, nodata, _ = self.read_clip()
                self.assertEqual(nodata, value)
                self.assertEqual(int((data == nodata).sum()), 46)
                self.assertEqual(data[9, 0], 0)

    def test_nan_nodata_is_replaced_by_finite_nodata(self):
        self.create_source(nodata=float('nan'))
        data, nodata, _ = self.read_clip()
        self.assertTrue(math.isfinite(nodata))
        self.assertFalse(np.isnan(data).any())
        self.assertEqual(int((data == nodata).sum()), 46)

    def test_unsigned_dtm_can_represent_negative_nodata(self):
        self.create_source(data_type=gdal.GDT_UInt16)
        data, nodata, data_type = self.read_clip()
        self.assertEqual(data_type, gdal.GDT_Float64)
        self.assertEqual(nodata, DTM_NODATA)
        self.assertEqual(int((data == nodata).sum()), 45)
        self.assertEqual(data[9, 0], 0)

    def test_unsigned_dtm_preserves_declared_zero_nodata(self):
        self.create_source(data_type=gdal.GDT_UInt16, nodata=0)
        data, nodata, _ = self.read_clip()
        self.assertEqual(nodata, 0)
        self.assertEqual(int((data == nodata).sum()), 47)

    def test_float64_elevation_precision_is_preserved(self):
        self.create_source(data_type=gdal.GDT_Float64)
        elevation = 123.1234567890123
        ds = gdal.Open(self.source_path, gdal.GA_Update)
        ds.GetRasterBand(1).WriteArray(np.array([[elevation]]), 0, 9)
        ds = None
        data, _, data_type = self.read_clip()
        self.assertEqual(data_type, gdal.GDT_Float64)
        self.assertEqual(data[9, 0], elevation)

    def test_legitimate_negative_elevation_is_preserved(self):
        self.create_source()
        ds = gdal.Open(self.source_path, gdal.GA_Update)
        ds.GetRasterBand(1).WriteArray(np.array([[-32768.0]]), 0, 9)
        ds = None
        data, nodata, _ = self.read_clip()
        self.assertEqual(data[9, 0], -32768)
        self.assertNotEqual(data[9, 0], nodata)

    def test_polygon_hole_is_nodata(self):
        self.create_source()
        self.create_mask(
            'POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), '
            '(3 3, 3 7, 7 7, 7 3, 3 3))'
        )
        data, nodata, _ = self.read_clip()
        self.assertEqual(int((data == nodata).sum()), 16)
        self.assertTrue((data[3:7, 3:7] == nodata).all())

    def test_failed_warp_raises(self):
        self.create_source()
        with patch('topo_drain.core.topo_drain_core.gdal.Warp', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Failed to clip DTM'):
                self.read_clip()

    def test_cancelled_warp_raises(self):
        self.create_source()
        self.feedback.cancelled = True
        with self.assertRaisesRegex(RuntimeError, 'cancelled by user'):
            self.read_clip()

    def test_unreadable_source_raises(self):
        with patch('topo_drain.core.topo_drain_core.gdal.Open', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Cannot open input DTM'):
                self.read_clip()

    def test_source_crs_metadata_and_optional_feedback(self):
        self.create_source()
        result = self.core._clip_raster(
            self.source_path, self.mask_path, self.output_path
        )
        ds = gdal.Open(result)
        try:
            self.assertTrue(ds.GetSpatialRef().IsSame(self.crs))
            self.assertEqual(ds.GetRasterBand(1).GetNoDataValue(), DTM_NODATA)
        finally:
            ds = None


if __name__ == '__main__':
    unittest.main()
