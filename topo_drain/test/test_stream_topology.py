"""Regression tests for short links between stream confluences."""

import os
import tempfile
import unittest

import geopandas as gpd
import numpy as np
from osgeo import gdal
from shapely.geometry import LineString

from topo_drain.core.topo_drain_core import TopoDrainCore


class Feedback:
    def __init__(self):
        self.messages = []
        self.warnings = []

    def pushInfo(self, message):
        self.messages.append(message)

    def pushWarning(self, message):
        self.warnings.append(message)


class TestStreamTopology(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.core = TopoDrainCore.__new__(TopoDrainCore)
        self.core.temp_directory = self.temp.name
        self.core.gdal_driver_mapping = {'.tif': 'GTiff'}
        self.feedback = Feedback()
        self.path = os.path.join(self.temp.name, 'facc.tif')
        self.values = np.zeros((9, 9))
        self.values[1, 2] = 40
        self.values[1, 6] = 50
        self.values[3, 4] = 100
        self.values[4, 5:8] = 20
        self.values[4:8, 4] = 20000
        self.streams = gpd.GeoDataFrame(
            {'LINK_ID': [628, 630, 629, 100, 101]},
            geometry=[
                LineString([(4.5, 5.5), (4.5, 4.5)]),
                LineString([(7.5, 4.5), (4.5, 4.5)]),
                LineString([(4.5, 4.5), (4.5, 1.5)]),
                LineString([(2.5, 7.5), (4.5, 5.5)]),
                LineString([(6.5, 7.5), (4.5, 5.5)]),
            ],
        )

    def run_topology(self):
        ds = gdal.GetDriverByName('GTiff').Create(
            self.path, 9, 9, 1, gdal.GDT_Float64
        )
        ds.SetGeoTransform((0, 1, 0, 9, 0, -1))
        ds.GetRasterBand(1).SetNoDataValue(-9999)
        ds.GetRasterBand(1).WriteArray(self.values)
        ds = None
        return self.core._assign_stream_topology(
            self.streams, self.path, snap_tolerance=0.25, feedback=self.feedback
        ).set_index('LINK_ID')

    def test_two_cell_link_continues_main_stem(self):
        result = self.run_topology()
        self.assertEqual(result.loc[628, 'DS_LINK_ID'], 629)
        self.assertEqual(result.loc[628, 'TRIB_ID'], result.loc[629, 'TRIB_ID'])
        self.assertNotEqual(result.loc[630, 'TRIB_ID'], result.loc[629, 'TRIB_ID'])
        self.assertTrue(any('628' in message for message in self.feedback.messages))
        self.assertFalse(self.feedback.warnings)

    def test_downstream_sum_does_not_override_larger_competitor(self):
        self.values[4, 5:8] = 200
        result = self.run_topology()
        self.assertEqual(result.loc[630, 'TRIB_ID'], result.loc[629, 'TRIB_ID'])
        self.assertNotEqual(result.loc[628, 'TRIB_ID'], result.loc[629, 'TRIB_ID'])

    def test_invalid_start_value_is_warned_not_used(self):
        for value in (-9999, 0, float('nan'), float('inf')):
            with self.subTest(value=value):
                self.feedback = Feedback()
                self.values[3, 4] = value
                result = self.run_topology()
                self.assertEqual(result.loc[630, 'TRIB_ID'], result.loc[629, 'TRIB_ID'])
                self.assertTrue(any('LINK_ID=628' in msg for msg in self.feedback.warnings))

    def test_exclusive_cells_take_precedence_over_startpoint(self):
        self.streams.at[0, 'geometry'] = LineString(
            [(4.5, 6.5), (4.5, 5.5), (4.5, 4.5)]
        )
        self.streams.at[3, 'geometry'] = LineString([(2.5, 7.5), (4.5, 6.5)])
        self.streams.at[4, 'geometry'] = LineString([(6.5, 7.5), (4.5, 6.5)])
        self.values[2, 4] = 10
        self.values[4, 5:8] = 50
        result = self.run_topology()
        self.assertEqual(result.loc[628, 'TRIB_ID'], result.loc[629, 'TRIB_ID'])
        self.assertFalse(any('LINK_ID=628' in msg for msg in self.feedback.messages))

    def test_same_cell_endpoints_do_not_sample_downstream_sum(self):
        self.streams.at[0, 'geometry'] = LineString([(4.6, 4.6), (4.5, 4.5)])
        self.run_topology()
        self.assertTrue(any('LINK_ID=628' in msg for msg in self.feedback.warnings))
        self.assertFalse(any('LINK_ID=628' in msg for msg in self.feedback.messages))

    def test_outside_raster_startpoint_is_warned(self):
        self.streams.at[0, 'geometry'] = LineString([(-0.5, 4.5), (4.5, 4.5)])
        self.values[:] = 0
        self.values[4, 4] = 20000
        self.run_topology()
        self.assertTrue(any('LINK_ID=628' in msg for msg in self.feedback.warnings))


if __name__ == '__main__':
    unittest.main()
