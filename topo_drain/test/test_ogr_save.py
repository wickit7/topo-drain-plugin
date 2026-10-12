"""OGR export tests using real drivers and minimal QGIS import substitutes."""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
from osgeo import ogr
from shapely.geometry import LineString


def load_utils():
    path = os.path.join(os.path.dirname(__file__), '..', 'utils.py')
    spec = importlib.util.spec_from_file_location('ogr_save_test_utils', path)
    module = importlib.util.module_from_spec(spec)
    try:
        import qgis.core
    except ModuleNotFoundError as exc:
        if exc.name not in ('qgis', 'qgis.core'):
            raise
        qgis_module = types.ModuleType('qgis')
        core_module = types.ModuleType('qgis.core')
        for name in (
            'QgsProject', 'QgsRasterLayer', 'QgsVectorLayer', 'QgsFeatureSource',
            'QgsFeatureRequest', 'QgsProcessingFeatureSource',
        ):
            setattr(core_module, name, type(name, (), {}))
        core_module.QgsProcessingException = type('QgsProcessingException', (Exception,), {})
        qgis_module.core = core_module
        with patch.dict(sys.modules, {'qgis': qgis_module, 'qgis.core': core_module}):
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)
    return module


utils = load_utils()


class Feedback:
    def __init__(self):
        self.messages = []

    def pushInfo(self, message):
        self.messages.append(message)

    def pushWarning(self, message):
        self.messages.append(message)


class TestOgrSave(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.core = types.SimpleNamespace(
            crs='EPSG:2056',
            ogr_driver_mapping={'.shp': 'ESRI Shapefile', '.gpkg': 'GPKG', '.geojson': 'GeoJSON'},
        )
        self.feedback = Feedback()
        self.gdf = gpd.GeoDataFrame(
            {
                'change_after': [0.4, 0.6],
                'slope_after': [0.01, 0.02],
                'CHANGE_AFT': [3, 4],
                'change_after_extra': [5, 6],
            },
            geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        )

    def check_export(self, extension):
        path = os.path.join(self.temp.name, 'lines' + extension)
        original_columns = self.gdf.columns.tolist()
        with ogr.ExceptionMgr(useExceptions=True):
            utils.save_gdf_to_file_ogr(
                self.gdf, path, self.core, self.feedback, all_upper=True
            )
        self.assertEqual(self.gdf.columns.tolist(), original_columns)
        ds = ogr.Open(path)
        layer = ds.GetLayer(0)
        self.assertEqual(layer.GetFeatureCount(), 2)
        self.assertEqual(layer.GetSpatialRef().GetAuthorityCode(None), '2056')
        definition = layer.GetLayerDefn()
        names = [
            definition.GetFieldDefn(i).GetName()
            for i in range(definition.GetFieldCount())
        ]
        self.assertEqual(len(names), 4)
        self.assertEqual(len(set(names)), 4)
        for feature, expected in zip(layer, [(0.4, 0.01, 3, 5), (0.6, 0.02, 4, 6)]):
            for index, value in enumerate(expected):
                self.assertAlmostEqual(feature.GetField(index), value)
        feature = definition = layer = ds = None
        return names

    def test_shapefile_laundered_names_and_collisions(self):
        names = self.check_export('.shp')
        self.assertTrue(all(len(name) <= 10 for name in names))
        self.assertTrue(any('CHANGE_AFTER' in msg for msg in self.feedback.messages))

    def test_geopackage_preserves_long_names(self):
        self.assertEqual(
            self.check_export('.gpkg'),
            ['CHANGE_AFTER', 'SLOPE_AFTER', 'CHANGE_AFT', 'CHANGE_AFTER_EXTRA'],
        )

    def test_geojson_preserves_long_names(self):
        self.assertEqual(
            self.check_export('.geojson'),
            ['CHANGE_AFTER', 'SLOPE_AFTER', 'CHANGE_AFT', 'CHANGE_AFTER_EXTRA'],
        )

    def test_create_field_failure_is_reported(self):
        driver = MagicMock()
        layer = driver.CreateDataSource.return_value.CreateLayer.return_value
        layer.GetLayerDefn.return_value.GetFieldCount.return_value = 0
        layer.CreateField.return_value = ogr.OGRERR_FAILURE
        with patch.object(ogr, 'GetDriverByName', return_value=driver):
            with self.assertRaisesRegex(utils.QgsProcessingException, 'Could not create field'):
                utils.save_gdf_to_file_ogr(
                    self.gdf, os.path.join(self.temp.name, 'failed.shp'),
                    self.core, self.feedback
                )
        self.assertFalse(any('saved using OGR' in msg for msg in self.feedback.messages))

    def test_create_feature_failure_is_reported(self):
        driver = MagicMock()
        layer = driver.CreateDataSource.return_value.CreateLayer.return_value
        definition = ogr.FeatureDefn('lines')
        layer.GetLayerDefn.return_value = definition

        def create_field(field):
            definition.AddFieldDefn(field)
            return ogr.OGRERR_NONE

        layer.CreateField.side_effect = create_field
        layer.CreateFeature.return_value = ogr.OGRERR_FAILURE
        with patch.object(ogr, 'GetDriverByName', return_value=driver):
            with self.assertRaisesRegex(utils.QgsProcessingException, 'Could not write feature'):
                utils.save_gdf_to_file_ogr(
                    self.gdf, os.path.join(self.temp.name, 'failed.shp'),
                    self.core, self.feedback
                )
        self.assertFalse(any('saved using OGR' in msg for msg in self.feedback.messages))


if __name__ == '__main__':
    unittest.main()
