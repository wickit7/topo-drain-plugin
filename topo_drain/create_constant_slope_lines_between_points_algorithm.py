"""
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterEnum,
                       QgsProcessingUtils,
                       QgsWkbTypes,
                       QgsFields,
                       QgsField,
                       QgsFeature,
                       QgsGeometry,
                       QgsPointXY,
                       QgsProject)
from qgis import processing
from PyQt5.QtCore import QVariant
import os
import tempfile
import shutil
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString

from .core.topo_drain_core import TopoDrainCore
from .utils import get_crs_from_layer, update_core_crs_if_needed

pluginPath = os.path.dirname(__file__)


class CreateConstantSlopeLinesBetweenPointsAlgorithm(QgsProcessingAlgorithm):
    """
    This algorithm creates constant slope lines between two specified points.
    """

    # Constants used to refer to parameters and outputs
    INPUT_DTM = 'INPUT_DTM'
    INPUT_START_POINT = 'INPUT_START_POINT'
    INPUT_END_POINT = 'INPUT_END_POINT'
    INPUT_BARRIER_FEATURES = 'INPUT_BARRIER_FEATURES'
    OUTPUT_SLOPE_LINES = 'OUTPUT_SLOPE_LINES'
    SLOPE_DEVIATION_THRESHOLD = 'SLOPE_DEVIATION_THRESHOLD'
    MAX_ITERATIONS_SLOPE = 'MAX_ITERATIONS_SLOPE'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        instance = CreateConstantSlopeLinesBetweenPointsAlgorithm(core=self.core)
        if hasattr(self, 'plugin'):
            instance.plugin = self.plugin
        return instance

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'create_constant_slope_lines_between_points'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Create Constant Slope Lines Between Points')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr('Slope Analysis')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'slope_analysis'

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it.
        """
        return self.tr("""
        <h3>Create Constant Slope Lines Between Points</h3>
        <p>This algorithm creates constant slope lines between two specified points using cost-distance analysis with iterative refinement.</p>
        
        <h4>Description</h4>
        <p>The algorithm calculates the optimal path between a start point and end point that maintains a constant slope. It uses a hybrid approach that:</p>
        <ul>
        <li>Calculates the target slope from euclidean distance and elevation difference</li>
        <li>Uses cost-distance analysis to find the optimal path</li>
        <li>Iteratively refines the slope calculation for improved accuracy</li>
        </ul>
        
        <h4>Parameters</h4>
        <ul>
        <li><b>DTM</b>: Digital Terrain Model raster layer for elevation data</li>
        <li><b>Start Point</b>: Point layer containing the starting point</li>
        <li><b>End Point</b>: Point layer containing the ending point</li>
        <li><b>Barrier Features</b> (optional): Line or polygon features to avoid during path calculation</li>
        <li><b>Slope Deviation Threshold</b> (Advanced): Maximum allowed deviation from target slope (0.0-1.0)</li>
        <li><b>Max Iterations</b> (Advanced): Maximum number of refinement iterations (1-50)</li>
        </ul>
        
        <h4>Output</h4>
        <ul>
        <li><b>Slope Lines</b>: LineString features representing the constant slope path</li>
        </ul>
        """)
    
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'icons', 'topo_drain.svg'))
    
    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm.
        """
        
        # DTM input
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DTM,
                self.tr('DTM'),
                defaultValue=None
            )
        )

        # Start point input
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_START_POINT,
                self.tr('Start Point'),
                [QgsProcessing.TypeVectorPoint],
                defaultValue=None
            )
        )

        # End point input
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_END_POINT,
                self.tr('End Point'),
                [QgsProcessing.TypeVectorPoint],
                defaultValue=None
            )
        )

        # Barrier features input (optional)
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_BARRIER_FEATURES,
                self.tr('Barrier Features (optional)'),
                [QgsProcessing.TypeVectorLine, QgsProcessing.TypeVectorPolygon],
                optional=True,
                defaultValue=None
            )
        )

        # Slope parameter - REMOVED: slope is calculated automatically from elevation difference

        # Advanced parameters
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE_DEVIATION_THRESHOLD,
                self.tr('Advanced: Slope Deviation Threshold'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.2,
                minValue=0.0,
                maxValue=1.0
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ITERATIONS_SLOPE,
                self.tr('Advanced: Max Iterations'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1,
                maxValue=50
            )
        )

        # Output
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_SLOPE_LINES,
                self.tr('Slope Lines'),
                type=QgsProcessing.TypeVectorLine,
                defaultValue=None
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        try:
            # Validate and read input parameters
            dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
            start_point_source = self.parameterAsSource(parameters, self.INPUT_START_POINT, context)
            end_point_source = self.parameterAsSource(parameters, self.INPUT_END_POINT, context)
            barrier_source = self.parameterAsSource(parameters, self.INPUT_BARRIER_FEATURES, context)
            slope_deviation_threshold = self.parameterAsDouble(parameters, self.SLOPE_DEVIATION_THRESHOLD, context)
            max_iterations_slope = self.parameterAsInt(parameters, self.MAX_ITERATIONS_SLOPE, context)

            if feedback:
                feedback.pushInfo("[CreateConstantSlopeLinesBetweenPoints] Starting algorithm...")

            # Validate inputs
            if dtm_layer is None:
                raise QgsProcessingException("DTM layer is required")
            if start_point_source is None:
                raise QgsProcessingException("Start point layer is required")
            if end_point_source is None:
                raise QgsProcessingException("End point layer is required")

            # Check that start and end point layers have exactly one feature each
            if start_point_source.featureCount() != 1:
                raise QgsProcessingException("Start point layer must contain exactly one point feature or use selection of one point")
            if end_point_source.featureCount() != 1:
                raise QgsProcessingException("End point layer must contain exactly one point feature or use selection of one point")

            # Get DTM file path and validate
            dtm_path = dtm_layer.source()
            if not dtm_path or not os.path.exists(dtm_path):
                raise QgsProcessingException(f"DTM file not found: {dtm_path}")
            
            if feedback:
                feedback.pushInfo(f"[CreateConstantSlopeLinesBetweenPoints] DTM path: {dtm_path}")

            # Read CRS from the DTM using QGIS layer with safe fallback
            feedback.pushInfo("Reading CRS from DTM...")
            dtm_crs = get_crs_from_layer(dtm_layer, fallback_crs="EPSG:2056")
            feedback.pushInfo(f"DTM Layer crs: {dtm_crs}")

            # Ensure WhiteboxTools is configured before running
            if hasattr(self, 'plugin') and self.plugin:
                try:
                    if not self.plugin.ensure_whiteboxtools_configured():
                        feedback.pushWarning("WhiteboxTools is not configured. Please install and configure the WhiteboxTools for QGIS plugin.")
                except Exception as e:
                    feedback.pushWarning(f"Could not check WhiteboxTools configuration: {e}")
            else:
                # Try to automatically find and connect to the TopoDrain plugin
                feedback.pushInfo("Plugin reference not available - attempting to connect to TopoDrain plugin")
                try:
                    from qgis.utils import plugins
                    if 'topo_drain' in plugins:
                        topo_drain_plugin = plugins['topo_drain']
                        # Set the plugin reference for this instance
                        self.plugin = topo_drain_plugin
                        feedback.pushInfo("Successfully connected to TopoDrain plugin")
                        
                        # Now try to configure WhiteboxTools
                        if hasattr(topo_drain_plugin, 'ensure_whiteboxtools_configured'):
                            if not topo_drain_plugin.ensure_whiteboxtools_configured():
                                feedback.pushWarning("WhiteboxTools is not configured. Please install and configure the WhiteboxTools for QGIS plugin.")
                            else:
                                feedback.pushInfo("WhiteboxTools configuration verified")
                        else:
                            feedback.pushWarning("TopoDrain plugin found but configuration method not available")
                    else:
                        feedback.pushWarning("TopoDrain plugin not found in QGIS registry - cannot verify WhiteboxTools configuration")
                except Exception as e:
                    feedback.pushWarning(f"Could not connect to TopoDrain plugin: {e} - continuing without WhiteboxTools verification")

            # Update core CRS if needed (dtm_crs is guaranteed to be valid)
            update_core_crs_if_needed(self.core, dtm_crs, feedback)

            # Extract start and end points
            start_feature = next(start_point_source.getFeatures())
            end_feature = next(end_point_source.getFeatures())
            
            start_geom = start_feature.geometry()
            end_geom = end_feature.geometry()
            
            if start_geom.isEmpty() or end_geom.isEmpty():
                raise QgsProcessingException("Start and end point geometries cannot be empty")

            start_point_xy = start_geom.asPoint()
            end_point_xy = end_geom.asPoint()
            
            start_point = Point(start_point_xy.x(), start_point_xy.y())
            end_point = Point(end_point_xy.x(), end_point_xy.y())

            if feedback:
                feedback.pushInfo(f"[CreateConstantSlopeLinesBetweenPoints] Start point: {start_point}")
                feedback.pushInfo(f"[CreateConstantSlopeLinesBetweenPoints] End point: {end_point}")

            # Convert barrier features to GeoDataFrame list if provided
            barrier_features = []
            if barrier_source is not None:
                if feedback:
                    feedback.pushInfo("[CreateConstantSlopeLinesBetweenPoints] Processing barrier features...")
                
                # Convert QGIS vector layer to GeoDataFrame with better CRS handling
                barrier_gdf = self._qgis_source_to_geodataframe(barrier_source, context)
                if not barrier_gdf.empty:
                    # Ensure CRS consistency
                    if barrier_gdf.crs is None:
                        barrier_gdf.crs = dtm_crs
                    barrier_gdf = barrier_gdf.to_crs(self.core.crs)
                    barrier_features = [barrier_gdf]
                    feedback.pushInfo(f"[CreateConstantSlopeLinesBetweenPoints] Processed {len(barrier_gdf)} barrier features")

            if feedback:
                feedback.pushInfo("[CreateConstantSlopeLinesBetweenPoints] Running constant slope line calculation...")

            # Run the constant slope line calculation using the core instance
            result_line = self.core.get_constant_slope_line_between_points(
                dtm_path=dtm_path,
                start_point=start_point,
                end_point=end_point,
                barrier_features=barrier_features,
                slope_deviation_threshold=slope_deviation_threshold,
                max_iterations_slope=max_iterations_slope,
                feedback=feedback
            )

            if result_line is None:
                raise QgsProcessingException("No valid constant slope line could be generated between the specified points")

            # Prepare output
            fields = QgsFields()
            fields.append(QgsField('id', QVariant.Int))
            fields.append(QgsField('length', QVariant.Double))
            fields.append(QgsField('start_elev', QVariant.Double))
            fields.append(QgsField('end_elev', QVariant.Double))
            fields.append(QgsField('slope', QVariant.Double))

            (sink, dest_id) = self.parameterAsSink(
                parameters,
                self.OUTPUT_SLOPE_LINES,
                context,
                fields,
                QgsWkbTypes.LineString,
                dtm_layer.crs()
            )

            if sink is None:
                raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT_SLOPE_LINES))

            # Calculate actual slope from elevation difference and path length
            with rasterio.open(dtm_path) as src:
                start_coords = [(start_point.x, start_point.y)]
                end_coords = [(end_point.x, end_point.y)]
                
                start_elevations = list(src.sample(start_coords))
                end_elevations = list(src.sample(end_coords))
                
                start_elevation = float(start_elevations[0][0])
                end_elevation = float(end_elevations[0][0])

            elevation_difference = end_elevation - start_elevation
            actual_slope = elevation_difference / result_line.length if result_line.length > 0 else 0

            # Create output feature
            feature = QgsFeature()
            feature.setFields(fields)
            
            # Convert Shapely LineString to QGIS geometry
            coords = list(result_line.coords)
            qgs_points = [QgsPointXY(x, y) for x, y in coords]
            qgs_geom = QgsGeometry.fromPolylineXY(qgs_points)
            
            feature.setGeometry(qgs_geom)
            feature.setAttributes([
                1,  # id
                result_line.length,  # length
                start_elevation,  # start_elev
                end_elevation,  # end_elev
                actual_slope  # slope
            ])

            sink.addFeature(feature, QgsFeatureSink.FastInsert)

            if feedback:
                feedback.pushInfo(f"[CreateConstantSlopeLinesBetweenPoints] Successfully created constant slope line with length: {result_line.length:.2f}m")

            return {self.OUTPUT_SLOPE_LINES: dest_id}

        except Exception as e:
            if feedback:
                feedback.pushInfo(f"[CreateConstantSlopeLinesBetweenPoints] Error: {str(e)}")
            raise QgsProcessingException(str(e))

    def _qgis_source_to_geodataframe(self, source, context):
        """
        Convert a QGIS vector source to a GeoDataFrame.
        """
        from shapely.wkt import loads

        features = []
        for feature in source.getFeatures():
            geom = feature.geometry()
            if not geom.isEmpty():
                # Convert QGIS geometry to Shapely geometry
                wkt = geom.asWkt()
                shapely_geom = loads(wkt)
                
                # Get attributes
                attrs = {}
                for field in feature.fields():
                    field_name = field.name()
                    attrs[field_name] = feature[field_name]
                
                attrs['geometry'] = shapely_geom
                features.append(attrs)

        if not features:
            return gpd.GeoDataFrame()

        gdf = gpd.GeoDataFrame(features)
        
        # Set CRS from source
        source_crs = source.sourceCrs()
        if source_crs.isValid():
            gdf.crs = source_crs.authid()

        return gdf