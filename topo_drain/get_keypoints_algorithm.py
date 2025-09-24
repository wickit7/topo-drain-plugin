# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: get_keypoints_algorithm.py
#
# Purpose: QGIS Processing Algorithm to detect keypoints along valley lines
#          based on curvature analysis of elevation profiles
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterNumber)
import geopandas as gpd
import os
from .utils import get_crs_from_layer, update_core_crs_if_needed, clean_qvariant_data

pluginPath = os.path.dirname(__file__)

class GetKeypointsAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for detecting keypoints along valley lines based on curvature analysis of elevation profiles.

    This algorithm identifies keypoints (points of high convexity) along valley lines by analyzing
    the curvature of elevation profiles extracted from a DTM. The elevation profile is extracted
    along each valley line and smoothed using a Savitzky-Golay filter. The second derivative
    (curvature) is then computed, and locations with the strongest convex curvature are selected
    as keypoints.

    The algorithm:
    - Extracts elevation profiles along each valley line using the DTM
    - Applies Savitzky-Golay smoothing to reduce noise
    - Computes the second derivative (curvature) of the elevation profile
    - Identifies inflection points where curvature changes from concave to convex
    - Selects the top N keypoints per valley line based on curvature strength
    - Ensures minimum distance between selected keypoints

    This is useful for identifying significant morphological features along drainage channels,
    such as knickpoints, channel transitions, or locations suitable for water retention
    structures in keyline design applications.
    """

    # Constants used to refer to parameters and outputs
    INPUT_VALLEY_LINES = 'INPUT_VALLEY_LINES'
    INPUT_DTM = 'INPUT_DTM'
    SMOOTHING_WINDOW = 'SMOOTHING_WINDOW'
    POLYORDER = 'POLYORDER'
    MIN_DISTANCE = 'MIN_DISTANCE'
    MAX_KEYPOINTS = 'MAX_KEYPOINTS'
    OUTPUT_KEYPOINTS = 'OUTPUT_KEYPOINTS'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        instance = GetKeypointsAlgorithm(core=self.core)
        if hasattr(self, 'plugin'):
            instance.plugin = self.plugin
        return instance

    def name(self):
        return 'get_keypoints'

    def displayName(self):
        return self.tr('Get Keypoints (along Main Valley Lines)')

    def group(self):
        return self.tr('Basic Watershed Analysis')

    def groupId(self):
        return 'basic_watershed_analysis'

    def shortHelpString(self):
        return self.tr(
            """Detect keypoints along valley lines based on curvature analysis of elevation profiles.
            
This algorithm identifies keypoints (points of high convexity) along valley lines by analyzing the curvature of elevation profiles extracted from a DTM. 

The algorithm:
- Extracts elevation profiles along each valley line using the DTM at pixel resolution
- Applies smoothing to reduce noise depending on the parameters "smoothing window" and "polyorder"
- Computes the second derivative (curvature) of the elevation profile
- Identifies inflection points where curvature changes from concave to convex
- Selects the top N keypoints ("Number of keypoints per valley line") per valley line based on curvature strength
- Ensures minimum distance between selected keypoints ("Minimum distance between keypoints (m)")

This is useful for identifying significant morphological features along drainage channels, such as knickpoints, channel transitions, or locations suitable for water retention structures in keyline design applications.

Input Requirements:
- Valley Lines: Must have 'LINK_ID' attribute (from Create Valleys algorithm). LINK_ID is the standard cross-platform identifier.
- DTM: Digital Terrain Model for elevation profile extraction

OUTPUT_KEYPOINTS:
Point layer containing detected keypoints with attributes: VALLEY_ID, ELEV_INDEX, RANK, CURVATURE

Simplified Parameters:
- Maximum keypoints: Limits output per valley line
- Minimum distance: Ensures spatial separation between keypoints
- Smoothing parameters: Control noise reduction in elevation profiles"""
        )

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'icons', 'topo_drain.svg'))

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm.
        """
        # Input valley lines
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_VALLEY_LINES,
                self.tr('Input Main Valley Lines (must have LINK_ID attribute)'),
                [QgsProcessing.TypeVectorLine]
            )
        )

        # Input DTM
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DTM,
                self.tr('Input DTM (Digital Terrain Model)')
            )
        )

        # Maximum keypoints per valley line
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_KEYPOINTS,
                self.tr('(Max.) Number of keypoints per valley line'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=2,
                minValue=1
           )
        )

        # Minimum distance between keypoints
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_DISTANCE,
                self.tr('Minimum distance between keypoints (m)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=1.0
            )
        )

        # Polynomial order
        self.addParameter(
            QgsProcessingParameterNumber(
                self.POLYORDER,
                self.tr('Advanced: Polynomial order for smoothing'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=2,
                minValue=1,
                maxValue=5
            )
        )
        
        # Smoothing window
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SMOOTHING_WINDOW,
                self.tr('Advanced: Smoothing window size (must be odd and larger than Polynomial order)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=9,
                minValue=3,
                maxValue=51
            )
        )

        # Output keypoints
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_KEYPOINTS,
                self.tr('Output Keypoints')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Validate and read input parameters
        valley_lines_source = self.parameterAsSource(parameters, self.INPUT_VALLEY_LINES, context)
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        smoothing_window = self.parameterAsInt(parameters, self.SMOOTHING_WINDOW, context)
        polyorder = self.parameterAsInt(parameters, self.POLYORDER, context)
        min_distance = self.parameterAsDouble(parameters, self.MIN_DISTANCE, context)
        max_keypoints = self.parameterAsInt(parameters, self.MAX_KEYPOINTS, context)

        # Validate smoothing parameters
        if smoothing_window <= polyorder:
            raise QgsProcessingException(f"Smoothing window ({smoothing_window}) must be larger than polynomial order ({polyorder})")
        
        if smoothing_window % 2 == 0:
            new_window = smoothing_window - 1
            feedback.reportError(f"Smoothing window ({smoothing_window}) must be odd. Adjusting to {new_window}.")
            smoothing_window = new_window

        # Get file paths
        dtm_path = dtm_layer.source()
        
        # Validate file existence
        if not dtm_path or not os.path.exists(dtm_path):
            raise QgsProcessingException(f"DTM file not found: {dtm_path}")

        # Get output file path using parameterAsOutputLayer
        keypoints_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_KEYPOINTS, context)
        keypoints_file_path = keypoints_output_layer if isinstance(keypoints_output_layer, str) else keypoints_output_layer

        feedback.pushInfo("Reading CRS from DTM layer...")
        # Read CRS from the DTM layer with safe fallback (since we can't get CRS from source directly)
        dtm_crs = get_crs_from_layer(dtm_layer, fallback_crs="EPSG:2056")
        feedback.pushInfo(f"DTM CRS: {dtm_crs}")

        # Update core CRS if needed (dtm_crs is guaranteed to be valid)
        update_core_crs_if_needed(self.core, dtm_crs, feedback)

        # Ensure WhiteboxTools is configured before running
        if hasattr(self, 'plugin') and self.plugin:
            if not self.plugin.ensure_whiteboxtools_configured():
                raise QgsProcessingException("WhiteboxTools is not configured. Please install and configure the WhiteboxTools for QGIS plugin.")
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

        # Load input data as GeoDataFrame
        feedback.pushInfo("Loading valley lines...")
        valley_lines_gdf = gpd.GeoDataFrame.from_features(valley_lines_source.getFeatures())
        
        if valley_lines_gdf.empty:
            raise QgsProcessingException("No features found in valley lines input")

        # Clean QVariant objects from valley lines data to avoid field type errors
        feedback.pushInfo("Cleaning valley lines data types...")
        valley_lines_gdf = clean_qvariant_data(valley_lines_gdf)

        # Check for LINK_ID attribute (required)
        if 'LINK_ID' not in valley_lines_gdf.columns:
            raise QgsProcessingException("Valley lines must have a 'LINK_ID' attribute. Please use valley lines generated by the Create Valleys algorithm.")

        # Run keypoint detection
        feedback.pushInfo("Running keypoint detection...")
        keypoints_gdf = self.core.get_keypoints(
            valley_lines=valley_lines_gdf,
            dtm_path=dtm_path,
            smoothing_window=smoothing_window,
            polyorder=polyorder,
            min_distance=min_distance,
            max_keypoints=max_keypoints,
            feedback=feedback
        )

        if keypoints_gdf.empty:
            raise QgsProcessingException("No keypoints were detected")

        feedback.pushInfo(f"Detected {len(keypoints_gdf)} keypoints")

        # Ensure the keypoints GeoDataFrame has the correct CRS
        keypoints_gdf = keypoints_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Keypoints CRS: {keypoints_gdf.crs}")

        # Clean any QVariant objects from the GeoDataFrame before saving
        feedback.pushInfo("Cleaning keypoints data types before saving...")
        keypoints_gdf = clean_qvariant_data(keypoints_gdf)

        # Save result
        try:
            keypoints_gdf.to_file(keypoints_file_path)
            feedback.pushInfo(f"Keypoints saved to: {keypoints_file_path}")
        except Exception as e:
            raise QgsProcessingException(f"Failed to save keypoints output: {e}")

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters:
                results[outputName] = parameters[outputName]

        return results
