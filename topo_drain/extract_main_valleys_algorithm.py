# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: extract_main_valleys_algorithm.py
#
# Purpose: QGIS Processing Algorithm to extract main valley lines based on flow accumulation
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterNumber, QgsProcessingParameterBoolean,
                       QgsProcessing, QgsProcessingParameterFeatureSource, QgsProcessingException)
import os
import geopandas as gpd
from .utils import get_crs_from_layer

pluginPath = os.path.dirname(__file__)

class ExtractMainValleysAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for extracting main valley lines based on flow accumulation and valley lines (generated previously with processing tool "Extract Valleys").

    This algorithm identifies the main valley lines (flow paths) from a complete valley (stream) network
    by selecting the tributaries with the highest flow accumulation values within a given
    perimeter (area of interest). If more than one feature polygon is inside perimeter, the analysis is done for each polygon separately. If no perimeter is provided, it uses the extent of the 
    valley lines.

    The algorithm:
    - Clips valley lines to the specified perimeter (or uses full extent if none provided)
    - Extracts flow accumulation values at valley line locations
    - Identifies point with highest flow accumulation for each tributary (defined by attribute TRIB_ID)
    - Selects the top N tributaries by maximum flow accumulation
    - Merges line segments belonging to each selected tributary (using attribut DS_LINK_ID)

    This is useful for identifying the most significant drainage channels (valleys) in a watershed
    or study area, focusing analysis on the primary valleys resp. flow paths.
    """

    INPUT_VALLEY_LINES = 'INPUT_VALLEY_LINES'
    INPUT_FACC_RASTER = 'INPUT_FACC_RASTER'
    INPUT_PERIMETER = 'INPUT_PERIMETER'
    OUTPUT_MAIN_VALLEYS = 'OUTPUT_MAIN_VALLEYS'
    NR_MAIN = 'NR_MAIN'
    CLIP_TO_PERIMETER = 'CLIP_TO_PERIMETER'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core
        
    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ExtractMainValleysAlgorithm(core=self.core)

    def name(self):
        return 'extract_main_valleys'

    def displayName(self):
        return self.tr('Extract Main Valleys')

    def group(self):
        return self.tr('Basic Watershed Analysis')

    def groupId(self):
        return 'basic_watershed_analysis'

    def shortHelpString(self):
        return self.tr(
            """QGIS Processing Algorithm for extracting main valley lines based on flow accumulation and valley lines (generated previously with processing tool "Extract Valleys").

This algorithm identifies the main valley lines (flow paths) from a complete valley (stream) network by selecting the tributaries with the highest flow accumulation values within a given perimeter (area of interest). If more than one feature polygon is inside perimeter, the analysis is done for each polygon separately. If no perimeter is provided, it uses the extent of the valley lines.

The algorithm:
- Clips valley lines to the specified perimeter (or uses full extent if none provided)
- Extracts flow accumulation values at valley line locations
- Identifies point with highest flow accumulation for each tributary (defined by attribute TRIB_ID)
- Selects the top N tributaries by maximum flow accumulation
- Merges line segments belonging to each selected tributary (using attribut DS_LINK_ID)

This is useful for identifying the most significant drainage channels (valleys) in a watershed or study area, focusing analysis on the primary valleys resp. flow paths.

Input Requirements:
- Valley Lines: Should have 'FID', 'TRIB_ID', and 'DS_LINK_ID' attributes (e.g., from Create Valleys algorithm)
- Flow Accumulation Raster: Raster showing accumulated flow (e.g., from Create Valleys algorithm)
- Perimeter (optional): Polygon defining the study area boundary. If not provided, uses the extent of valley lines"""
        )

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'icons', 'topo_drain.svg'))

    def initAlgorithm(self, config=None):        
        # Input parameters
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VALLEY_LINES,
                self.tr('Input Valley Lines (must have FID, TRIB_ID, DS_LINK_ID attributes)'),
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_FACC_RASTER,
                self.tr('Input Flow Accumulation Raster')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_PERIMETER,
                self.tr('Input Perimeter Polygon (area of interest)'),
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True
            )
        )
        
        # Algorithm parameters
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NR_MAIN,
                self.tr('Number of main valleys to extract'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=2,
                minValue=1
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CLIP_TO_PERIMETER,
                self.tr('Clip output to perimeter (if perimeter provided)'),
                defaultValue=True
            )
        )
        
        # Output parameters
        main_valleys_param = QgsProcessingParameterVectorDestination(
            self.OUTPUT_MAIN_VALLEYS,
            self.tr('Output Main Valley Lines'),
            type=QgsProcessing.TypeVectorLine,
            defaultValue=None
        )
        self.addParameter(main_valleys_param)

    def processAlgorithm(self, parameters, context, feedback):
        # Validate and read input parameters
        valley_lines_layer = self.parameterAsVectorLayer(parameters, self.INPUT_VALLEY_LINES, context)
        facc_raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_FACC_RASTER, context)
        perimeter_source = self.parameterAsSource(parameters, self.INPUT_PERIMETER, context)

        # Get file paths
        valley_lines_path = valley_lines_layer.source()
        facc_raster_path = facc_raster_layer.source()
        
        # Validate file existence
        if not valley_lines_path or not os.path.exists(valley_lines_path):
            raise QgsProcessingException(f"Valley lines file not found: {valley_lines_path}")
        if not facc_raster_path or not os.path.exists(facc_raster_path):
            raise QgsProcessingException(f"Flow accumulation raster not found: {facc_raster_path}")
        
        # Use parameterAsOutputLayer to preserve checkbox state information
        main_valleys_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_MAIN_VALLEYS, context)
        
        # Get algorithm parameters
        nr_main = self.parameterAsInt(parameters, self.NR_MAIN, context)
        clip_to_perimeter = self.parameterAsBool(parameters, self.CLIP_TO_PERIMETER, context)

        # Extract actual file path from layer object for processing
        main_valleys_file_path = main_valleys_output_layer if isinstance(main_valleys_output_layer, str) else main_valleys_output_layer

        feedback.pushInfo("Reading CRS from valley lines...")
        # Read CRS from the valley lines layer
        valley_crs = get_crs_from_layer(valley_lines_layer)
        feedback.pushInfo(f"Valley lines CRS: {valley_crs}")

        feedback.pushInfo("Processing extract_main_valleys via TopoDrainCore...")
        if not self.core:
            from topo_drain.core.topo_drain_core import TopoDrainCore
            feedback.reportError("TopoDrainCore not set, creating default instance.")
            self.core = TopoDrainCore()  # fallback: create default instance (not recommended for plugin use)

        # Check if self.core.crs matches valley_crs, warn and update if not
        if valley_crs:
            if self.core and hasattr(self.core, "crs"):
                if self.core.crs != valley_crs:
                    feedback.reportError(f"Warning: TopoDrainCore CRS ({self.core.crs}) does not match valley lines CRS ({valley_crs}). Updating TopoDrainCore CRS to match valley lines.")
                    self.core.crs = valley_crs

        # Ensure WhiteboxTools is configured before running
        if hasattr(self, 'plugin') and self.plugin:
            if not self.plugin.ensure_whiteboxtools_configured():
                raise QgsProcessingException("WhiteboxTools is not configured. Please install and configure the WhiteboxTools for QGIS plugin.")
        else:
            feedback.pushInfo("Warning: Plugin reference not available - WhiteboxTools configuration cannot be checked")

        # Load input data as GeoDataFrame
        feedback.pushInfo("Loading valley lines...")
        valley_lines_gdf = gpd.read_file(valley_lines_path)

        if valley_lines_gdf.empty:
            raise QgsProcessingException("No features found in valley lines input")

        # Load perimeter if provided, otherwise will be None (and core function will handle it)
        perimeter_gdf = None
        if perimeter_source:
            feedback.pushInfo("Loading perimeter...")
            perimeter_gdf = gpd.GeoDataFrame.from_features(perimeter_source.getFeatures())
        else:
            feedback.pushInfo("No perimeter provided, will use valley lines extent")

        if perimeter_gdf is not None and perimeter_gdf.empty:
            feedback.reportError("No features found in perimeter input")
                                 
        # Check for required attributes (case-insensitive)
        feedback.pushInfo(f"Checking valley lines attributes: {list(valley_lines_gdf.columns)}")
        required_attrs = ['FID', 'TRIB_ID', 'DS_LINK_ID']
        # Convert column names to uppercase for case-insensitive comparison
        available_attrs_upper = [col.upper() for col in valley_lines_gdf.columns]
        feedback.pushInfo(f"Valley lines attributes (uppercase): {available_attrs_upper}")
        missing_attrs = [attr for attr in required_attrs if attr not in available_attrs_upper]
        if missing_attrs:
            raise QgsProcessingException(f"Valley lines missing required attributes: {missing_attrs}. Please use output from Create Valleys algorithm. Available attributes: {list(valley_lines_gdf.columns)}")

        # Call the core function
        feedback.pushInfo("Running extract main valleys...")
        main_valleys_gdf = self.core.extract_main_valleys(
            valley_lines=valley_lines_gdf,
            facc_path=facc_raster_path,
            perimeter=perimeter_gdf,
            nr_main=nr_main,
            clip_to_perimeter=clip_to_perimeter,
            feedback=feedback
        )

        if main_valleys_gdf.empty:
            raise QgsProcessingException("No main valleys were detected")

        feedback.pushInfo(f"Created {len(main_valleys_gdf)} main valleys")

        # Ensure the main valleys GeoDataFrame has the correct CRS
        main_valleys_gdf = main_valleys_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Main valley lines CRS: {main_valleys_gdf.crs}")

        # Save result
        try:
            main_valleys_gdf.to_file(main_valleys_file_path)
            feedback.pushInfo(f"Main valley lines saved to: {main_valleys_file_path}")
        except Exception as e:
            raise QgsProcessingException(f"Failed to save main valleys output: {e}")

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters:
                results[outputName] = parameters[outputName]

        return results
