# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: extract_main_ridges_algorithm.py
#
# Purpose: QGIS Processing Algorithm to extract main ridge lines based on flow accumulation
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

class ExtractMainRidgesAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for extracting main ridge lines based on flow accumulation and ridge lines (generated previously with processing tool "Extract Ridges").

    This algorithm identifies the main ridge lines (watershed divides) from a complete ridge network
    by selecting the ridges with the highest flow accumulation values within a given
    perimeter (area of interest). If more than one feature polygon is inside perimeter, the analysis is done for each polygon separately. If no perimeter is provided, it uses the extent of the 
    ridge lines.

    The algorithm:
    - Clips ridge lines to the specified perimeter (or uses full extent if none provided)
    - Extracts flow accumulation values at ridge line locations
    - Identifies point with highest flow accumulation for each ridge (defined by attribute TRIB_ID)
    - Selects the top N ridges by maximum flow accumulation
    - Merges line segments belonging to each selected ridge (using attribut DS_LINK_ID)

    This is useful for identifying the most significant watershed divides (ridges) in a watershed
    or study area, focusing analysis on the primary ridges resp. drainage divides.
    """

    INPUT_RIDGE_LINES = 'INPUT_RIDGE_LINES'
    INPUT_FACC_RASTER = 'INPUT_FACC_RASTER'
    INPUT_PERIMETER = 'INPUT_PERIMETER'
    OUTPUT_MAIN_RIDGES = 'OUTPUT_MAIN_RIDGES'
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
        return ExtractMainRidgesAlgorithm(core=self.core)

    def name(self):
        return 'extract_main_ridges'

    def displayName(self):
        return self.tr('Extract Main Ridges')

    def group(self):
        return self.tr('Basic Watershed Analysis')

    def groupId(self):
        return 'basic_watershed_analysis'

    def shortHelpString(self):
        return self.tr(
            """QGIS Processing Algorithm for extracting main ridge lines based on flow accumulation and ridge lines (generated previously with processing tool "Extract Ridges").
            
This algorithm identifies the main ridge lines (watershed divides) from a complete ridge network by selecting the ridges with the highest flow accumulation values within a given perimeter (area of interest). If more than one feature polygon is inside perimeter, the analysis is done for each polygon separately. If no perimeter is provided, it uses the extent of the ridge lines.

The algorithm:
- Clips ridge lines to the specified perimeter (or uses full extent if none provided)
- Extracts flow accumulation values at ridge line locations
- Identifies point with highest flow accumulation for each ridge (defined by attribute TRIB_ID)
- Selects the top N ridges by maximum flow accumulation
- Merges line segments belonging to each selected ridge (using attribut DS_LINK_ID)

This is useful for identifying the most significant watershed divides (ridges) in a watershed or study area, focusing analysis on the primary ridges resp. drainage divides.

Input Requirements:
- Ridge Lines: Should have 'FID', 'TRIB_ID', and 'DS_LINK_ID' attributes (e.g., from Create Ridges algorithm)
- Flow Accumulation Raster: Raster showing accumulated flow (e.g., from Create Ridges algorithm, based on inverted DTM)
- Perimeter (optional): Polygon defining the study area boundary. If not provided, uses the extent of ridge lines"""
        )
    
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'icons', 'topo_drain.svg'))

    def initAlgorithm(self, config=None):        
        # Input parameters
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_RIDGE_LINES,
                self.tr('Input Ridge Lines (must have FID, TRIB_ID, DS_LINK_ID attributes)'),
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_FACC_RASTER,
                self.tr('Input Flow Accumulation Raster (from inverted DTM)')
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
                self.tr('Number of main ridges to extract'),
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
        main_ridges_param = QgsProcessingParameterVectorDestination(
            self.OUTPUT_MAIN_RIDGES,
            self.tr('Output Main Ridge Lines'),
            type=QgsProcessing.TypeVectorLine,
            defaultValue=None
        )
        self.addParameter(main_ridges_param)

    def processAlgorithm(self, parameters, context, feedback):
        # Validate and read input parameters
        ridge_lines_layer = self.parameterAsVectorLayer(parameters, self.INPUT_RIDGE_LINES, context)
        facc_raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_FACC_RASTER, context)
        perimeter_layer = self.parameterAsSource(parameters, self.INPUT_PERIMETER, context)

        # Get file paths
        ridge_lines_path = ridge_lines_layer.source()
        facc_raster_path = facc_raster_layer.source()
        
        # Validate file existence
        if not ridge_lines_path or not os.path.exists(ridge_lines_path):
            raise QgsProcessingException(f"Ridge lines file not found: {ridge_lines_path}")
        if not facc_raster_path or not os.path.exists(facc_raster_path):
            raise QgsProcessingException(f"Flow accumulation raster not found: {facc_raster_path}")
        
        # Use parameterAsOutputLayer to preserve checkbox state information
        main_ridges_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_MAIN_RIDGES, context)
        
        # Get algorithm parameters
        nr_main = self.parameterAsInt(parameters, self.NR_MAIN, context)
        clip_to_perimeter = self.parameterAsBool(parameters, self.CLIP_TO_PERIMETER, context)

        # Extract actual file path from layer object for processing
        main_ridges_file_path = main_ridges_output_layer if isinstance(main_ridges_output_layer, str) else main_ridges_output_layer

        feedback.pushInfo("Reading CRS from ridge lines...")
        # Read CRS from the ridge lines layer
        ridge_crs = get_crs_from_layer(ridge_lines_layer)
        feedback.pushInfo(f"Ridge lines CRS: {ridge_crs}")

        # Check if self.core.crs matches ridge_crs, warn and update if not
        if ridge_crs:
            if self.core and hasattr(self.core, "crs"):
                if self.core.crs != ridge_crs:
                    feedback.reportError(f"Warning: TopoDrainCore CRS ({self.core.crs}) does not match ridge lines CRS ({ridge_crs}). Updating TopoDrainCore CRS to match ridge lines.")
                    self.core.crs = ridge_crs

        feedback.pushInfo("Processing extract_main_ridges via TopoDrainCore...")
        if not self.core:
            from topo_drain.core.topo_drain_core import TopoDrainCore
            feedback.reportError("TopoDrainCore not set, creating default instance.")
            self.core = TopoDrainCore()  # fallback: create default instance (not recommended for plugin use)

        # Load input data as GeoDataFrame
        feedback.pushInfo("Loading ridge lines...")
        ridge_lines_gdf = gpd.read_file(ridge_lines_path)

        if ridge_lines_gdf.empty:
            raise QgsProcessingException("No features found in ridge lines input")

        # Load perimeter if provided, otherwise will be None (and core function will handle it)
        perimeter_gdf = None
        if perimeter_layer:
            feedback.pushInfo("Loading perimeter...")
            perimeter_gdf = gpd.GeoDataFrame.from_features(perimeter_layer.getFeatures())
        else:
            feedback.pushInfo("No perimeter provided, will use ridge lines extent")

        if perimeter_gdf is not None and perimeter_gdf.empty:
            feedback.reportError("No features found in perimeter input")

        # Check for required attributes (case-insensitive)
        feedback.pushInfo(f"Checking ridge lines attributes: {list(ridge_lines_gdf.columns)}")
        required_attrs = ['FID', 'TRIB_ID', 'DS_LINK_ID']
        # Convert column names to uppercase for case-insensitive comparison
        available_attrs_upper = [col.upper() for col in ridge_lines_gdf.columns]
        feedback.pushInfo(f"Ridge lines attributes (uppercase): {available_attrs_upper}")
        missing_attrs = [attr for attr in required_attrs if attr not in available_attrs_upper]
        if missing_attrs:
            raise QgsProcessingException(f"Ridge lines missing required attributes: {missing_attrs}. Please use output from Create Ridges algorithm. Available attributes: {list(ridge_lines_gdf.columns)}")

        # Call the core function
        feedback.pushInfo("Running extract main ridges...")
        main_ridges_gdf = self.core.extract_main_ridges(
            ridge_lines=ridge_lines_gdf,
            facc_path=facc_raster_path,
            perimeter=perimeter_gdf,
            nr_main=nr_main,
            clip_to_perimeter=clip_to_perimeter,
            feedback=feedback
        )

        if main_ridges_gdf.empty:
            raise QgsProcessingException("No main ridges were detected")

        feedback.pushInfo(f"Created {len(main_ridges_gdf)} main ridges")

        # Ensure the main ridges GeoDataFrame has the correct CRS
        main_ridges_gdf = main_ridges_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Main ridge lines CRS: {main_ridges_gdf.crs}")

        # Save result
        try:
            main_ridges_gdf.to_file(main_ridges_file_path)
            feedback.pushInfo(f"Main ridge lines saved to: {main_ridges_file_path}")
        except Exception as e:
            raise QgsProcessingException(f"Failed to save main ridges output: {e}")
        
        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters:
                results[outputName] = parameters[outputName]

        return results
