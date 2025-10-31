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
from .utils import get_crs_from_layer, update_core_crs_if_needed, ensure_whiteboxtools_configured, save_gdf_to_file, load_gdf_from_file, load_gdf_from_qgis_source, get_raster_ext, get_vector_ext

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
        instance = ExtractMainRidgesAlgorithm(core=self.core)
        if hasattr(self, 'plugin'):
            instance.plugin = self.plugin
        return instance

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
- Adds RANK attribute (1=highest flow accumulation, 2=second highest, etc.)

This is useful for identifying the most significant watershed divides (ridges) in a watershed or study area, focusing analysis on the primary ridges resp. drainage divides.

Input Requirements:
- Ridge Lines: Must have 'LINK_ID', 'TRIB_ID', and 'DS_LINK_ID' attributes (from Create Ridges algorithm). LINK_ID is the standard cross-platform identifier.
- Flow Accumulation Raster: Raster showing accumulated flow (e.g., from Create Ridges algorithm, based on inverted DTM)
- Perimeter (optional): Polygon defining the study area boundary. If not provided, uses the extent of ridge lines

OUTPUT_MAIN_RIDGES:
Line layer containing main ridge lines with attributes: LINK_ID, TRIB_ID, RANK, POLYGON_ID, DS_LINK_ID"""
        )
    
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'icons', 'topo_drain.svg'))

    def initAlgorithm(self, config=None):        
        # Input parameters
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_RIDGE_LINES,
                self.tr('Input Ridge Lines (must have LINK_ID, TRIB_ID, DS_LINK_ID attributes)'),
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
        # Ensure WhiteboxTools is configured before running
        if not ensure_whiteboxtools_configured(self, feedback):
            return {}
        
        # Validate and read input parameters
        ridge_lines_layer = self.parameterAsVectorLayer(parameters, self.INPUT_RIDGE_LINES, context)
        facc_raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_FACC_RASTER, context)
        perimeter_layer = self.parameterAsSource(parameters, self.INPUT_PERIMETER, context)

        # Get file paths and validate formats
        ridge_lines_path = ridge_lines_layer.source()
        ridge_ext = get_vector_ext(ridge_lines_path, feedback)
        
        # Get FACC raster path and validate format
        facc_raster_path = facc_raster_layer.source()
        facc_ext = get_raster_ext(facc_raster_path, feedback)
        
        # Create supported formats lists
        supported_vector_formats = list(self.core.ogr_driver_mapping.keys()) if hasattr(self.core, 'ogr_driver_mapping') else []
        supported_raster_formats = list(self.core.gdal_driver_mapping.keys())
        
        # Validate vector format compatibility with OGR driver mapping (warning only)
        if hasattr(self.core, 'ogr_driver_mapping') and ridge_ext not in self.core.ogr_driver_mapping:
            feedback.pushWarning(f"Ridge lines format '{ridge_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to load it automatically.")
        
        # Validate raster format compatibility with GDAL driver mapping
        if hasattr(self.core, 'gdal_driver_mapping') and facc_ext not in self.core.gdal_driver_mapping:
            raise QgsProcessingException(f"Flow accumulation raster format '{facc_ext}' is not supported. Supported formats: {supported_raster_formats}")
        
        # Use parameterAsOutputLayer to preserve checkbox state information
        main_ridges_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_MAIN_RIDGES, context)
        
        # Get algorithm parameters
        nr_main = self.parameterAsInt(parameters, self.NR_MAIN, context)
        clip_to_perimeter = self.parameterAsBool(parameters, self.CLIP_TO_PERIMETER, context)

        # Extract actual file path from layer object for processing
        main_ridges_file_path = main_ridges_output_layer
        
        # Validate output vector format compatibility with OGR driver mapping
        output_ext = get_vector_ext(main_ridges_file_path, feedback, check_existence=False)
        if hasattr(self.core, 'ogr_driver_mapping') and output_ext not in self.core.ogr_driver_mapping:
            feedback.pushWarning(f"Output file format '{output_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to save it automatically.")

        feedback.pushInfo("Reading CRS from ridge lines...")
        # Read CRS from the ridge lines layer with safe fallback
        ridge_crs = get_crs_from_layer(ridge_lines_layer)
        feedback.pushInfo(f"Ridge lines CRS: {ridge_crs}")
        # Update core CRS if needed (ridge_crs is guaranteed to be valid)
        update_core_crs_if_needed(self.core, ridge_crs, feedback)

        # Load input data as GeoDataFrame with Windows-safe CRS handling
        feedback.pushInfo("Loading ridge lines...")
        try:
            ridge_lines_gdf = load_gdf_from_file(ridge_lines_path, feedback)
            ridge_lines_gdf.crs = self.core.crs
            feedback.pushInfo(f"Successfully loaded {len(ridge_lines_gdf)} ridge line features with safe CRS: {self.core.crs}")
        except Exception as e:
            feedback.pushInfo(f"Failed to load ridge lines with safe CRS handling: {e}")
            raise QgsProcessingException(f"Failed to load ridge lines: {e}")

        if ridge_lines_gdf.empty:
            raise QgsProcessingException("No features found in ridge lines input")

        # Load perimeter if provided, otherwise will be None (and core function will handle it)
        perimeter_gdf = None
        if perimeter_layer:
            feedback.pushInfo("Loading perimeter...")
            try:
                # Load perimeter features with automatic data cleaning
                perimeter_gdf = load_gdf_from_qgis_source(perimeter_layer, feedback)
                if not perimeter_gdf.empty:
                    perimeter_gdf.crs = self.core.crs
                    feedback.pushInfo(f"Successfully loaded {len(perimeter_gdf)} perimeter features with safe CRS")
            except Exception as e:
                feedback.pushInfo(f"Failed to load perimeter with safe CRS handling: {e}")
                raise QgsProcessingException(f"Failed to load perimeter: {e}")
        else:
            feedback.pushInfo("No perimeter provided, will use ridge lines extent")

        if perimeter_gdf is not None and perimeter_gdf.empty:
            feedback.reportError("No features found in perimeter input")

        # Check for required attributes (case-insensitive)
        feedback.pushInfo(f"Checking ridge lines attributes: {list(ridge_lines_gdf.columns)}")
        required_attrs = ['LINK_ID', 'TRIB_ID', 'DS_LINK_ID']
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

        # Save result with proper format handling
        save_gdf_to_file(main_ridges_gdf, main_ridges_file_path, self.core, feedback)
        
        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters:
                results[outputName] = parameters[outputName]

        return results
