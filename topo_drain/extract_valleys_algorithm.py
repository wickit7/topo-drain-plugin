# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: extract_valleys_algorithm.py
#
# Purpose: QGIS Processing Algorithm to create valley lines (river network) based on WhiteboxTools
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFileDestination, QgsProcessingParameterNumber)
import os
import rasterio
import geopandas as gpd
from topo_drain.core.topo_drain_core import extract_valleys

class ExtractValleysAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for extracting valley lines (stream network) from a DEM resp. DTM.

    This algorithm leverages several WhiteboxTools (WBT) processes:
    - BreachDepressionsLeastCost: Optimally breaches depressions in the DEM to prepare it for hydrological analysis, providing a lower-impact alternative to depression filling.
    - D8Pointer: Generates a flow direction raster using the D8 algorithm, assigning flow from each cell to its steepest downslope neighbor.
    - fd8_flow_accumulation: Calculates flow accumulation (contributing area) using the FD8 algorithm, distributing flow among downslope neighbors.
    - ExtractStreams: Extracts stream networks from the flow accumulation raster based on a user-defined threshold, identifying significant flow paths.
    - RasterStreamsToVector: Converts rasterized stream networks into vector line features for further analysis or export.
    - StreamLinkIdentifier: Assigns unique identifiers to each stream segment (link) in the raster stream network.
    - VectorStreamNetworkAnalysis: Analyzes the vectorized stream network, calculating stream order (e.g., HORTON), tributary IDs (TRIB_ID), and additional attributes such as the downstream link ID (DS_LINK_ID) for each stream segment (FID).

    For more customization, you can use individual WhiteboxTools algorithms directly in the QGIS Processing Toolbox (WhiteBox Pulgin) step by step.
    """

    INPUT_DTM = 'INPUT_DTM'
    OUTPUT_VALLEYS = 'OUTPUT_VALLEYS'
    OUTPUT_FILLED = 'OUTPUT_FILLED'
    OUTPUT_FDIR = 'OUTPUT_FDIR'
    OUTPUT_FACC = 'OUTPUT_FACC'
    OUTPUT_FACC_LOG = 'OUTPUT_FACC_LOG'
    ACCUM_THRESHOLD = 'ACCUM_THRESHOLD'
    DIST_FACC = 'DIST_FACC'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ExtractValleysAlgorithm()

    def name(self):
        return 'extract_valleys'

    def displayName(self):
        return self.tr('Extract Valleys (stream network)')

    def group(self):
        return self.tr('Basic Hydrological Analysis')

    def groupId(self):
        return 'basic_hydrological_analysis'

    def shortHelpString(self):
        return self.tr(
            "Extracts valley lines (stream network) from a digital terrain model using WhiteboxTools." \

        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DTM,
                self.tr('Input DTM (GeoTIFF)')
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_VALLEYS,
                self.tr('Output Valley Lines'),
                fileFilter='Shapefile (*.shp);;GeoPackage (*.gpkg)'
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_FILLED,
                self.tr('Output Filled DTM (output of WBT `BreachDepressionsLeastCost`)'),
                fileFilter='GeoTIFF (*.tif)',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_FDIR,
                self.tr('Output Flow Direction Raster (output of WBT `D8Pointer`)'),
                fileFilter='GeoTIFF (*.tif)',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_FACC,
                self.tr('Output Flow Accumulation Raster (output of WBT `D8FlowAccumulation`)'),
                fileFilter='GeoTIFF (*.tif)',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_FACC_LOG,
                self.tr('Output Log-Scaled Accumulation Raster'),
                fileFilter='GeoTIFF (*.tif)',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DIST_FACC,
                self.tr('Maximum search distance for breach paths in cells (for WBT algorithm `BreachDepressionsLeastCost`)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.ACCUM_THRESHOLD,
                self.tr('Accumulation Threshold (for WBT algorithm `ExtractStreams`)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1000
            )
        )


    def processAlgorithm(self, parameters, context, feedback):
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        dtm_path = dtm_layer.source()

        valley_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_VALLEYS, context)
        filled_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FILLED, context)
        fdir_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FDIR, context)
        facc_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FACC, context)
        facc_log_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FACC_LOG, context)
        accumulation_threshold = self.parameterAsInt(parameters, self.ACCUM_THRESHOLD, context)
        dist_facc = self.parameterAsDouble(parameters, self.DIST_FACC, context)

        feedback.pushInfo("Input:")
        feedback.pushInfo(f"DTM Input: {dtm_path}")
        feedback.pushInfo(f"Valley Output: {valley_output_path}")
        if filled_output_path:
            feedback.pushInfo(f"Filled DTM Output: {filled_output_path}")
        if fdir_output_path:
            feedback.pushInfo(f"Flow Direction Output: {fdir_output_path}")
        if facc_output_path:
            feedback.pushInfo(f"Flow Accumulation Output: {facc_output_path}")
        if facc_log_output_path:
            feedback.pushInfo(f"Log-Scaled Accumulation Output: {facc_log_output_path}") 
        feedback.pushInfo(f"Accumulation Threshold: {accumulation_threshold}")
        feedback.pushInfo(f"Max Search Distance for Breach Paths: {dist_facc}")

        feedback.pushInfo("Validating input DTM...")
        if not os.path.exists(dtm_path):
            raise FileNotFoundError(f"[Input Error] DTM file not found: {dtm_path}")
        if not dtm_path.lower().endswith(('.tif', '.tiff')):
            raise ValueError(f"[Input Error] DTM must be a GeoTIFF file: {dtm_path}")
        feedback.pushInfo("Validating output paths...")
        if not os.path.isdir(os.path.dirname(valley_output_path)):
            raise FileNotFoundError(f"[Input Error] directory not found: {os.path.dirname(valley_output_path)}")
        if filled_output_path and not os.path.isdir(os.path.dirname(filled_output_path)):
            raise FileNotFoundError(f"[Input Error] directory not found: {os.path.dirname(filled_output_path)}")
        if fdir_output_path and not os.path.isdir(os.path.dirname(fdir_output_path)):
            raise FileNotFoundError(f"[Input Error] directory not found: {os.path.dirname(fdir_output_path)}")
        if facc_output_path and not os.path.isdir(os.path.dirname(facc_output_path)):
            raise FileNotFoundError(f"[Input Error] directory not found: {os.path.dirname(facc_output_path)}")
        if facc_log_output_path and not os.path.isdir(os.path.dirname(facc_log_output_path)):
            raise FileNotFoundError(f"[Input Error] directory not found: {os.path.dirname(facc_log_output_path)}")
        
        feedback.pushInfo("Reading CRS from DTM...")
        # Read CRS from the DTM
        dtm_crs = None
        with rasterio.open(dtm_path) as src:
            dtm_crs = src.crs # Read CRS from the DTM
            feedback.pushInfo(f"crs: {dtm_crs}")

        feedback.pushInfo("Processig extract_valleys...")
        # Run the core extraction
        gdf_valleys = extract_valleys(
            dtm_path=dtm_path,
            filled_output_path=filled_output_path,
            fdir_output_path=fdir_output_path,
            facc_output_path=facc_output_path,
            facc_log_output_path=facc_log_output_path,
            accumulation_threshold=float(accumulation_threshold),
            dist_facc=dist_facc,
            feedback=feedback
        )

        # Assign the CRS from the input DTM
        if dtm_crs:
            feedback.pushInfo("Setting CRS for output valley lines...")
            if gdf_valleys.crs is None:
                gdf_valleys.crs = dtm_crs
            else:
                feedback.pushInfo("Overriding existing CRS with DTM CRS...")
                # If the GeoDataFrame already has a CRS, we override it with the DTM 
                # CRS to ensure consistency
                gdf_valleys = gdf_valleys.set_crs(dtm_crs, allow_override=True)

        # Save result
        try:
            gdf_valleys.to_file(valley_output_path)
            feedback.pushInfo(f"Valley lines saved to: {valley_output_path}")
        except Exception as e:
            raise RuntimeError(f"[ExtractValleysAlgorithm] failed to save valley output: {e}")

        # Add result to QGIS project
        try:
            feedback.pushInfo("Add Valley lines layer to QGIS Map...")  
            from qgis.core import QgsVectorLayer, QgsProject
            vlayer = QgsVectorLayer(valley_output_path, "Valley Lines", "ogr")
            if not vlayer.isValid():
                feedback.reportError(f"Failed to load valley lines layer: {valley_output_path}")
            else:
                QgsProject.instance().addMapLayer(vlayer)
                feedback.pushInfo("Valley lines layer added to QGIS project.")
        except Exception as e:
            feedback.reportError(f"Could not add valley lines to QGIS project: {e}")

        results = {self.OUTPUT_VALLEYS: valley_output_path}
        if filled_output_path:
            results[self.OUTPUT_FILLED] = filled_output_path
        if fdir_output_path:
            results[self.OUTPUT_FDIR] = fdir_output_path
        if facc_output_path:
            results[self.OUTPUT_FACC] = facc_output_path
        if facc_log_output_path:
            results[self.OUTPUT_FACC_LOG] = facc_log_output_path

        return results