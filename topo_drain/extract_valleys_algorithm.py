# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: extract_valleys_algorithm.py
#
# Purpose: QGIS Processing Algorithm to create valley lines (river network) based on WhiteboxTools
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing, QgsRasterLayer, QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFileDestination, QgsProcessingParameterNumber,
                       QgsVectorLayer, QgsProject, QgsProcessingParameterBoolean,
                       QgsSymbol, QgsLineSymbol, QgsMarkerLineSymbolLayer, 
                       QgsSimpleMarkerSymbolLayer, QgsSingleSymbolRenderer, QgsMarkerSymbol)
from qgis.PyQt.QtGui import QColor

import os
import rasterio
import geopandas as gpd


class ExtractValleysAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for extracting valley lines (stream network) from a DEM resp. DTM.

    This algorithm leverages several WhiteboxTools (WBT) processes:
    - BreachDepressionsLeastCost: Optimally breaches depressions in the DEM to prepare it for hydrological analysis, providing a lower-impact alternative to depression filling.
    - D8Pointer: Generates a flow direction raster using the D8 algorithm, assigning flow from each cell to its steepest downslope neighbor.
    - D8FlowAccumulation: Calculates flow accumulation (contributing area) using the FD8 algorithm, distributing flow among downslope neighbors.
    - ExtractStreams: Extracts stream networks from the flow accumulation raster based on a user-defined threshold, identifying significant flow paths.
    - RasterStreamsToVector: Converts rasterized stream networks into vector line features for further analysis or export.
    - StreamLinkIdentifier: Assigns unique identifiers to each stream segment (link) in the raster stream network.
    - VectorStreamNetworkAnalysis: Analyzes the vectorized stream network, calculating stream order (e.g., HORTON), tributary IDs (TRIB_ID), and additional attributes such as the downstream link ID (DS_LINK_ID) for each stream segment (FID).

    For more customization, you can use individual WhiteboxTools algorithms directly in the QGIS Processing Toolbox (WhiteBox Plugin) step by step.
    """

    INPUT_DTM = 'INPUT_DTM'
    OUTPUT_VALLEYS = 'OUTPUT_VALLEYS'
    OUTPUT_FILLED = 'OUTPUT_FILLED'
    OUTPUT_FDIR = 'OUTPUT_FDIR'
    OUTPUT_FACC = 'OUTPUT_FACC'
    OUTPUT_FACC_LOG = 'OUTPUT_FACC_LOG'
    OUTPUT_STREAMS = 'OUTPUT_STREAMS'
    ACCUM_THRESHOLD = 'ACCUM_THRESHOLD'
    DIST_FACC = 'DIST_FACC'

    ADD_VALLEYS_TO_MAP = 'ADD_VALLEYS_TO_MAP'
    ADD_FILLED_TO_MAP = 'ADD_FILLED_TO_MAP'
    ADD_FDIR_TO_MAP = 'ADD_FDIR_TO_MAP'
    ADD_FACC_TO_MAP = 'ADD_FACC_TO_MAP'
    ADD_FACC_LOG_TO_MAP = 'ADD_FACC_LOG_TO_MAP'
    ADD_STREAMS_TO_MAP = 'ADD_STREAMS_TO_MAP'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core
    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ExtractValleysAlgorithm(core=self.core)

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
            """QGIS Processing Algorithm for extracting valley lines (stream network) from a DEM resp. DTM
                This algorithm leverages several WhiteboxTools (WBT) processes:
                - BreachDepressionsLeastCost: Optimally breaches depressions in the DEM to prepare it for hydrological analysis, providing a lower-impact alternative to depression filling.
                - D8Pointer: Generates a flow direction raster using the D8 algorithm, assigning flow from each cell to its steepest downslope neighbor.
                - D8FlowAccumulation: Calculates flow accumulation (contributing area) using the FD8 algorithm, distributing flow among downslope neighbors.
                - ExtractStreams: Extracts stream networks from the flow accumulation raster based on a user-defined threshold, identifying significant flow paths.
                - RasterStreamsToVector: Converts rasterized stream networks into vector line features for further analysis or export.
                - StreamLinkIdentifier: Assigns unique identifiers to each stream segment (link) in the raster stream network.
                - VectorStreamNetworkAnalysis: Analyzes the vectorized stream network, calculating stream order (e.g., HORTON), tributary IDs (TRIB_ID), and additional attributes such as the downstream link ID (DS_LINK_ID) for each stream segment (FID).
                For more customization, you can use individual WhiteboxTools algorithms directly in the QGIS Processing Toolbox (WhiteBox Plugin) step by step."""
        )

    def initAlgorithm(self, config=None):        
        # Input parameters
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DTM,
                self.tr('Input DTM (GeoTIFF)')
            )
        )
        # Algorithm parameters
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
        # Output parameters with "Add to Map" options
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_VALLEYS,
                self.tr('Output Valley Lines'),
                fileFilter='Shapefile (*.shp);;GeoPackage (*.gpkg)'
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ADD_VALLEYS_TO_MAP,
                self.tr('Add Valley Lines to QGIS map'),
                defaultValue=True
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
            QgsProcessingParameterBoolean(
                self.ADD_FILLED_TO_MAP,
                self.tr('Add Filled DTM to QGIS map'),
                defaultValue=False
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
            QgsProcessingParameterBoolean(
                self.ADD_FDIR_TO_MAP,
                self.tr('Add Flow Direction to QGIS map'),
                defaultValue=False
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
            QgsProcessingParameterBoolean(
                self.ADD_FACC_TO_MAP,
                self.tr('Add Flow Accumulation to QGIS map'),
                defaultValue=False
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
            QgsProcessingParameterBoolean(
                self.ADD_FACC_LOG_TO_MAP,
                self.tr('Add Log-Scaled Accumulation to QGIS map'),
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_STREAMS,
                self.tr('Output Stream Raster'),
                fileFilter='GeoTIFF (*.tif)',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ADD_STREAMS_TO_MAP,
                self.tr('Add Stream Raster to QGIS map'),
                defaultValue=False
            )
        )

    def _apply_valley_symbology(self, vlayer, feedback=None):
        """Apply blue line symbology with flow direction markers to valley lines layer."""
        try:
            # Create a line symbol with blue color
            line_symbol = QgsLineSymbol.createSimple({
                'color': '#0066CC',  # Blue color
                'width': '0.4',
                'capstyle': 'round',
                'joinstyle': 'round'
            })
            
            # Create marker line symbol layer for flow direction arrows
            marker_line = QgsMarkerLineSymbolLayer()
            marker_line.setPlacement(QgsMarkerLineSymbolLayer.Interval)
            marker_line.setInterval(20)  # Place markers every 20 map units
            marker_line.setRotateMarker(True)  # Rotate markers along line direction
            
            # Create marker symbol for arrows
            marker_symbol = QgsMarkerSymbol()
            marker_symbol.deleteSymbolLayer(0)  # Remove default layer
            
            # Create arrow marker layer
            arrow_marker = QgsSimpleMarkerSymbolLayer()
            arrow_marker.setShape(QgsSimpleMarkerSymbolLayer.ArrowHead)
            arrow_marker.setSize(4)  # Arrow size
            arrow_marker.setColor(QColor('#003366'))  # Dark blue for arrows
            arrow_marker.setStrokeColor(QColor('#0066CC'))  # Even darker outline
            arrow_marker.setStrokeWidth(0.2)
            arrow_marker.setAngle(0)  # Don't add extra rotation, let the marker line handle it
            
            # Add arrow to marker symbol
            marker_symbol.appendSymbolLayer(arrow_marker)
            
            # Set the marker symbol to the marker line
            marker_line.setSubSymbol(marker_symbol)
            
            # Add marker line to the main line symbol
            line_symbol.appendSymbolLayer(marker_line)
            
            # Apply the symbol to the layer
            renderer = QgsSingleSymbolRenderer(line_symbol)
            vlayer.setRenderer(renderer)
            vlayer.triggerRepaint()
            
        except Exception as e:
            # If symbology fails, just continue without it - for debugging, let's see the error
            if feedback:
                feedback.reportError(f"Failed to apply symbology: {str(e)}")
            pass

    def processAlgorithm(self, parameters, context, feedback):
        # Validate and read input parameters
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        dtm_path = dtm_layer.source()
        if not dtm_path or not os.path.exists(dtm_path):
            raise FileNotFoundError(f"[Input Error] DTM file not found: {dtm_path}")
        valley_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_VALLEYS, context)
        filled_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FILLED, context)
        fdir_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FDIR, context)
        facc_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FACC, context)
        facc_log_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FACC_LOG, context)
        streams_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_STREAMS, context)
        accumulation_threshold = self.parameterAsInt(parameters, self.ACCUM_THRESHOLD, context)
        dist_facc = self.parameterAsDouble(parameters, self.DIST_FACC, context)
        # Read boolean parameters for adding layers to the map
        add_valleys = self.parameterAsBool(parameters, self.ADD_VALLEYS_TO_MAP, context)
        add_filled = self.parameterAsBool(parameters, self.ADD_FILLED_TO_MAP, context)
        add_fdir = self.parameterAsBool(parameters, self.ADD_FDIR_TO_MAP, context)
        add_facc = self.parameterAsBool(parameters, self.ADD_FACC_TO_MAP, context)
        add_facc_log = self.parameterAsBool(parameters, self.ADD_FACC_LOG_TO_MAP, context)
        add_streams = self.parameterAsBool(parameters, self.ADD_STREAMS_TO_MAP, context)

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
        if streams_output_path:
            feedback.pushInfo(f"Stream Raster Output: {streams_output_path}")
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
        if streams_output_path and not os.path.isdir(os.path.dirname(streams_output_path)):
            raise FileNotFoundError(f"[Input Error] directory not found: {os.path.dirname(streams_output_path)}")
        
        feedback.pushInfo("Reading CRS from DTM...")
        # Read CRS from the DTM
        dtm_crs = None
        with rasterio.open(dtm_path) as src:
            dtm_crs = src.crs # Read CRS from the DTM
            feedback.pushInfo(f"crs: {dtm_crs}")

        feedback.pushInfo("Processing extract_valleys via TopoDrainCore...")
        if not self.core:
            from topo_drain.core.topo_drain_core import TopoDrainCore
            print("TopoDrainCore not set, creating default instance.")
            self.core = TopoDrainCore()  # fallback: create default instance (not recommended for plugin use)

        gdf_valleys = self.core.extract_valleys(
            dtm_path=dtm_path,
            filled_output_path=filled_output_path,
            fdir_output_path=fdir_output_path,
            facc_output_path=facc_output_path,
            facc_log_output_path=facc_log_output_path,
            streams_output_path=streams_output_path,
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

        # Optionally add each output to QGIS project
        # Add valley lines
        if add_valleys:
            try:
                feedback.pushInfo("Add Valley lines layer to QGIS Map...")  
                vlayer = QgsVectorLayer(valley_output_path, "Valley Lines", "ogr")
                if not vlayer.isValid():
                    feedback.reportError(f"Failed to load valley lines layer: {valley_output_path}")
                else:
                    # Apply blue line symbology with flow direction markers
                    self._apply_valley_symbology(vlayer, feedback)
                    QgsProject.instance().addMapLayer(vlayer)
                    feedback.pushInfo("Valley lines layer added to QGIS project with flow direction symbology.")
            except Exception as e:
                feedback.reportError(f"Could not add valley lines to QGIS project: {e}")
        # Add filled DTM
        if add_filled and filled_output_path and os.path.exists(filled_output_path):
            try:
                rlayer = QgsRasterLayer(filled_output_path, "Filled DTM")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    feedback.pushInfo("Filled DTM layer added to QGIS project.")
                else:
                    feedback.reportError(f"Failed to load filled DTM: {filled_output_path}")
            except Exception as e:
                feedback.reportError(f"Could not add filled DTM to QGIS project: {e}")
        # Add flow direction
        if add_fdir and fdir_output_path and os.path.exists(fdir_output_path):
            try:
                rlayer = QgsRasterLayer(fdir_output_path, "Flow Direction")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    feedback.pushInfo("Flow Direction layer added to QGIS project.")
                else:
                    feedback.reportError(f"Failed to load flow direction: {fdir_output_path}")
            except Exception as e:
                feedback.reportError(f"Could not add flow direction to QGIS project: {e}")
        # Add flow accumulation
        if add_facc and facc_output_path and os.path.exists(facc_output_path):
            try:
                rlayer = QgsRasterLayer(facc_output_path, "Flow Accumulation")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    feedback.pushInfo("Flow Accumulation layer added to QGIS project.")
                else:
                    feedback.reportError(f"Failed to load flow accumulation: {facc_output_path}")
            except Exception as e:
                feedback.reportError(f"Could not add flow accumulation to QGIS project: {e}")
        # Add log-scaled accumulation
        if add_facc_log and facc_log_output_path and os.path.exists(facc_log_output_path):
            try:
                rlayer = QgsRasterLayer(facc_log_output_path, "Log-Scaled Accumulation")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    feedback.pushInfo("Log-Scaled Accumulation layer added to QGIS project.")
                else:
                    feedback.reportError(f"Failed to load log-scaled accumulation: {facc_log_output_path}")
            except Exception as e:
                feedback.reportError(f"Could not add log-scaled accumulation to QGIS project: {e}")
        # Add streams
        if add_streams and streams_output_path and os.path.exists(streams_output_path):
            try:
                rlayer = QgsRasterLayer(streams_output_path, "Streams")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    feedback.pushInfo("Streams layer added to QGIS project.")
                else:
                    feedback.reportError(f"Failed to load streams: {streams_output_path}")
            except Exception as e:
                feedback.reportError(f"Could not add streams to QGIS project: {e}")

        results = {self.OUTPUT_VALLEYS: valley_output_path}
        if filled_output_path:
            results[self.OUTPUT_FILLED] = filled_output_path
        if fdir_output_path:
            results[self.OUTPUT_FDIR] = fdir_output_path
        if facc_output_path:
            results[self.OUTPUT_FACC] = facc_output_path
        if facc_log_output_path:
            results[self.OUTPUT_FACC_LOG] = facc_log_output_path
        if streams_output_path:
            results[self.OUTPUT_STREAMS] = streams_output_path

        return results