# Keyline Design Manual

This manual guides you through implementing a Keyline Design using the TopoDrain plugin tools.

## Table of Contents
- [DTM Preprocessing](#dtm-preprocessing)
- [Terrain Visualization: Contours and Hillshade](#terrain-visualization-contours-and-hillshade)
- [Creating Valleys and Ridges](#creating-valleys-and-ridges)
- [Create Perimeter](#create-perimeter)
- [Extract Main Valleys and Ridges](#extract-main-valleys-and-ridges)

## DTM Preprocessing

Before using the TopoDrain tools, it's important to prepare your Digital Terrain Model (DTM) properly.

### Obtaining High-Resolution DTM Data

Get a high-resolution Digital Terrain Model (DTM) raster dataset for your study area.

**For users in Switzerland:**
- Use the QGIS plugin **"Swiss Geo Downloader"** to download the **swissALTI3D** dataset
- ⚠️ **Important:** Make sure to choose the best resolution (0.5 m) before downloading

### Merging Multiple DTM Files

If you have downloaded multiple DTM tiles, you need to merge them into a single file:

1. Use the GDAL processing tool **"Merge"** to combine all tiles into one DTM file (.tif)
   - *Alternative:* You can also use the **"r.patch"** tool

> **Tip:** If your DTM has a large extent, you may want to clip it to your study area to make subsequent processing faster. Use the GDAL processing tool **"Clip raster by extent"** or **"Clip raster by mask layer"**.

### Optional: Burning Streams at Roads (for Watershed Delineation)

If you plan to delineate water basins later (e.g., using the TopoDrain tool **"Delineate Watershed"**), you may want to apply preprocessing to ensure meaningful delineation at road crossings.

**Use the WhiteboxTools processing tool "BurnStreamsAtRoads":**

This tool burns (lowers) the DTM elevation along streams at roads (bridges), which ensures that watershed delineation respects the actual hydrological connectivity.

**Requirements:**
- Official river/stream vector layer
- Road network vector layer

> **Alternative:** If you don't have a road network layer, you can use the WhiteboxTools tool **"FillBurn"** which only requires a stream vector layer.

**More information:** https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html?highlight=BurnStreamsAtRoads#burnstreamsatroads

---

## Terrain Visualization: Contours and Hillshade

While not required for the processing steps, creating contour lines and a hillshade visualization helps you better understand the terrain characteristics. The hillshade in particular may reveal terrain features, which are interesting to consider for planning keylines.

### Creating Contour Lines

1. Use the WhiteboxTools processing tool **"ContoursFromRaster"**
2. Choose an appropriate contour interval (e.g., 1 m)
3. Run the tool on your DTM

<img src="../resources/ContoursFromRaster.png" alt="Creating Contours from Raster" width="600">

### Creating Hillshade

1. Use the WhiteboxTools processing tool **"Hillshade"**
2. Run the tool on your DTM
3. Adjust the symbology of the contour lines layer to display thin grey lines for better visualization

<img src="../resources/Hillshade.png" alt="Hillshade" width="500">

---

## Creating Valleys and Ridges

### Create Valleys (Stream Network)

The **Create Valleys (stream network)** tool is the starting point for analyzing drainage patterns in your terrain. This tool processes a Digital Terrain Model (DTM) to generate valley lines representing the natural drainage network.

<img src="../resources/CreateValleys.png" alt="Create Valleys Dialog" width="500">

The tool uses a series of WhiteboxTools processes:
1. **D8Pointer** - Calculates flow direction for each cell
2. **D8FlowAccumulation** - Determines accumulated flow (drainage area) for each cell  
3. **ExtractStreams** - Identifies stream network based on flow accumulation threshold
4. **RasterStreamsToVector** - Converts raster streams to vector polylines
5. **StreamLinkIdentifier** - Assigns unique IDs to individual stream segments
6. **VectorStreamNetworkAnalysis** - Analyzes network topology and calculates stream properties

#### Parameters

- **Input DTM (Digital Terrain Model)**: Select your preprocessed Digital Terrain Model

- **Advanced: Maximum search distance for breach paths in cells**: Maximum distance to search when breaching depressions
  - Default: 0 cells
  - Usually the default value works well
  - See WhiteboxTools `BreachDepressionsLeastCost` for details

- **Accumulation Threshold**: Controls stream network density
  - Lower values = denser network with more small tributaries
  - Higher values = simplified network with only major valleys
  - Default: 1000 cells
  - See WhiteboxTools `ExtractStreams` for details
  - **Tip**: Prefer smaller values to capture more detail, as you can extract the main valley lines in a later step

#### Outputs

The tool generates several layers:

1. **Output Valley Lines (polylines)** - The primary output showing valley lines
   - **Main result** layer for further analysis
   - Recommend styling with thin blue lines for visualization
   
2. **Output Filled DTM (raster)** - Depression-breached terrain model
   - Not needed in further Keyline-Design process
   
3. **Output Flow Direction Raster** - D8 flow direction grid
   - Only needed in further processing if you optionally you want to delinate watershed with "Delinate Watersheds" later
   
4. **Output Flow Accumulation Raster** - Shows accumulated flow for each cell
   - **Needed** in further Keyline-Design process (Extract Main Valleys)
   - Higher values indicate larger drainage areas
   
5. **Output Log Accumulation Raster** - Logarithmic scale of flow accumulation
   - Better for visualization than raw accumulation values
   - Makes small and large values easier to distinguish

6. **Output Stream Raster** - Rasterized stream network
   - Intermediate output before vectorization
   - Not needed in further Keyline-Design process

<img src="../resources/CreateValleys.png" alt="Create Valleys Result" width="500">

---

### Create Ridges (Inverted Stream Network)

The **Create Ridges (inverted stream network)** tool works similarly to the Create Valleys tool, but analyzes ridge lines instead of valley lines. Simply provide your normal DTM as input - the tool automatically inverts the elevation values internally before processing.

It uses the same WhiteboxTools processes (D8Pointer, D8FlowAccumulation, ExtractStreams, etc.) on the inverted terrain model, where ridges become valleys in the inverted space.

The parameters and outputs are identical to the Create Valleys tool, so refer to that section for detailed explanations.

**Styling recommendation:** Display ridge lines in orange/brown to distinguish them from valley lines.

<img src="../resources/CreateRidges.png" alt="Create Ridges Result" width="500">

---

## Create Perimeter

Before extracting main valleys and ridges, you need to define the boundary (perimeter) of your study area. This perimeter will be used to clip and focus the analysis on your area of interest.

### Creating a Perimeter Polygon

1. In the **Browser panel**, navigate to **Project Home** → folder where you want to create file i.e. **Vectors**
2. Right-click and select **New** → **ShapeFile**
3. Configure the new shapefile:
   - **File name**: Choose a descriptive name (e.g., `Perimeter`)
   - **Geometry type**: Select **Polygon**
   - **Coordinate Sysetme (CRS)**: Set to match your map/DTM CRS
   - **Optional**: Add a text field named "Name" to store the study area name
4. Click **OK** to create the shapefile
5. Add the new layer to your map if not automatically added
6. Toggle editing mode (click the pencil icon) and use the **Add Polygon Feature** tool to digitize your perimeter polygon around your study area (e.g., around the agricultural field on the hillslope)
7. Save edits when finished

**Tip**: Consider valley and ridge lines when drawing your polygon. You may want to include important branches (valleys or ridges) that extend beyond the exact boundary of the agricultural field, as these features can be relevant for the overall Keyline Design. Conversely, you may want to exclude main branches that intersect the perimeter edge but are not relevant for your study area analysis. You can also iterate between the Extract Main Valleys and Extract Main Ridges tools to refine the alternation of valley, ridge, and perimeter lines — the perimeter will act as a "final" ridge or valley.

**Styling recommendation**: Set the polygon to have a black outline (no fill or transparent fill).

<img src="../resources/EditPerimeter.png" alt="Editing Perimeter Polygon" width="500">


> **Note**: If you run a tool multiple times and overwrite an existing layer, you may need to refresh the layer visibility in QGIS. Right-click on the layer → **Change Data Source** → select the data file again to refresh the layer.


---

## Extract Main Valleys and Ridges

After creating the complete valley and ridge networks, you need to identify and extract the main (most significant) valleys and ridges for your study area.

### Extract Main Valleys

The **Extract Main Valleys** tool identifies the most significant valley lines (flow paths) from the complete valley network by selecting the tributaries with the highest flow accumulation values within your perimeter. 

<img src="../resources/ExtractMainValleys.png" alt="Extract Main Valleys Dialog" width="500">

#### How it works

The algorithm:
- Clips valley lines to the specified perimeter (analyzes each polygon feature separately if multiple polygons exist)
- Extracts flow accumulation values at valley line locations
- Identifies the point with highest flow accumulation for each tributary (defined by attribute `TRIB_ID`)
- Selects the top N tributaries by maximum flow accumulation
- Merges line segments belonging to each selected tributary (using attribute `DS_LINK_ID`)
- Adds a `RANK` attribute (1 = highest flow accumulation, 2 = second highest, etc.)

#### Parameters

- **Input Valley Lines**: Select the **Output Valley Lines** from the "Create Valleys" tool
  - ⚠️ **Important**: Must have `LINK_ID`, `TRIB_ID`, and `DS_LINK_ID` attributes (automatically created by Create Valleys)

- **Input Flow Accumulation Raster**: Select the **Output Flow Accumulation Raster** from the "Create Valleys" tool
  - ⚠️ **Important**: Use the same flow accumulation raster that was used to create the valley lines

- **Input Perimeter Polygon**: Select your perimeter polygon layer
  - If multiple polygons exist, analysis is performed separately for each polygon
  - Optional: If not provided, uses the full extent of valley lines

- **Number of main valleys to extract**: Specify how many main valleys to extract
  - Default: 2
  - **Tip**: Try to estimate the number of main branches from the Valleys layer. It's better to choose a higher value - you can always delete unnecessary lines later in edit mode
  - **Example**: If you see approximately 4 main valleys, choose 5 for this parameter to ensure you capture all important features

- **Clip output to perimeter**: Whether to clip the output lines to the perimeter boundary
  - Default: True
  - Keep enabled to focus on your study area

#### Output

**Output Main Valleys**: Line layer containing the main valley lines with attributes:
- `LINK_ID` - Standard cross-platform identifier for each line segment
- `TRIB_ID` - Tributary identifier
- `RANK` - Valley order (1 = first main valley with highest flow accumulation, 2 = second highest, etc.)
- `POLYGON_ID` - Identifier if multiple perimeter polygons were used
- `DS_LINK_ID` - Downstream link identifier

#### Styling Recommendation

Style the Main Valleys layer with a **blue line that is thicker** than the Valleys layer, and add a marker line showing flow direction:

1. **Line style**: Thick blue line
2. **Add marker line** (for flow direction):
   - **Symbol layer type**: Marker line
   - **Interval**: e.g., 10 m
   - **Marker**: Triangle (blue)
   - **Size**: 2
   - **Rotation**: **90°** (to align with flow direction)

<img src="../resources/MainValleysStyle.png" alt="Main Valleys Styling" width="500">

#### Label Recommendation

Add labels showing the **RANK** attribute to identify the valley order:

1. Enable labels for the Main Valleys layer
2. **Label with**: `RANK`
3. This displays: 1 = first main valley (highest flow accumulation), 2 = second highest, etc.

<img src="../resources/MainValleysLabel.png" alt="Main Valleys Labels" width="500">


---

### Extract Main Ridges

The **Extract Main Ridges** tool works identically to Extract Main Valleys, but operates on ridge lines instead of valley lines.

Use the same approach:
- **Input Ridge Lines**: Select the **Output Ridge Lines** from the "Create Ridges" tool
- **Input Flow Accumulation Raster**: Select the **Output Inverted Flow Accumulation Raster** from the "Create Ridges" tool (inverted flow accumulation)
- Configure the same parameters as described in the Extract Main Valleys section

**Styling recommendation**: Use orange/brown thick lines to distinguish from main valleys.

<img src="../resources/ExtractMainRidges.png" alt="Extract Main Ridges Result" width="500">

---

### Adjust Main Valleys and Main Ridges

**Goal**: Create a clean valley-ridge pattern suitable for keyline design. Topography is often complex, resulting in intricate main valley and main ridge patterns. Some creativity is needed here to create a useful pattern that works for your design.

> **Note**: Depending on your terrain, you may even want to create Main Valleys and Main Ridges from scratch by manual digitizing rather than extracting them. However, this example shows how to adjust extracted main valleys and main ridges.

<img src="../resources/AdjustMainValleysMainRidges.png" alt="Areas to Adjust (marked in red)" width="500">

#### Step 1: Smooth the Lines

Before editing the Main Ridges and Main Valleys in QGIS, it's recommended to use the WhiteboxTools processing tool **"SmoothVectors"**:

- This reduces the number of vertices, making editing easier
- It also improves accuracy for subsequent tools like "Get Points Along Line" (distance calculations are more accurate on smoothed lines than on very wavy lines)

<img src="../resources/SmoothMainValleysMainRidges.png" alt="Smooth Vectors Tool" width="500">

#### Step 2: Edit the Pattern

Edit the Main Valleys and Main Ridges layers to create a clean pattern:

1. Toggle editing mode for the Main Valleys layer
2. Use vertex editing tools to adjust line geometry
3. Delete unnecessary segments or features
4. Repeat for Main Ridges layer

**Pattern recommendation**: While the "Create Keylines" tool can handle valley-to-valley or ridge-to-ridge tracing, it's recommended to create a pattern with **alternating** features:

**Perimeter → Valley → Ridge → Valley → ... → Perimeter**

This alternation creates a more organized and practical keyline pattern.

<img src="../resources/EditMainValley_step1.png" alt="Edit Main Valley - Step 1" width="500">

<img src="../resources/EditMainValley_step2.png" alt="Edit Main Valley - Step 2" width="500">

<img src="../resources/EditMainValley_step3.png" alt="Edit Main Valley - Step 3" width="500">

#### Final Pattern

After editing, you should have a clean, alternating valley-ridge pattern ready for keyline creation:

<img src="../resources/MainValleysMainRidges_processed.png" alt="Final Processed Pattern" width="500">

---


---