# TopoDrain-plugin
A QGIS plugin for planning surface drainage water management. It automates the extraction of main valleys and ridges, and supports water retention planning methods, such as Keyline Design (keypoints, keylines). The algorithms are mainly based on WhiteboxTools (Lindsay, 2017–2020)

⚠️  DISCLAIMER: Managing surface runoff is a complex process influenced by topography, soil properties, farmland management practices, and other factors. This tool supports experienced users in planning and analysis and should be applied iteratively alongside expert judgment and complementary planning tools.

## Table of Contents
- [Installation Guide](#installation-guide)
  - [Installing QGIS](#installing-qgis)
    - [Windows Installation](#windows-installation)
    - [macOS Installation](#macos-installation)
    - [Python Dependencies](#python-dependencies)
  - [Installing and Configuring WhiteboxTools for QGIS](#installing-and-configuring-whiteboxtools-for-qgis)
    - [Step 1: Download WhiteboxTools](#step-1-download-whiteboxtools)
    - [Step 2: Install the WhiteboxTools QGIS Plugin](#step-2-install-the-whiteboxtools-qgis-plugin)
    - [Step 3: Configure the WhiteboxTools Provider](#step-3-configure-the-whiteboxtools-provider)
    - [Verify Installation](#verify-installation)
  - [Installing the TopoDrain Plugin](#installing-the-topodrain-plugin)
- [Recommended QGIS Plugins](#recommended-qgis-plugins)
  - [Profile Tool](#profile-tool)
  - [For Users in Switzerland: Swiss Geo Downloader](#for-users-in-switzerland-swiss-geo-downloader)

## Installation Guide

### Installing QGIS

We recommend installing the newest **QGIS LTR (Long Term Release)** version to ensure stability and compatibility with the TopoDrain plugin.

#### Windows Installation
The best way to install QGIS on Windows is using the **OSGeo4W Network Installer**:
1. Download the OSGeo4W installer from [qgis.org](https://qgis.org/resources/installation-guide/)
2. Run the installer and follow the setup wizard
3. Select the QGIS LTR version during installation

#### macOS Installation
For macOS, install QGIS using the **DMG installer**:
1. Download the DMG file "Long Term Version for maxOS" from [qgis.org](https://qgis.org/download/)
2. Open the DMG file and drag QGIS to your Applications folder
3. Launch QGIS from your Applications

#### Python Dependencies
Installing QGIS through these official methods ensures that all required Python packages are already available for the TopoDrain plugin. 

**Required Python packages:**
- `numpy`
- `pandas`
- `geopandas`
- `shapely`
- `scipy`

These packages are included by default in newer QGIS versions (tested with QGIS 3.40.5-Bratislava and later).

**If still a package is missing in your QGIS installation:**
- **Windows (OSGeo4W installation):** 
  - Use the OSGeo4W Setup installer (`osgeo4w-setup.exe`) to install missing packages
  - Alternatively, follow the instructions at: https://landscapearchaeology.org/2018/installing-python-packages-in-qgis-3-for-windows/
  
- **macOS (Terminal-based installation):** 
  - Follow the instructions at: https://gis.stackexchange.com/questions/351280/installing-python-modules-for-qgis-3-on-mac

⚠️ **Important:** Ensure you install package versions compatible with your QGIS Python environment.


**More Information:** https://qgis.org/resources/installation-guide/

### Installing and Configuring WhiteboxTools for QGIS

The TopoDrain plugin depends on **WhiteboxTools for QGIS**. You need to install both the WhiteboxTools executable and the QGIS plugin, then configure the connection.

📺 **Watch the installation video:** https://www.youtube.com/watch?v=xJXDBsNbcTg

#### Step 1: Download WhiteboxTools
1. Download the WhiteboxTools executable from the official repository https://www.whiteboxgeo.com/download-direct/
2. Extract the files to an appropriate location on your machine - for example:
   - **Windows:** `C:\WBTools\`
   - **macOS:** `/Library/WhiteboxTools_darwin_m_series/WBT/`

#### Step 2: Install the WhiteboxTools QGIS Plugin
1. Open QGIS
2. Go to **Plugins → Manage and Install Plugins**
3. Search for "WhiteboxTools for QGIS"
4. Click **Install Plugin**

#### Step 3: Configure the WhiteboxTools Provider
1. In QGIS, go to **Settings → Options → Processing → Providers**
2. Expand the **WhiteboxTools** section
3. Set the **WhiteboxTools executable** path:
   - **Windows:** `C:\WBTools\whitebox_tools.exe`
   - **macOS:** `/Library/WhiteboxTools_darwin_m_series/WBT/whitebox_tools` (without `.exe`)

<img src="resources/WBT_configure_provider.png" alt="WhiteboxTools Provider Configuration" width="500">


#### Verify Installation
To verify that WhiteboxTools is properly configured:
1. Open the **Processing Toolbox** (Processing → Toolbox)
2. Look for the **WhiteboxTools** section
3. Test for instance the processing tool **ContoursFromRaster** if it works

### Installing the TopoDrain Plugin

#### Install TopoDrain from the QGIS Plugin Repository
1. In QGIS, go to **Plugins → Manage and Install Plugins**
2. Go to the **Settings** tab and make sure **"Show also experimental plugins"** is checked
   - ⚠️ *Note: This setting is required temporarily. The experimental flag will be removed once the plugin has been successfully tested by a handful of users.*
3. Go to the **All** tab and search for **"TopoDrain"**
4. Select the TopoDrain plugin and click **Install Plugin**
   - ⚠️ **Important:** Make sure to install the newest version (at least version ≥0.1.7)

After installation, you will see TopoDrain tools in the **Processing Toolbox** under the TopoDrain section.

<img src="resources/TopoDrain_installed.png" alt="TopoDrain in Processing Toolbox" width="500">

## Recommended QGIS Plugins

### Profile Tool
The **Profile tool** plugin is highly recommended for verifying results created with TopoDrain tools. It allows you to plot terrain profiles, which is essential for:
- Checking the slope of created keylines respectively constant slope lines

**Installation:** Go to **Plugins → Manage and Install Plugins**, search for "Profile tool", and click **Install Plugin**.

### For Users in Switzerland: Swiss Geo Downloader
The **Swiss Geo Downloader** plugin is extremely useful for downloading Digital Terrain Data (swissALTI3D) directly within QGIS:
1. Install the plugin from **Plugins → Manage and Install Plugins**
2. Open the plugin: **Plugins → Swiss Geo Downloader**
3. Search for dataset **swissALTI3D**
4. Request file list
5. Choose best resolution!
6. Download tiles for your study site (TIF files)
7. Use the GDAL **"Merge"** tool to combine multiple tiles into a single TIF file



