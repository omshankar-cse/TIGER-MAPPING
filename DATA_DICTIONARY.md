# Data Dictionary: AI/ML-Based Tiger Habitat Prediction System

This document describes all relevant data files, their types, columns, spatial reference systems, and characteristics in the tiger habitat dataset for India.

---

## 1. Spatial Coordinates & Occurrence Data (GeoJSON Files)

Raw GeoJSON files are located in the `tiger dataset - Copy/tiger_in_<YEAR>/` directories from **2001 to 2020**. These files represent various spatial features like species ranges, survey locations, restoration areas, and administrative boundaries.

### Coordinate Reference System (CRS)
* **Projection**: WGS 84 (Geographic)
* **EPSG Code**: 4326
* **Units**: Decimal Degrees

### Core GeoJSON Files and Properties
Each GeoJSON file contains a list of geographic features. The fields defined in the `properties` block of these features are described below:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `country` | String / Null | Name of the country (e.g., `"India"`, `"Bhutan"`, `"Nepal"`, or `null` for administrative states). |
| `iso2` | String | Two-letter country code (e.g., `"IN"` for India, `"BT"` for Bhutan). |
| `isonumeric` | Integer | Three-digit ISO numeric code (e.g., `356` for India). |
| `area` | Float | Area of the spatial feature in square kilometers ($km^2$). |
| `protected`| Float | Area of the feature overlapping with protected areas ($km^2$). |
| `eph` | Float | **Effective Potential Habitat** area in square kilometers ($km^2$). |
| `lsid` | Integer | Landscape ID linking the spatial feature to metrics in the Excel sheets. |
| `ls_key` | String | Type of feature (e.g., `"scl_species"`, `"scl_survey"`, `"scl_restoration"`, `"scl_survey_fragment"`, etc.). |
| `biome` | String | Major biome category (e.g., `"Tropical & Subtropical Moist Broadleaf Forests"`, `"Temperate Broadleaf & Mixed Forests"`). |
| `ecoregion` | String | Specific ecological region name (e.g., `"Brahmaputra Valley semi-evergreen forests"`). |

#### A. Tiger Species Presence Range (`scl_species_<YEAR>.geojson`)
* **Geometry Types**: `Polygon`, `MultiPolygon`
* **Description**: Represents the geographic boundary of the tiger presence range for the given year.
* **Usage**: Features where `country == "India"` represent the tiger occurrence area. Centroids or points sampled within these polygons are treated as tiger presence points (`presence = 1`).

#### B. Tiger Survey Locations (`scl_survey_<YEAR>.geojson`)
* **Geometry Types**: `Polygon`, `MultiPolygon`
* **Description**: Represents the spatial boundaries of surveys conducted for tigers.
* **Usage**: Used to identify where surveys took place.

#### C. Habitat Restoration Zones (`scl_restoration_<YEAR>.geojson`)
* **Geometry Types**: `Polygon`, `MultiPolygon`
* **Description**: Represents candidate regions identified for tiger habitat restoration.

#### D. State and Administrative Boundaries (`scl_states_<YEAR>.geojson`)
* **Geometry Types**: `Polygon`, `MultiPolygon`
* **Description**: Represents state boundaries and regional administrative divisions.
* **Usage**: Features where `iso2 == "IN"` represent the 35 states and union territories of India. They are merged to form the study boundary of India.

---

## 2. Tabular Attributes & Landscape Trends (Excel Files)

Raw Excel files are named `tiger_<YEAR>_in.xlsx` and are located in `tiger dataset - Copy/tiger_in_<YEAR>/`. They contain sheets detailing tabular attributes, definitions, and metrics for each landscape.

### Sheet: `Landscapes`
This sheet lists tabular metrics for individual landscapes.

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Analysis date` | Integer | The year of analysis (e.g., `2001`). |
| `Lsid` | Integer | Landscape ID, matching the `lsid` property in the GeoJSON files. |
| `Landscape type`| String | The class of landscape (`"species"`, `"survey"`, `"restoration"`, `"survey_fragment"`, `"restoration_fragment"`). |
| `Structural habitat` | Float | Area of structural forest/habitat ($km^2$). |
| `Effective potential habitat` | Float | Area of effective potential habitat ($km^2$). |
| `% Protected` | Float | Percentage of the landscape area covered by protected areas (0.0 to 1.0). |

### Sheet: `Species landscape by admin`
Provides mapping of landscape IDs (`Lsid`) to administrative countries.

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Analysis date` | Integer | Year of analysis. |
| `ID` | Integer | Landscape ID (matches `Lsid`). |
| `Type` | String | Feature class type (e.g., `"species"`, `"survey"`). |
| `India` | Integer | Binary flag (`1` if the landscape covers India, `0` otherwise). |

### Sheet: `Species landscape by biome`
Maps landscape IDs (`Lsid`) to specific biomes (e.g., percentage overlap with Deciduous Forest, Tropical Moist Broadleaf Forest, etc.).

---

## 3. Climate & Environmental Data (WorldClim Rasters)

Climate data is stored as monthly raster GeoTIFF files under the `worlclim/` directory, split into two decade folders: `2000-2009` and `2010-2019`.

### Spatial Properties
* **CRS**: WGS 84 (EPSG:4326)
* **Resolution**: 2.5 arc-minutes (approximately $4.5 \text{ km} \times 4.5 \text{ km}$ grid spacing at the equator)
* **Extent**: Global coverage (cropped to India bounds `[68°E, 6°N, 98°E, 38°N]` during preprocessing)

### Climate Variables
* **Precipitation (`PREP`)**: Located in `precipitation/` subfolders. Filename format: `wc2.1_cruts4.09_2.5m_prec_YYYY-MM.tif`. Units: Millimeters ($mm$).
* **Maximum Temperature (`TMAX`)**: Located in `tmax/` subfolders. Filename format: `wc2.1_cruts4.09_2.5m_tmax_YYYY-MM.tif`. Units: Degrees Celsius (°C).
* **Minimum Temperature (`TMIN`)**: Located in `tmin/` subfolders. Filename format: `wc2.1_cruts4.09_2.5m_tmin_YYYY-MM.tif`. Units: Degrees Celsius (°C).

### Temporal Aggregation Rules
Because climate files are monthly and tiger occurrences are annual, monthly values are aggregated to annual statistics:
1. **Annual Precipitation**: Sum of monthly precipitation rasters for year $Y$.
2. **Annual TMAX**: Mean of monthly maximum temperature rasters for year $Y$.
3. **Annual TMIN**: Mean of monthly minimum temperature rasters for year $Y$.

*Note: Since 2020 climate layers are not present in the dataset (data ends in 2019), the 2019 annual aggregated layers are used as a proxy for 2020 occurrences.*
