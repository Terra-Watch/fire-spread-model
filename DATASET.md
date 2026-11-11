# Fire-spread dataset: data card

Cases: **2930** fires from 2019-01-02 to 2025-04-26 (start of the forecast), 25 states, 165 one-degree blocks. Splits are by 1 x 1 degree block (dev 2017, test 913); five folds by block hash for tuning. Usable for scoring (at least 5 burned cells and 50 other cells): 24 h 2930, 48 h 2929.

## What one case is
Two consecutive satellite looks at one forest fire; the forecast starts at the first. Each case has 48 picture layers (192 x 192 cells of 100 m, centred on the fire), 38 numbers, and labels for 24 h and 48 h.

## Labels
* `y_modis`: cells that burned within the horizon according to the MODIS burned-area product (500 m, dated to the day), only the patches touching this fire, outside the 5-cell start zone. The main target.
* `y_modis_all`: all burned cells in the square.
* `y_viirs_new`: cells within 190 m of a later VIIRS detection of this fire that is more than 0.75 km from everything seen up to the start; `y_viirs_any`: any later detection (includes the fire's own persistence).
* `burn_rel_day`: MODIS burn day relative to the start day (0-5), -1 = none.
* `cand` marks the cells asked about (5 to 80 cells from the start, MODIS-mapped, not burned earlier), `modis_valid` the cells MODIS mapped.

## Layers and numbers
| name | unit | source | resolution | how |
|---|---|---|---|---|

| seed | 0/1 | NASA FIRMS VIIRS 375 m | 375 m | cells within 190 m of a detection of this fire up to the start |
| det_frp | MW | FIRMS VIIRS | 375 m | summed fire radiative power of the fire's detections, spread to the cells they cover |
| det_age_h | h | FIRMS VIIRS | 375 m | hours between the detection and the start time |
| dist | cells of 100 m | derived | 100 m | distance to the fire footprint |
| tree | share 0-1 | ESA WorldCover v200 (2021) | 10 m | share of 10 m pixels of that class in the 100 m cell |
| shrub | share | ESA WorldCover | 10 m | as above |
| grass | share | ESA WorldCover | 10 m | as above |
| crop | share | ESA WorldCover | 10 m | as above |
| built | share | ESA WorldCover | 10 m | as above |
| water | share | ESA WorldCover | 10 m | as above |
| bare | share | ESA WorldCover | 10 m | as above |
| forest_share | share | boundary-defined forest-land mask | 0.001 deg (about 110 m) | forest land by legal boundary, not tree cover |
| forest_5 | share | derived | 500 m | forest_share averaged over 5 x 5 cells |
| forest_15 | share | derived | 1.5 km | forest_share averaged over 15 x 15 cells |
| tree_5 | share | derived | 500 m | tree share averaged over 5 x 5 cells |
| tree_15 | share | derived | 1.5 km | tree share averaged over 15 x 15 cells |
| nonburn_dist | cells | derived | 100 m | distance to the nearest cell that cannot burn (built, water, bare, little vegetation) |
| nonburn_frac15 | share | derived | 1.5 km | share of non-burnable cells in 15 x 15 |
| dist_built | cells | WorldCover | 100 m | distance to the nearest cell with at least 30 % built-up |
| dist_crop | cells | WorldCover | 100 m | distance to the nearest cell with at least 50 % cropland |
| dist_water | cells | WorldCover | 100 m | distance to the nearest cell with at least 30 % water |
| elev | m | Copernicus DEM GLO-30 | 30 m | averaged to 100 m |
| rel_elev | m | Copernicus DEM | 100 m | elevation minus the fire's mean elevation |
| tpi | m | Copernicus DEM | 1.1 km | elevation minus its 11 x 11 cell mean |
| slope | degrees | Copernicus DEM | 100 m | Horn slope |
| aspect_sin | -1..1 | Copernicus DEM | 100 m | sine of the uphill direction |
| aspect_cos | -1..1 | Copernicus DEM | 100 m | cosine of the uphill direction |
| slope_along | tan | Copernicus DEM | 100 m | slope in the direction from the fire to the cell (positive uphill from the fire) |
| ndvi | - | MODIS MOD13Q1 v061 | 250 m | latest 16-day composite that started at least 16 days before the start |
| ndvi_prev | - | MODIS MOD13Q1 | 250 m | the same composite a year earlier |
| ndvi_anom | - | MODIS | 250 m | ndvi minus ndvi_prev (negative: drier than a year ago) |
| s2_nir | reflectance | Sentinel-2 L2A B08 | 10 m | newest clear pixel in the 45 days before the start, averaged to 100 m |
| s2_swir1 | reflectance | Sentinel-2 L2A B11 | 20 m | as above |
| s2_swir2 | reflectance | Sentinel-2 L2A B12 | 20 m | as above |
| s2_ndmi | - | Sentinel-2 | 100 m | (NIR - SWIR1) / (NIR + SWIR1): vegetation moisture |
| s2_nbr | - | Sentinel-2 | 100 m | (NIR - SWIR2) / (NIR + SWIR2): burn ratio |
| s2_age | days | Sentinel-2 | 100 m | days between the scene and the start |
| s2_valid | 0/1 | Sentinel-2 | 100 m | 1 where a clear pixel was found |
| road_dist | cells | OpenStreetMap (January 2021 extract) | 100 m | distance to the nearest motorway, trunk, primary, secondary, tertiary, unclassified road or track |
| river_dist | cells | OpenStreetMap | 100 m | distance to the nearest river |
| pop | people/km2 | WorldPop 1 km (2019 or 2020) | 1 km | bilinear to 100 m |
| path_t | h | fire-spread engine, no wind | 100 m | fuel-aware travel time from the fire (goes round barriers) |
| burn_days_since | days | MODIS MCD64A1 | 500 m | days since the latest burn in the 365 days before the start; 366 = none |
| burn_n | months | MODIS MCD64A1 | 500 m | months with a burn in that year |
| fire_hist_90 | detections | FIRMS VIIRS | 375 m | detections within reach of the cell in the 90 days before the start |
| fire_hist_365 | detections | FIRMS VIIRS | 375 m | same, 365 days |
| fire_days_since | days | FIRMS VIIRS | 375 m | days since the latest detection at the cell, 366 = none |
| other_fires_72h | detections | FIRMS VIIRS | 375 m | detections of other fires in the last 72 hours |
| n_start | log1p count | FIRMS | - | detections of the fire up to the start |
| frp_sum_log | log1p MW | FIRMS | - | summed power |
| extent_km | km | FIRMS | - | bounding-box diagonal of the fire |
| act5 | log1p count | FIRMS | - | forest detections of other fires within 5 km in the 72 h before |
| act20 | log1p count | FIRMS | - | same, 20 km |
| act_ev20 | log1p count | FIRMS | - | number of other fires within 20 km |
| lat | deg | - | - | fire centre |
| lon | deg | - | - | fire centre |
| month_sin | - | - | - | season |
| month_cos | - | - | - | season |
| doy_sin | - | - | - | day of year |
| doy_cos | - | - | - | day of year |
| hour_sin | - | - | - | local solar hour of the start |
| hour_cos | - | - | - | local solar hour of the start |
| wind | m/s | Open-Meteo archived forecast stream | hourly | mean vector wind over the horizon |
| wind_u | m/s | Open-Meteo | hourly | east component of the mean wind |
| wind_v | m/s | Open-Meteo | hourly | north component |
| wind_max | m/s | Open-Meteo | hourly | strongest hourly wind in the horizon |
| gust_max | m/s | Open-Meteo | hourly | strongest gust in the horizon |
| temp_mean | C | Open-Meteo | hourly | mean over the horizon |
| temp_max | C | Open-Meteo | hourly | maximum |
| rh_mean | % | Open-Meteo | hourly | mean relative humidity |
| rh_min | % | Open-Meteo | hourly | minimum |
| vpd_max | kPa | Open-Meteo | hourly | largest vapour pressure deficit |
| rain_sum | mm | Open-Meteo | hourly | rain over the horizon |
| moist_mean | fraction | derived from Open-Meteo | hourly | dead-fuel moisture (equilibrium moisture with a 10 h lag, rain pushes it up) |
| moist_min | fraction | derived | hourly | driest hour |
| rain7 | mm | ERA5 daily via Open-Meteo | daily | rain in the 7 days before the start day |
| rain30 | mm | ERA5 | daily | 30 days |
| rain60 | mm | ERA5 | daily | 60 days |
| days_dry | days | ERA5 | daily | days since at least 2 mm of rain |
| balance30 | mm | ERA5 | daily | 30-day rain minus evaporation |
| tmax14 | C | ERA5 | daily | mean daily maximum, 14 days |
| rh14 | % | ERA5 | daily | mean humidity, 14 days |
| sm7 | m3/m3 | ERA5 hourly | hourly | soil moisture 0-7 cm, last 24 h before the start |
| sm28 | m3/m3 | ERA5 | hourly | soil moisture 7-28 cm |
| sm7_change | m3/m3 | ERA5 | hourly | change in 0-7 cm moisture over the last 7 days |
| vpd3 | kPa | ERA5 | hourly | mean vapour pressure deficit, last 72 h |

## Audit results
* Missing data: layers with more than 10 % missing: {'ndvi': 0.314, 'ndvi_prev': 0.314, 'ndvi_anom': 0.315, 's2_nir': 0.997, 's2_swir1': 0.997, 's2_swir2': 0.997, 's2_ndmi': 0.997, 's2_nbr': 0.997, 's2_age': 0.997, 'burn_days_since': 0.999, 'burn_n': 0.999}.
* Share of cases with each source: has_dem 100%, has_ndvi 69%, has_s2 1%, has_burnhist 0%, has_osm 100%.
* Weather source used: {'hist': 2919, 'era5': 11}.
* Spatial leakage: 15.3% of dev fires lie within 25 km of a test fire (flag `near_test_25km` in `meta_all.parquet`; drop them for a clean variant).

* Label statistics (mean over parts): 24 h: MODIS burned share of asked cells 2.18%, VIIRS new 0.26%, share of VIIRS new cells inside MODIS-burned 40%; 48 h: MODIS burned share of asked cells 2.93%, VIIRS new 0.56%, share of VIIRS new cells inside MODIS-burned 44%.

## Known limits
* The MODIS burned-area truth is 500 m and misses small burns; the VIIRS truth is 375 m and misses cloudy passes. The two agree only partly (see above).
* Weather in the horizon comes from the archived forecast stream (a forecast, not observations).
* Land cover is 2021, roads 2021, population 2019/2020, regardless of the fire year.
* Canopy height is not included: the only open global map we could reach has no overviews and row-strip tiling, about ten minutes per case.
* Fires larger than 40 km, with fewer than 3 detections at the start, or outside forest land are not in the set.
* Fire seasons differ a lot between years (2021 has about 7 times the March detections of 2020).
