import sys
import datacube
import numpy as np
import xarray as xr
from deafrica_tools.datahandling import load_ard
from deafrica_tools.bandindices import calculate_indices
from odc.algo import xr_geomedian
from datacube.utils import masking


def geomedian_with_indices_wrapper(ds):
    """Apply xr_geomedian function and calculate a series of indices"""

    indices = ["NDVI", "LAI", "SAVI", "MSAVI", "MNDWI"]
    satellite_mission = "s2"

    ds_geomedian = xr_geomedian(ds)

    ds_geomedian = calculate_indices(
        ds_geomedian,
        index=indices,
        drop=False,
        satellite_mission=satellite_mission,
    )

    return ds_geomedian


def indices_wrapper(ds):
    """Calculate list of indices"""

    indices = ["NDVI", "LAI", "SAVI", "MSAVI", "MNDWI"]
    satellite_mission = "s2"

    ds = calculate_indices(
        ds,
        index=indices,
        drop=False,
        satellite_mission=satellite_mission,
    )

    return ds


def median_wrapper(ds):
    """Calculate median"""

    ds = ds.median(dim="time")

    return ds


def mean_wrapper(ds):
    """calculate median"""

    ds = ds.mean(dim="time")

    return ds


def apply_function_over_custom_times(ds, func, func_name, time_ranges):
    """Apply generic function over an xarray dataset"""

    output_list = []

    for timelabel, timeslice in time_ranges.items():

        if isinstance(timeslice, slice):
            ds_timeslice = ds.sel(time=timeslice)
        else:
            ds_timeslice = ds.sel(time=timeslice, method="nearest")

        ds_modified = func(ds_timeslice)

        rename_dict = {
            key: f"{key}_{func_name}_{timelabel}" for key in list(ds_modified.keys())
        }

        ds_modified = ds_modified.rename(name_dict=rename_dict)

        if "time" in list(ds_modified.coords):
            ds_modified = ds_modified.reset_coords().drop_vars(["time", "spatial_ref"])

        output_list.append(ds_modified)

    return output_list


# Define functions to load features
def feature_layers(query):
    """Compute feature layers according to datacube query"""
    
    baseline_query = query.copy() # include to make sure original query isn't modified

    # Connnect to datacube
    dc = datacube.Datacube(app="crop_type_ml")

    # Check query for required time ranges and remove them
    if all(
        [
            key in baseline_query.keys()
            for key in [
                "annual_geomedian_times",
                "semiannual_geomedian_times",
                "s1_time_ranges",
                "time_ranges",
            ]
        ]
    ):
        pass
    else:
        print(
            "Query missing at least one of annual_geomedian_times, semiannual_geomedian_times, s1_time_ranges, or time_ranges"
        )
        sys.exit(1)

    # ----------------- STORE TIME RANGES FOR CUSTOM QUERIES -----------------
    # This removes these items from the query so it can be used for loads

    annual_geomedian_times = baseline_query.pop("annual_geomedian_times")
    semiannual_geomedian_times = baseline_query.pop("semiannual_geomedian_times")
    s1_time_ranges = baseline_query.pop("s1_time_ranges")
    time_ranges = baseline_query.pop("time_ranges")

    # ----------------- DEFINE MEASUREMENTS TO USE FOR EACH PRODUCT -----------------

    s2_measurements = [
        "blue",
        "green",
        "red",
        "nir",
        "swir_1",
        "swir_2",
        "red_edge_1",
        "red_edge_2",
        "red_edge_3",
    ]

    s2_geomad_measurements = s2_measurements + ["smad", "emad", "bcmad"]

    s1_measurements = ["vv", "vh"]

    fc_measurements = ["bs", "pv", "npv", "ue"]

    rainfall_measurements = ["rainfall"]

    slope_measurements = ["slope"]

    # ----------------- S2 CUSTOM GEOMEDIANS -----------------
    # These are designed to take the geomedian for every range in time_ranges
    # This is controlled through the input query
    
    s2_query = baseline_query.copy()
    
    s2_query_times = list(time_ranges.values())
    s2_start_date = s2_query_times[0].start
    s2_end_date = s2_query_times[-1].stop
    s2_query.update({"time": (s2_start_date, s2_end_date)})

    ds = load_ard(
        dc=dc,
        products=["s2_l2a"],
        measurements=s2_measurements,
        group_by="solar_day",
        verbose=False,
        **s2_query,
    )
    
    
    # Apply geomedian over time ranges and calculate band indices
    s2_geomad_list = apply_function_over_custom_times(
        ds, geomedian_with_indices_wrapper, "s2", time_ranges
    )
    
    
    # ----------------- S2 ANNUAL GEOMEDIAN -----------------

    # Update query to use annual_geomedian_times
    ds_annual_geomad_query = baseline_query.copy()
    annual_query_times = list(annual_geomedian_times.values())
    annual_start_date = annual_query_times[0]
    annual_end_date = annual_query_times[-1]
    ds_annual_geomad_query.update({"time": (annual_start_date, annual_end_date)})

    # load s2 annual geomedian
    ds_s2_geomad = dc.load(
        product="gm_s2_annual",
        measurements=s2_geomad_measurements,
        **ds_annual_geomad_query,
    )

    # Calculate band indices
    s2_annual_list = apply_function_over_custom_times(
        ds_s2_geomad, indices_wrapper, "s2", annual_geomedian_times
    )
    
    
    # ----------------- S2 SEMIANNUAL GEOMEDIAN -----------------

    # Update query to use semiannual_geomedian_times
    ds_semiannual_geomad_query = baseline_query.copy()
    semiannual_query_times = list(semiannual_geomedian_times.values())
    semiannual_start_date = semiannual_query_times[0]
    semiannual_end_date = semiannual_query_times[-1]
    ds_semiannual_geomad_query.update({"time": (semiannual_start_date, semiannual_end_date)})

    # load s2 semiannual geomedian
    ds_s2_semiannual_geomad = dc.load(
        product="gm_s2_semiannual",
        measurements=s2_geomad_measurements,
        **ds_semiannual_geomad_query,
    )

    # Calculate band indices
    s2_semiannual_list = apply_function_over_custom_times(
        ds_s2_semiannual_geomad, indices_wrapper, "s2", semiannual_geomedian_times
    )

    # ----------------- S1 CUSTOM GEOMEDIANS -----------------

    # Update query to suit Sentinel 1
    s1_query = baseline_query.copy()
    s1_query.update({"sat_orbit_state": "ascending"})
    
    s1_query_times = list(s1_time_ranges.values())
    s1_start_date = s1_query_times[0].start
    s1_end_date = s1_query_times[-1].stop
    s1_query.update({"time": (s1_start_date, s1_end_date)})

    # Load s1
    s1_ds = load_ard(
        dc=dc,
        products=["s1_rtc"],
        measurements=s1_measurements,
        group_by="solar_day",
        verbose=False,
        **s1_query,
    )

    # Apply geomedian
    s1_geomad_list = apply_function_over_custom_times(
        s1_ds, xr_geomedian, "s1_xrgm", s1_time_ranges
    )

    # -------- LANDSAT BIMONTHLY FRACTIONAL COVER -----------

    # Update query to suit fractional cover
    fc_query = baseline_query.copy()
    fc_query.update({"resampling": "bilinear", "measurements": fc_measurements})
    
    fc_query_times = list(time_ranges.values())
    fc_start_date = fc_query_times[0].start
    fc_end_date = fc_query_times[-1].stop
    fc_query.update({"time": (fc_start_date, fc_end_date)})

    # load fractional cover
    ds_fc = dc.load(product="fc_ls", collection_category="T1", **fc_query)
    
    
    # Make a clear (no-cloud) and dry (no-water) pixel mask
    # load wofls
    ds_wofls = dc.load(product='wofs_ls',
                like=ds_fc.geobox,
                time=fc_query['time'],
                collection_category='T1')
    
    clear_and_dry = masking.make_mask(ds_wofls, dry=True).water
    
    #keep mostly clear scenes by calculating the number of good pixels per scene and applying a threshold
    #set a good data fraction
    min_gooddata = 0.95

    #keep only the images that are at least as clear as min_gooddata
    good_slice = clear_and_dry.mean(['x','y']) >= min_gooddata
    
    #apply the "clear mask" and filter to just the scenes that are mostly free of cloud and water
    ds_fc_clear = ds_fc.where(clear_and_dry).isel(time=good_slice)

    # Apply median
    fc_median_list = apply_function_over_custom_times(
        ds_fc_clear, median_wrapper, "median", time_ranges
    )

    # -------- CHIRPS MONTHLY RAINFALL -----------

    # Update query to suit CHIRPS rainfall
#     rainfall_query = query.copy()
#     rainfall_query.update(
#         {"resampling": "bilinear", "measurements": rainfall_measurements}
#     )

#     # Load rainfall and update no data values
#     ds_rainfall = dc.load(product="rainfall_chirps_monthly", **rainfall_query)

#     rainfall_nodata = -9999.0
#     ds_rainfall = ds_rainfall.where(
#         ds_rainfall.rainfall != rainfall_nodata, other=np.nan
#     )

#     # Apply mean
#     rainfall_mean_list = apply_function_over_custom_times(
#         ds_rainfall, mean_wrapper, "mean", time_ranges
#     )

    # -------- DEM SLOPE -----------
    slope_query = baseline_query.copy()
    slope_query.update(
        {
            "resampling": "bilinear",
            "measurements": slope_measurements,
            "time": "2000-01-01",
        }
    )
    
    # Load slope data and update no data values and coordinates
    ds_slope = dc.load(product="dem_srtm_deriv", **slope_query)

    slope_nodata = -9999.0
    ds_slope = ds_slope.where(ds_slope != slope_nodata, np.nan)

    ds_slope = ds_slope.squeeze("time")#.reset_coords("time", drop=True)

    # ----------------- FINAL MERGED XARRAY -----------------

    # Create a list to keep all items for final merge
    ds_list = []
    ds_list.extend(s2_geomad_list)
    ds_list.extend(s2_annual_list)
    ds_list.extend(s2_semiannual_list)
    ds_list.extend(s1_geomad_list)
    ds_list.extend(fc_median_list)
#     ds_list.extend(rainfall_mean_list)
    ds_list.append(ds_slope)

    ds_final = xr.merge(ds_list)
    

    return ds_final
