## Time series decomposition on Helheim velocity
## 6 May 2020  EHU
import numpy as np
import matplotlib.pyplot as plt
import iceutils as ice
import sys

## Set up combined hdf5 stack
#fpath='/Users/lizz/Documents/Research/Gld-timeseries/Stack/'
hel_stack = ice.MagStack(files=['vx.h5', 'vy.h5'])
data_key = 'igram' # B. Riel convention for access to datasets in hdf5 stack

## Extract time series at selected points
xy_1 = (308103., -2577200.) #polar stereo coordinates of a point near Helheim 2009 terminus, in m
xy_2 = (302026., -2566770.) # point up on North branch
xy_3 = (297341., -2571490.) # point upstream on main branch
xy_4 = (294809., -2577580.) # point on southern tributary
xys = (xy_1, xy_2, xy_3, xy_4)
labels = ('Near terminus', 'North branch', 'Main branch', 'South branch')

series = [hel_stack.timeseries(xy=xyi, key=data_key) for xyi in xys]

## Set up design matrix and perform lasso regression, according to Bryan's documentation
def build_collection(dates):
    """
    Function that creates a list of basis functions for a given datetime vector dates.
    """
    # Get date bounds
    tstart, tend = dates[0], dates[-1]

    # Initalize a collection and relevant basis functions
    collection = ice.tseries.timefn.TimefnCollection()
    periodic = ice.tseries.timefn.fnmap['periodic']
    ispl = ice.tseries.timefn.fnmap['isplineset']
    poly = ice.tseries.timefn.fnmap['poly']

    # Add polynomial first for secular components
    collection.append(poly(tref=tstart, order=1, units='years'))

    # Add seasonal terms
    collection.append(periodic(tref=tstart, units='years', period=0.5,
                               tmin=tstart, tmax=tend))
    collection.append(periodic(tref=tstart, units='years', period=1.0,
                               tmin=tstart, tmax=tend))
    
    # Integrated B-slines for transient signals
    # In general, we don't know the timescales of transients a prior
    # Therefore, we add integrated B-splines of various timescales where the
    # timescale is controlled by the 'nspl' value (this means to divide the time
    # vector into 'nspl' equally spaced spline center locations)
    for nspl in [128, 64, 32, 16, 8, 4]:
        collection.append(ispl(order=3, num=nspl, units='years', tmin=tstart, tmax=tend))
    
    # Done
    return collection

# Create an evenly spaced time array for time series predictions
t_grid = np.linspace(hel_stack.tdec[0], hel_stack.tdec[-1], 1000)

# First convert the time vectors to a list of datetime
dates = ice.tdec2datestr(hel_stack.tdec, returndate=True)
dates_grid = ice.tdec2datestr(t_grid, returndate=True)

# Build the collection
collection = build_collection(dates)

# Instantiate a model for inversion
model = ice.tseries.Model(dates, collection=collection)

# Instantiate a model for prediction
model_pred = ice.tseries.Model(dates_grid, collection=collection)

## Access the design matrix for plotting
G = model.G
# plt.plot(hel_stack.tdec, G)
# plt.xlabel('Year')
# plt.ylabel('Amplitude')
# plt.show()

# First create a solver with a minimum data threshold of 200 (i.e., we require at least
# 200 valid data points to perform an inversion). This should file for all four time series
ridge = ice.tseries.select_solver('ridge', reg_indices=model.itransient, penalty=2, n_min=200)

# Un-comment the following two lines to ensure warnings are printed out every time
#import warnings
#warnings.simplefilter('always', UserWarning)

# Loop over time series
for i in range(len(series)):
    # This should raise a warning and return a FAIL status (=0)
    status, m, Cm = ridge.invert(model.G, series[i])
    if status == ice.FAIL:
        print('Unsuccessful inversion')

# Create ridge regression solver that damps out the transient spline coefficients
#solver = ice.tseries.select_solver('ridge', reg_indices=model.itransient, penalty=2)
solver = ice.tseries.select_solver('lasso', reg_indices=model.itransient, penalty=1.5, rw_iter=1)

# Loop over time series
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
for i in range(len(series)):

    # Perform inversion to get coefficient vector and coefficient covariance matrix
    status, m, Cm = solver.invert(model.G, series[i]) # fit near-terminus (series[0]) first
    assert status == ice.SUCCESS, 'Failed inversion'
    
    # Model will perform predictions
    pred = model_pred.predict(m)

    # Separate out seasonal (short-term) and secular + transient (long-term) signals
    short_term = pred['seasonal']
    long_term = pred['secular'] + pred['transient']

    # Remove long-term signals from data
    series_short_term = series[i] - np.interp(hel_stack.tdec, t_grid, long_term)
    
    #print(len(pred['full']), len(series[0]))
    #print(sum(np.isnan(pred['full'])), sum(np.isnan(series[0])))
    #print(np.nanmean(pred['full']), np.nanmean(series[0]))
    
    # Plot long-term
    ax1.plot(hel_stack.tdec, series[i], '.')
    ax1.plot(t_grid, long_term, label=labels[i])

    # Plot short-term (add estimated bias (m[0]) for visual clarity)
    ax2.plot(hel_stack.tdec, series_short_term + m[0], '.')
    ax2.plot(t_grid, short_term + m[0], label=labels[i])

for ax in (ax1, ax2):
    ax.set_xlabel('Year')
    ax.set_ylabel('Velocity')

ax1.set_title('Multi-annual fit')
ax2.set_title('Seasonal fit')

plt.show()
