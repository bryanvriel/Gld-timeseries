## Time series decomposition on Helheim velocity
## 6 May 2020  EHU
import numpy as np
import matplotlib.pyplot as plt
import datetime
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
    bspl = ice.tseries.timefn.fnmap['bspline']
    ispl = ice.tseries.timefn.fnmap['isplineset']
    poly = ice.tseries.timefn.fnmap['poly']

    # Add polynomial first for secular components
    collection.append(poly(tref=tstart, order=1, units='years'))

    # Use B-splines for seasonal (short-term) signals
    Δtdec = 1.0 / 5.0 # years
    Δt = datetime.timedelta(days=int(Δtdec*365))
    t_current = tstart
    while t_current <= tend:
        collection.append(bspl(order=3, scale=Δtdec, units='years', tref=t_current))
        t_current += Δt
        
    # Integrated B-slines for transient signals
    # In general, we don't know the timescales of transients a prior
    # Therefore, we add integrated B-splines of various timescales where the
    # timescale is controlled by the 'nspl' value (this means to divide the time
    # vector into 'nspl' equally spaced spline center locations)
    for nspl in [128, 64, 32, 16, 8, 4]:
        collection.append(ispl(order=3, num=nspl, units='years', tmin=tstart, tmax=tend))
    
    # Done
    return collection

# Build a priori covariance matrix, mainly for repeating B-splines
def computeCm(collection, b_spl_sigma=1.0):
    from scipy.linalg import block_diag

    # Define some prior sigmas (large prior sigmas for secular and transient)
    sigma_secular = 100.0
    sigma_bspl = b_spl_sigma
    sigma_ispl = 100.0

    # Get lengths of different model partitions
    fnParts = ice.tseries.timefn.getFunctionTypes(collection)
    nBspl = len(fnParts['seasonal'])
    nIspl = len(fnParts['transient'])
    nSecular = len(fnParts['secular'])

    # Diagonal prior for secular components
    C_sec = sigma_secular**2 * np.eye(nSecular)

    # Get decimal times of B-spline centers   
    tdec = []
    for basis in collection.data:
        if basis.fnname == 'BSpline':
            tdec.append(ice.datestr2tdec(pydtime=basis.tref))
    tdec = np.array(tdec)

    # Correlation with splines w/ similar knot times
    # Decaying covariance w/ time
    tau = 1.0
    C_bspl = np.zeros((nBspl, nBspl))
    for i in range(nBspl):
        weight_decay = np.exp(-1.0 * np.abs(tdec - tdec[i]) / tau)
        ind = np.zeros((nBspl,), dtype=bool)
        ind[i::5] = True
        ind[i::-5] = True
        C_bspl[i,ind] = weight_decay[ind]
    C_bspl *= sigma_bspl**2

    # Diagonal prior for integrated B-splines
    C_ispl = sigma_ispl**2 * np.eye(nIspl)

    return block_diag(C_sec, C_bspl, C_ispl)

# Create an evenly spaced time array for time series predictions
t_grid = np.linspace(hel_stack.tdec[0], hel_stack.tdec[-1], 1000)

# First convert the time vectors to a list of datetime
dates = ice.tdec2datestr(hel_stack.tdec, returndate=True)
dates_grid = ice.tdec2datestr(t_grid, returndate=True)

# Build the collection
collection = build_collection(dates)

# Construct a priori covariance
Cm = computeCm(collection)
iCm = np.linalg.inv(Cm)

# Instantiate a model for inversion
model = ice.tseries.Model(dates, collection=collection)

# Instantiate a model for prediction
model_pred = ice.tseries.Model(dates_grid, collection=collection)

## Access the design matrix for plotting
G = model.G

# Create lasso regression solver that does the following:
# i) Uses an a priori covariance matrix for damping out the B-splines
# ii) Uses sparsity-enforcing regularization (lasso) on the integrated B-splines
solver = ice.tseries.select_solver('lasso', reg_indices=model.itransient, penalty=0.05,
                                   rw_iter=1, regMat=iCm)

# List of B-spline sigmas for a priori covariance matrix 
# (using spatially varying sigma values seems to help)
sigmas = [2.0, 1.5, 1.5, 1.5]

# Loop over time series
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,6))
for i in range(len(series)):

    # Construct a priori covariance
    Cm = computeCm(collection, b_spl_sigma=sigmas[i])
    solver.regMat = np.linalg.inv(Cm)
    
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

    # Remove short-term signals from data
    series_long_term = series[i] - np.interp(hel_stack.tdec, t_grid, short_term)
    
    # Plot full data
    ax1.plot(hel_stack.tdec, series[i], '.')
    ax1.plot(t_grid, pred['full'], label=labels[i])

    # Plot short-term (add estimated bias (m[0]) for visual clarity)
    ax2.plot(hel_stack.tdec, series_short_term + m[0], '.')
    ax2.plot(t_grid, short_term + m[0], label=labels[i])
    
    # Plot long-term
    ax3.plot(hel_stack.tdec, series_long_term, '.')
    ax3.plot(t_grid, long_term, label=labels[i])
    
ylim = ax1.get_ylim()
for ax in (ax1, ax2, ax3):
    ax.set_xlabel('Year')
    ax.set_ylabel('Velocity')
    ax.set_ylim(ylim)
    ax.set_xlim(t_grid[0], t_grid[-1])

ax1.set_title('Full fit')
ax2.set_title('Seasonal fit')
ax3.set_title('Multi-annual fit')

plt.show()
