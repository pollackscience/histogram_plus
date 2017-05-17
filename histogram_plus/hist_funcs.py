from __future__ import division
# import warnings
from numbers import Number
from collections import Iterable
import colorsys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.cbook as cbook
import pandas as pd

from astroML.density_estimation import\
    scotts_bin_width, freedman_bin_width,\
    knuth_bin_width

from bb.tools.bayesian_blocks_modified import bayesian_blocks


def hist(x, bins=10, range=None, errorbars=None, scale=None, **kwargs):
    """Enhanced histogram, based on `hist` function from astroML, which in turn is built off the
    `hist` function from matplotlib.  Various additions are also inspired from ROOT histograms.
    The main additional features are the ability to use data-driven binning algorithms,
    the addition of errorbars, and more scaling options (like dividing all bin values by their
    widths).

    Args:
    x (array_like, or list of array_like): Array of data to be histogrammed.
    bins (int or List or str, optional):  If `int`, `bins` number of equal-width bins are generated.
        The width is determined by either equal divison of the given `range`, or equal division
        between the first and last data point if no `range` is specified.

        If `List`, bin edges are taken directly from `List` (can be unequal width).

        If `str`, then it must be one of:
        'blocks' : use bayesian blocks for dynamic bin widths
        'knuth' : use Knuth's rule to determine bins
        'scott' : use Scott's rule to determine bins
        'freedman' : use the Freedman-diaconis rule to determine bins

    ax : Axes instance (optional)
        Specify the Axes on which to draw the histogram.  If not specified,
        then the current active axes will be used.

    scale : Number or str (optional)
        If Number, all bin contents are multiplied by the given value.
        If str:
            'binwidth' : every bin content is divided by the bin width.

    **kwargs :
        Overloaded kwargs variants:
            histtype:
                'markers' : plot the bin contents as markers, centered on the bin centers.
                If this method is chosen, all additional kwargs for `pylab.plot()` can be used.

        Other keyword arguments are described in `pylab.hist()`.



    Brian Pollack, 2017
    """

    # Manage the input data in the same fashion as mpl

    if np.isscalar(x):
        x = [x]

    input_empty = (np.size(x) == 0)

    # Massage 'x' for processing.
    if input_empty:
        x = np.array([[]])
    else:
        x = cbook._reshape_2D(x)
    nx = len(x)  # number of datasets
    # We need to do to 'weights' what was done to 'x'

    if 'weights' in kwargs and kwargs['weights'] is not None:
        w = cbook._reshape_2D(kwargs['weights'])
    else:
        w = [None]*nx

    if len(w) != nx:
        raise ValueError('weights should have the same shape as x')

    for xi, wi in zip(x, w):
        if wi is not None and len(wi) != len(xi):
            raise ValueError('weights should have the same shape as x')

    # Prevent hiding of builtins, get important args from kwargs
    bin_range = range
    del range

    histtype = None
    if 'histtype' in kwargs:
        histtype = kwargs['histtype']
        if histtype == 'marker':
            kwargs.pop('histtype')  # Don't want to feed this arg to plt.plot
    else:
        kwargs['histtype'] = 'stepfilled'

    if 'stacked' in kwargs:
        stacked = kwargs['stacked']
        if histtype == 'marker' and stacked:
            raise ValueError('Do not stack with markers, that would be silly')
    else:
        stacked = False

    if histtype == 'barstacked' and not stacked:
        stacked = True

    # if 'normed' in kwargs:
    #     normed = kwargs['normed']
    # else:
    #     normed = False

    # Get current axis
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        ax = plt.gca()

    # Group arguments into different dicts
    # For error bars:
    err_dict = {}
    if not (errorbars is None or errorbars is False):
        err_dict['errorbars'] = errorbars
        if 'errtype' in kwargs:
            err_dict['errtype'] = kwargs.pop('errtype')
        else:
            err_dict['errtype'] = 'line'
        if 'errcolor' in kwargs:
            err_dict['errcolor'] = kwargs.pop('errcolor')
        else:
            err_dict['errcolor'] = 'inherit'
        if 'suppress_zero' in kwargs:
            err_dict['suppress_zero'] = kwargs.pop('suppress_zero')
        else:
            err_dict['suppress_zero'] = True

    # For data-driven binning
    bin_dict = {}
    if isinstance(bins, str):
        bin_dict['bins'] = bins
        if "weights" in kwargs:
            raise TypeError('Weights are not supported for data-driven binning methods')
            # warnings.warn(
            #     "weights argument is not supported for this binning method: it will be ignored.")
            # kwargs.pop('weights')
        if 'fitness' in kwargs:
            bin_dict['fitness'] = kwargs.pop('fitness')
        if 'p0' in kwargs:
            bin_dict['p0'] = kwargs.pop('p0')

    df_list, bin_edges, bin_range = _get_initial_vars(x, bins, bin_range, nx, w, stacked, bin_dict)
    # Generate histogram-like object.
    # There are various possible cases that need to be considered separately.

    if histtype == 'marker':
        bc, err_scale, vis_object = _do_marker_hist(x, bin_edges, bin_range, scale, ax, **kwargs)
    else:
        bc, err_scale, vis_object = _do_std_hist(x, bin_edges, bin_range, scale, ax, **kwargs)

    if len(err_dict) > 0:
        err_dict['histtype'] = histtype
        _do_err_bars(bc, bin_edges, err_scale, ax, vis_object, **err_dict)

    return bc, bin_edges, vis_object


def _do_marker_hist(x, bins, bin_range, scale, ax, **kwargs):
    '''Private function for making a marker-based histogram
    (typically used to represent actual data)'''

    if 'normed' in kwargs:
        normed = kwargs.pop('normed')
    else:
        normed = False
    if 'marker' in kwargs:
        markerstyle = kwargs.pop('marker')
    else:
        markerstyle = 'o'
    if 'linestyle' in kwargs:
        linestyle = kwargs.pop('linestyle')
    else:
        linestyle = ''

    # Break into scaling cases

    if not normed and not scale:
        # Not normalized or scaled, process is straightforward.
        bin_content, bin_edges = np.histogram(x, bins, range=bin_range)
        bin_err = np.sqrt(bin_content)

    elif normed and not scale:
        # Just normalized: calculate both the normed and default hist (ratio is needed for proper
        # error bar calculation
        bin_content, bin_edges = np.histogram(x, bins, density=True, range=bin_range)
        bin_content_no_norm, _ = np.histogram(x, bins, density=False, range=bin_range)
        bin_err = np.sqrt(bin_content_no_norm)*(bin_content/bin_content_no_norm)

    elif not normed and isinstance(scale, Number):
        # Just scaled with a single number
        bin_content_no_scale, bin_edges = np.histogram(x, bins, range=bin_range)
        bin_content = bin_content_no_scale * scale
        bin_err = np.sqrt(bin_content_no_scale)*scale

    elif not normed and scale == 'binwidth':
        # Each bin should be divided by its bin width
        bin_content_no_scale, bin_edges = np.histogram(x, bins, range=bin_range)
        bin_content = np.ones(bin_content_no_scale.shape)
        bin_err = np.sqrt(bin_content_no_scale)
        widths = bin_edges[1:] - bin_edges[:-1]
        for i, bc in enumerate(bin_content_no_scale):
            bin_content[i] = (bc/widths[i])
            bin_err[i] /= widths[i]

    elif normed and isinstance(scale, Number):
        # Normed and scaled (with a number): Assume the user is not mistakenly using both options
        # (but this will throw a warning).  First normalize, then scale (otherwise the scale would
        # be irrelevent
        bin_content_norm, bin_edges = np.histogram(x, bins, density=True, range=bin_range)
        bin_content_no_norm, _ = np.histogram(x, bins, density=False, range=bin_range)
        bin_content = bin_content_norm*scale
        bin_err = np.sqrt(bin_content_no_norm)*(bin_content_norm/bin_content_no_norm)*scale

    elif normed and scale == 'binwidth':
        # Normed and scaled (by binwidth): Assume the user is not mistakenly using both options
        # (but this will throw a warning).  First scale by binwidth, then normalize (this is the
        # reverse order when scaling by a single number)
        bin_content_no_scale, bin_edges = np.histogram(x, bins, range=bin_range)
        bin_content_scale = np.ones(bin_content_no_scale.shape)
        bin_err = np.sqrt(bin_content_no_scale)
        widths = bin_edges[1:] - bin_edges[:-1]
        for i, bc in enumerate(bin_content_no_scale):
            bin_content_scale[i] = (bc/widths[i])
            bin_err[i] /= widths[i]
        total_content = np.sum(bin_content_scale*widths)
        bin_content = bin_content_scale/total_content
        bin_err /= total_content

    widths = bin_edges[1:] - bin_edges[:-1]
    bin_centers = bin_edges[:-1]+widths*0.5
    # bin_error = np.sqrt(bin_content)
    # err_low = np.asarray([poisson_error(bc, suppress_zero)[0] for bc in bin_content_raw])
    # err_hi = np.asarray([poisson_error(bc, suppress_zero)[1] for bc in bin_content_raw])
    # err_scale = bin_content/bin_content_raw
    # err_low *= err_scale
    # err_hi *= err_scale
    # bin_error = [err_low, err_hi]
    # if errorbars:
    #     vis_objects_err = ax.errorbar(bin_centers, bin_content, linestyle=linestyle,
    #                                   marker=markerstyle, yerr=bin_error, **kwargs)
    vis_object = ax.plot(bin_centers, bin_content, linestyle=linestyle, marker=markerstyle,
                         **kwargs)
    # if 'color' in kwargs:
    #     kwargs.pop('color')
    return bin_content, bin_err, vis_object


def _do_std_hist(x, bins, bin_range, scale, ax, **kwargs):
    '''Private function for making a standard histogram'''

    if 'normed' in kwargs:
        normed = kwargs.pop('normed')
    else:
        normed = False

    if 'histtype' in kwargs:
        histtype = kwargs['histtype']
    else:
        histtype = 'bar'

    # Break into scaling cases

    if not normed and not scale:
        # Not normalized or scaled, process is straightforward.
        bin_content, bin_edges, patches = ax.hist(x, bins, range=bin_range, **kwargs)
        bin_err = np.sqrt(bin_content)

    elif normed and not scale:
        # Just normalized: calculate both the normed and default hist (ratio is needed for proper
        # error bar calculation
        bin_content, bin_edges, patches = ax.hist(x, bins, normed=True, range=bin_range, **kwargs)
        bin_content_no_norm, _ = np.histogram(x, bins, density=False, range=bin_range)
        bin_err = np.sqrt(bin_content_no_norm)*(bin_content/bin_content_no_norm)

    elif not normed and isinstance(scale, Number):
        # Just scaled with a single number
        bin_content_no_scale, bin_edges, patches = ax.hist(x, bins, range=bin_range, **kwargs)
        bin_content = bin_content_no_scale * scale
        bin_err = np.sqrt(bin_content_no_scale)*scale
        _redraw_hist(bin_content, patches, histtype, ax)

    elif not normed and scale == 'binwidth':
        # Each bin should be divided by its bin width
        bin_content_no_scale, bin_edges, patches = ax.hist(x, bins, range=bin_range, **kwargs)
        bin_content = np.ones(bin_content_no_scale.shape)
        bin_err = np.sqrt(bin_content_no_scale)
        widths = bin_edges[1:] - bin_edges[:-1]
        for i, bc in enumerate(bin_content_no_scale):
            bin_content[i] = (bc/widths[i])
            bin_err[i] /= widths[i]
        _redraw_hist(bin_content, patches, histtype, ax)

    elif normed and isinstance(scale, Number):
        # Normed and scaled (with a number): Assume the user is not mistakenly using both options
        # (but this will throw a warning).  First normalize, then scale (otherwise the scale would
        # be irrelevent
        bin_content_norm, bin_edges, patches = ax.hist(x, bins, normed=True, range=bin_range,
                                                       **kwargs)
        bin_content_no_norm, _ = np.histogram(x, bins, density=False, range=bin_range)
        bin_content = bin_content_norm*scale
        bin_err = np.sqrt(bin_content_no_norm)*(bin_content_norm/bin_content_no_norm)*scale
        _redraw_hist(bin_content, patches, histtype, ax)

    elif normed and scale == 'binwidth':
        # Normed and scaled (by binwidth): Assume the user is not mistakenly using both options
        # (but this will throw a warning).  First scale by binwidth, then normalize (this is the
        # reverse order when scaling by a single number)
        bin_content_no_scale, bin_edges, patches = ax.hist(x, bins, range=bin_range, **kwargs)
        bin_content_scale = np.ones(bin_content_no_scale.shape)
        bin_err = np.sqrt(bin_content_no_scale)
        widths = bin_edges[1:] - bin_edges[:-1]
        for i, bc in enumerate(bin_content_no_scale):
            bin_content_scale[i] = (bc/widths[i])
            bin_err[i] /= widths[i]
        total_content = np.sum(bin_content_scale*widths)
        bin_content = bin_content_scale/total_content
        bin_err /= total_content
        _redraw_hist(bin_content, patches, histtype, ax)

    return bin_content, bin_err, patches


def _get_initial_vars(data, bins, bin_range, n_data_sets, weights, stacked, bin_dict):
    '''Do an initial binning to get bin edges, total hist range, and break each set of data and
    weights into a dataframe (easier to handle errorbar calculation moving forward)'''

    # If bin edges are already determined, than skip initial histogramming
    if isinstance(bins, Iterable) and not isinstance(bins, str):
        bin_edges = bins
        if bin_range is None:
            bin_range = (bin_edges[0], bin_edges[-1])

    # If bin edges need to be determined, there's a few different cases to consider
    else:
        if bin_range is None:
            xmin = np.inf
            xmax = -np.inf
            for i in xrange(n_data_sets):
                if len(data[i]) > 0:
                    xmin = min(xmin, data[i].min())
                    xmax = max(xmax, data[i].max())
            bin_range = (xmin, xmax)

        # Special case for Bayesian Blocks
        if bins in ['block', 'blocks']:
            if n_data_sets == 1:
                bin_edges = bayesian_blocks(t=data, fitness='events', p0=0.02)
            # Stacked data sets
            elif stacked:
                bin_edges = bayesian_blocks(t=data.ravel(), fitness='events', p0=0.02)
            # Unstacked data
            else:
                bin_edges = []
                for i in xrange(n_data_sets):
                    bin_edges.append(bayesian_blocks(t=data[i], fitness='events', p0=0.02))
                bin_edges = np.asarray(np.sort(np.concatenate(bin_edges)))

        else:
            # Just a single data set
            if n_data_sets == 1:
                _, bin_edges = np.histogram(data, bins=bins, weights=weights, range=bin_range)
            # Stacked data sets
            elif stacked:
                _, bin_edges = np.histogram(data.ravel(), bins=bins, weights=weights.ravel(),
                                            range=bin_range)
            # Unstacked data
            else:
                _, bin_edges = np.histogram(data, bins=bins, weights=weights, range=bin_range)

    # Now put the data into dataframes with the weights and bins
    df_list = []
    for i in xrange(n_data_sets):
        df = pd.DataFrame({'data': data[i], 'weights': weights[i]})
        df_bins = pd.cut(df.data, bin_edges, include_lowest=True)
        df['bins'] = df_bins
        df_list.append(df)

    return df_list, bin_edges, bin_range


def _do_err_bars(bin_height, bin_edges, bin_err, ax, vis_object, **kwargs):
    width = (bin_edges[1:]-bin_edges[:-1])
    bin_centers = bin_edges[:-1]+width*0.5
    # print bin_height, err_scale
    if kwargs['errcolor'] == 'inherit':
        if kwargs['histtype'] == 'marker':
            err_color = colors.to_rgba(vis_object[0].get_color())
        elif kwargs['histtype'] in ['stepfilled', 'bar']:
            err_color = colors.to_rgba(vis_object[0].get_facecolor())
        elif kwargs['histtype'] == 'step':
            err_color = colors.to_rgba(vis_object[0].get_edgecolor())

        hls_tmp = colorsys.rgb_to_hls(*err_color[:-1])
        err_color = list(colorsys.hls_to_rgb(hls_tmp[0], hls_tmp[1]*0.7, hls_tmp[2])) + \
            [err_color[-1]]
    else:
        err_color = kwargs['errcolor']

    if kwargs['histtype'] == 'marker':
        ax.errorbar(bin_centers, bin_height, linestyle='', marker='',
                    yerr=bin_err, xerr=width*0.5, linewidth=2, color=err_color)
    else:
        ax.errorbar(bin_centers, bin_height, linestyle='', marker='',
                    yerr=bin_err, linewidth=2, color=err_color)


def _redraw_hist(bin_content, patches, histtype, ax):
    if histtype == 'bar':
        for i, bc in enumerate(bin_content):
            plt.setp(patches[i], 'height', bc)
    elif histtype == 'stepfilled' or histtype == 'step':
        xy = patches[0].get_xy()
        j = 0
        for i, bc in enumerate(bin_content):
            xy[j+1, 1] = bc
            xy[j+2, 1] = bc
            j += 2
        plt.setp(patches[0], 'xy', xy)

    ax.relim()
    ax.autoscale_view(False, False, True)


def poisson_error(bin_content, suppress_zero=False):
    '''Returns a high and low 1-sigma error bar for an input bin value, as defined in:
    https://www-cdf.fnal.gov/physics/statistics/notes/pois_eb.txt
    If bin_content > 9, returns the sqrt(bin_content)'''
    error_dict = {
        0: (0.000000, 1.000000),
        1: (0.381966, 2.618034),
        2: (1.000000, 4.000000),
        3: (1.697224, 5.302776),
        4: (2.438447, 6.561553),
        5: (3.208712, 7.791288),
        6: (4.000000, 9.000000),
        7: (4.807418, 10.192582),
        8: (5.627719, 11.372281),
        9: (6.458619, 12.541381)}

    if suppress_zero and bin_content == 0:
        return (0, 0)
    elif bin_content in error_dict:
        return error_dict[bin_content]
    else:
        return (np.sqrt(bin_content), np.sqrt(bin_content))
