from __future__ import division
import warnings
from numbers import Number
import numpy as np
from matplotlib import pyplot as plt

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
    """

    x = np.asarray(x)

    # Prevent hiding of builtins
    bin_range = range
    del range

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
        if 'suppress_zero' in kwargs:
            err_dict['suppress_zero'] = kwargs.pop('suppress_zero')
        else:
            err_dict['suppress_zero'] = True

    # For data-driven binning
    bin_dict = {}
    if isinstance(bins, str):
        bin_dict['bins'] = bins
        if "weights" in kwargs:
            warnings.warn(
                "weights argument is not supported for this binning method: it will be ignored.")
            kwargs.pop('weights')
        if 'fitness' in kwargs:
            bin_dict['fitness'] = kwargs.pop('fitness')
        if 'p0' in kwargs:
            bin_dict['p0'] = kwargs.pop('p0')

    # For marker-style
    if 'histtype' in kwargs and kwargs['histtype'] == 'marker':
        marker = kwargs.pop('histtype')
    else:
        marker = None

    # Do any data-driven binning:
    if len(bin_dict) > 0:
        # if range is specified with a data-driven binning method, we need to truncate the data for
        # the bin-finding routines
        if (bin_range and len(bin_dict) > 0):
            x = x[(x >= bin_range[0]) & (x <= bin_range[1])]

        # Do binning
        if bins in ['block', 'blocks']:
            bins = bayesian_blocks(t=x, fitness='events', p0=0.02)
        elif bins in ['knuth', 'knuths']:
            dx, bins = knuth_bin_width(x, True, disp=False)
        elif bins in ['scott', 'scotts']:
            dx, bins = scotts_bin_width(x, True)
        elif bins in ['freedman', 'freedmans']:
            dx, bins = freedman_bin_width(x, True)
        elif isinstance(bins, str):
            raise ValueError("unrecognized bin argument: '{}'".format(bins))

    # Generate histogram-like object.
    # There are various possible cases that need to be considered separately.

    if marker:
        bc, be, err_scale, vis_object = _do_marker_hist(x, bins, bin_range, scale, ax, **kwargs)
    else:
        bc, be, err_scale, vis_object = _do_std_hist(x, bins, bin_range, scale, ax, **kwargs)

    if len(err_dict) > 0:
        err_dict['marker'] = marker
        _do_err_bars(bc, be, err_scale, ax, **err_dict)

    '''
    else:

        # if normed:
        #     bin_content_raw, _ = np.histogram(x, bins, density=False, range=hrange)
        #     bin_content_raw = np.asarray(bin_content_raw)
        # else:
        #     bin_content_raw = bin_content

        # err_low = np.asarray([poisson_error(bc, suppress_zero)[0] for bc in bin_content_raw])
        # err_hi = np.asarray([poisson_error(bc, suppress_zero)[1] for bc in bin_content_raw])
        # err_scale = bin_content/bin_content_raw
        # err_low *= err_scale
        # err_hi *= err_scale
        # bin_error = [err_low, err_hi]
        # width = (bins[1:]-bins[:-1])
        # bin_centers = bins[:-1]+width*0.5
        # vis_color = vis_objects[0].get_facecolor()
        # if 'histtype' in kwargs:
        #     if kwargs['histtype'] == 'step':
        #         vis_color = vis_objects[0].get_edgecolor()
        #     kwargs.pop('histtype')
        # if 'weights' in kwargs:
        #     kwargs.pop('weights')
        # if 'label' in kwargs:
        #     kwargs.pop('label')
        # if 'linewidth' in kwargs:
        #     kwargs.pop('linewidth')
        # if errorbars:
        #     vis_objects_err = ax.errorbar(bin_centers, bin_content, linestyle='', marker='.',
        #                                   yerr=bin_error, linewidth=2,
        #                                   color=vis_color, **kwargs)

# perform any scaling if necessary, including redrawing of the scaled objects
    if scale:
        bin_content_scaled = []
        if vis_objects is not None:
            if isinstance(vis_objects[0], matplotlib.patches.Rectangle):
                if scale == 'binwidth':
                    for i, bc in enumerate(bin_content):
                        width = (bins[i+1]-bins[i])
                        bin_content_scaled.append(bin_content[i]/width)
                        plt.setp(vis_objects[i], 'height', vis_objects[i].get_height()/width)
                elif isinstance(scale, Number):
                    for i, bc in enumerate(bin_content):
                        bin_content_scaled.append(bin_content[i]*scale)
                        plt.setp(vis_objects[i], 'height', vis_objects[i].get_height()*scale)
                else:
                    warnings.warn("scale argument value `", scale, "` not supported: it will be ignored.")

            elif isinstance(vis_objects[0], matplotlib.patches.Polygon):
                xy = vis_objects[0].get_xy()
                j = 0
                if scale == 'binwidth':
                    for i, bc in enumerate(bin_content):
                        width = (bins[i+1]-bins[i])
                        bin_content_scaled.append(bin_content[i]/width)
                        xy[j+1,1] = bin_content_scaled[i]
                        xy[j+2,1] = bin_content_scaled[i]
                        j+=2
                elif isinstance(scale, Number):
                    for i, bc in enumerate(bin_content):
                        bin_content_scaled.append(bin_content[i]*scale)
                        xy[j+1,1] = bin_content_scaled[i]
                        xy[j+2,1] = bin_content_scaled[i]
                        j+=2
                else:
                    warnings.warn("scale argument value `", scale, "` not supported: it will be ignored.")
                plt.setp(vis_objects[0], 'xy', xy)

        if vis_objects_err is not None:
            if scale == 'binwidth':
                for i, bc in enumerate(bin_content):
                    width = (bins[i+1]-bins[i])
                    if len(bin_content_scaled) != len(bin_content):
                        bin_content_scaled.append(bin_content[i]/width)
                    bin_error[0][i] /= width
                    bin_error[1][i] /= width
            elif isinstance(scale, Number):
                for i, bc in enumerate(bin_content):
                    if len(bin_content_scaled) != len(bin_content):
                        bin_content_scaled.append(bin_content[i]*scale)
                    bin_error[0][i] *= scale
                    bin_error[1][i] *= scale
            else:
                warnings.warn("scale argument value `", scale, "` not supported: it will be ignored.")
            bin_content_scaled = np.asarray(bin_content_scaled)
            vis_objects_err[0].set_ydata(bin_content_scaled)

            vis_objects_err[1][0].set_ydata(bin_content_scaled-bin_error[0])
            vis_objects_err[1][1].set_ydata(bin_content_scaled+bin_error[1])
            #vis_objects_err[1][0].set_ydata(bin_error[0])
            #vis_objects_err[1][1].set_ydata(bin_error[1])
            tmplines = vis_objects_err[2][0].get_segments()
            for i, bc in enumerate(bin_content_scaled):
                tmplines[i][0][1] = bin_content_scaled[i]-bin_error[0][i]
                tmplines[i][1][1] = bin_content_scaled[i]+bin_error[1][i]
                #tmplines[i][0][1] = bin_error[0][i]
                #tmplines[i][1][1] = bin_error[1][i]
            vis_objects_err[2][0].set_segments(tmplines)

        ax.relim()
        ax.autoscale_view(False,False,True)

    try:
        bc = bin_content_scaled
    except:
        bc = bin_content
    return bc, bins, vis_objects
    '''


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
    return bin_content, bin_edges, bin_err, vis_object


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

    return bin_content, bin_edges, bin_err, patches


def _do_err_bars(bin_height, bin_edges, bin_err, ax, **kwargs):
    width = (bin_edges[1:]-bin_edges[:-1])
    bin_centers = bin_edges[:-1]+width*0.5
    # print bin_height, err_scale
    if kwargs['marker']:
        ax.errorbar(bin_centers, bin_height, linestyle='', marker='',
                    yerr=bin_err, xerr=width*0.5, linewidth=2, color='k')
    else:
        ax.errorbar(bin_centers, bin_height, linestyle='', marker='',
                    yerr=bin_err, linewidth=2, color='k')


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
