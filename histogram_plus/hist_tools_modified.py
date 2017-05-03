from __future__ import division
import warnings
from numbers import Number
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from astroML.density_estimation import\
    scotts_bin_width, freedman_bin_width,\
    knuth_bin_width

#from bb_poly import bayesian_blocks
from bayesian_blocks_modified import bayesian_blocks
#from fill_between_steps import fill_between_steps


def hist(x, bins=10, fitness='events', gamma=None, p0=0.05, errorbars=None, suppress_zero=False, *args, **kwargs):
    """Enhanced histogram, based on `hist` function from astroML.

    This is a histogram function that enables the use of more sophisticated
    algorithms for determining bins.  Aside from the `bins` argument allowing
    a string specified how bins are computed, additional scaling, errorbar, and
    plotting methods are introduced.  All other kwargs can be used as in `pylab.hist()`.

    Parameters
    ----------
    x : array_like
        Array of data to be histogrammed

    bins : int or list or str (optional)
        If bins is a string, then it must be one of:
        'blocks' : use bayesian blocks for dynamic bin widths
        'knuth' : use Knuth's rule to determine bins
        'scott' : use Scott's rule to determine bins
        'freedman' : use the Freedman-diaconis rule to determine bins

    fitness : str
        Param used for Bayesian Blocks binning.

    gamma: Number
        Param used for Bayesian Blocks binning, ignored if `p0` is present

    p0 : Number
        Fake rate value for Bayesian Blocks binning, supersedes `gamma`

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
# do initial checks and set-up for overloaded arguments
    x = np.asarray(x)

    if isinstance(bins, str) and "weights" in kwargs:
        warnings.warn("weights argument is not supported for this binning method: it will be ignored.")
        kwargs.pop('weights')

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        ax = plt.gca()

    # if range is specified, we need to truncate the data for
    # the bin-finding routines
    if ('range' in kwargs and kwargs['range'] is not None and (bins in ['blocks',
                                        'knuth', 'knuths',
                                        'scott', 'scotts',
                                        'freedman', 'freedmans'])):
        x = x[(x >= kwargs['range'][0]) & (x <= kwargs['range'][1])]


    if bins in ['block','blocks']:
        bins = bayesian_blocks(t=x,fitness=fitness,p0=p0,gamma=gamma)
    elif bins in ['knuth', 'knuths']:
        dx, bins = knuth_bin_width(x, True, disp=False)
    elif bins in ['scott', 'scotts']:
        dx, bins = scotts_bin_width(x, True)
    elif bins in ['freedman', 'freedmans']:
        dx, bins = freedman_bin_width(x, True)
    elif isinstance(bins, str):
        raise ValueError("unrecognized bin code: '%s'" % bins)

    if 'scale' in kwargs:
        scale = kwargs.pop('scale')
    else:
        scale = None
    if scale and "stacked" in kwargs:
        warnings.warn("scaling is not currently supported for stacked histograms: scaling will be ignored.")
        scale = None

    if 'histtype' in kwargs and kwargs['histtype'] == 'marker':
        marker = kwargs.pop('histtype')
    else:
        marker = None

# generate histogram-like object
    vis_objects = None
    vis_objects_err = None
    if marker:
        if 'normed' in kwargs:
            normed = kwargs.pop('normed')
        else:
            normed = False
        if 'marker' in kwargs:
            markerstyle = kwargs.pop('marker')
        else:
            markerstyle = '.'
        if 'linestyle' in kwargs:
            linestyle = kwargs.pop('linestyle')
        else:
            linestyle = ''
        hrange = None
        if 'range' in kwargs:
            hrange = kwargs.pop('range')

        bin_content, bins = np.histogram(x, bins, density=normed, range=hrange)
        bin_content = np.asarray(bin_content, dtype=float)
        if normed:
            bin_content_raw, _ = np.histogram(x, bins, density=False, range=hrange)
            bin_content_raw = np.asarray(bin_content_raw)
        else:
            bin_content_raw = bin_content
        width = (bins[1:]-bins[:-1])
        bin_centers = bins[:-1]+width*0.5
        # bin_error = np.sqrt(bin_content)
        err_low = np.asarray([poisson_error(bc, suppress_zero)[0] for bc in bin_content_raw])
        err_hi = np.asarray([poisson_error(bc, suppress_zero)[1] for bc in bin_content_raw])
        err_scale = bin_content/bin_content_raw
        err_low *= err_scale
        err_hi *= err_scale
        bin_error = [err_low, err_hi]
        if errorbars:
            vis_objects_err = ax.errorbar(bin_centers, bin_content, linestyle=linestyle,
                                          marker=markerstyle, yerr=bin_error, **kwargs)
        else:
            vis_objects = ax.plot(bin_centers, bin_content, linestyle=linestyle, marker=markerstyle,
                                  **kwargs)
        if 'color' in kwargs:
            kwargs.pop('color')

    else:
        if 'normed' in kwargs:
            normed = kwargs.pop('normed')
        else:
            normed = False
        hrange = None
        if 'range' in kwargs:
            hrange = kwargs.pop('range')

        bin_content, bins, vis_objects = ax.hist(x, bins, range=hrange, normed=normed, **kwargs)
        if 'color' in kwargs:
            kwargs.pop('color')
        bin_content = np.asarray(bin_content, dtype=float)
        if normed:
            bin_content_raw, _ = np.histogram(x, bins, density=False, range=hrange)
            bin_content_raw = np.asarray(bin_content_raw)
        else:
            bin_content_raw = bin_content

        err_low = np.asarray([poisson_error(bc, suppress_zero)[0] for bc in bin_content_raw])
        err_hi = np.asarray([poisson_error(bc, suppress_zero)[1] for bc in bin_content_raw])
        err_scale = bin_content/bin_content_raw
        err_low *= err_scale
        err_hi *= err_scale
        bin_error = [err_low, err_hi]
        width = (bins[1:]-bins[:-1])
        bin_centers = bins[:-1]+width*0.5
        vis_color = vis_objects[0].get_facecolor()
        if 'histtype' in kwargs:
            if kwargs['histtype'] == 'step':
                vis_color = vis_objects[0].get_edgecolor()
            kwargs.pop('histtype')
        if 'weights' in kwargs:
            kwargs.pop('weights')
        if 'label' in kwargs:
            kwargs.pop('label')
        if 'linewidth' in kwargs:
            kwargs.pop('linewidth')
        if errorbars:
            vis_objects_err = ax.errorbar(bin_centers, bin_content, linestyle='', marker='.',
                                          yerr=bin_error, linewidth=2,
                                          color=vis_color, **kwargs)

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
