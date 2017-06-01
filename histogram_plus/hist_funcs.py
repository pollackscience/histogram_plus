from __future__ import division
# import warnings
from numbers import Number
from collections import Iterable
# import itertools
import colorsys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.cbook as cbook
import pandas as pd

# from astroML.density_estimation import knuth_bin_width
# scotts_bin_width, freedman_bin_width,\

from bb.tools.bayesian_blocks_modified import bayesian_blocks


class HistContainer(object):
    """Class to hold histogram properties and members."""
    def __init__(self, x, bins, range, errorbars, scale, kwargs):

        if 'weights' in kwargs and kwargs['weights'] is not None:
            w = kwargs['weights']
            self.has_weights = True
        else:
            w = None
            self.has_weights = False

        x, w = self._checks_and_wrangling(x, w)

        # Prevent hiding of builtins, define members from args
        self.bin_range = range
        del range
        self.bins = bins
        self.errorbars = errorbars
        self.scale = scale
        self._arg_init(kwargs, w)
        self._df_binning_init(x, w)

        self.bin_content = None
        self.bin_error = None

        if self.normed:
            self.normalize()
        if self.scale:
            self.rescale(self.scale)

        if self.errorbars:
            self.draw_errorbars()


    def _checks_and_wrangling(self, x, w):
        # Manage the input data in the same fashion as mpl
        if np.isscalar(x):
            x = [x]

        input_empty = (np.size(x) == 0)

        # Massage 'x' for processing.
        if input_empty:
            x = np.array([[]])
        else:
            x = cbook._reshape_2D(x)

        self.n_data_sets = len(x)  # number of datasets

        # We need to do to 'weights' what was done to 'x'
        #if 'weights' in kwargs and kwargs['weights'] is not None:
        if w is not None:
            w = cbook._reshape_2D(w)

        if w is not None and len(w) != self.n_data_sets:
            raise ValueError('weights should have the same shape as x')

        if w is not None:
            for xi, wi in zip(x, w):
                if wi is not None and len(wi) != len(xi):
                    raise ValueError('weights should have the same shape as x')

        return x, w


    def _arg_init(self, kwargs, w):
        # Break up the kwargs into different chunks depending on the arg
        self.histtype = None
        if 'histtype' in kwargs:
            self.histtype = kwargs['histtype']
            # if histtype == 'marker':
            #     kwargs.pop('histtype')  # Don't want to feed this arg to plt.plot
        else:
            self.kwargs['histtype'] = 'stepfilled'

        if 'normed' in kwargs:
            self.normed = kwargs.pop('normed')
        else:
            self.normed = False


        self.stacked = False
        if 'stacked' in kwargs:
            self.stacked = kwargs['stacked']
            if self.histtype == 'marker' and self.stacked:
                raise ValueError('Do not stack with markers, that would be silly')

        if self.histtype == 'marker' and self.n_data_sets > 1:
            raise ValueError('`marker` histtype does not currently support multiple input datasets')

        if self.histtype == 'barstacked' and not self.stacked:
            self.stacked = True

        # Get current axis
        if 'ax' in kwargs:
            self.ax = kwargs.pop('ax')
        else:
            self.ax = plt.gca()

        # Group arguments into different dicts
        # For error bars:
        self.err_dict = {}
        if not (self.errorbars is None or self.errorbars is False):
            self.err_dict['errorbars'] = self.errorbars
            if 'err_type' in kwargs:
                self.err_dict['err_type'] = kwargs.pop('err_type')
            else:
                self.err_dict['err_type'] = 'line'
            if 'err_color' in kwargs:
                self.err_dict['err_color'] = kwargs.pop('err_color')
            else:
                self.err_dict['err_color'] = 'auto'
            if 'suppress_zero' in kwargs:
                self.err_dict['suppress_zero'] = kwargs.pop('suppress_zero')
            else:
                self.err_dict['suppress_zero'] = True

        # For data-driven binning
        self.bin_dict = {}
        if isinstance(self.bins, str):
            self.bin_dict['bins'] = self.bins
            if w is not None:
                raise TypeError('Weights are not supported for data-driven binning methods')
                # warnings.warn(
                #     "weights argument is not supported for this binning method: it will be ignored.")
                # kwargs.pop('weights')
            if 'fitness' in kwargs:
                self.bin_dict['fitness'] = kwargs.pop('fitness')
            if 'p0' in kwargs:
                self.bin_dict['p0'] = kwargs.pop('p0')

        self.hist_dict = kwargs

        if self.histtype == 'marker':
            self.hist_dict.pop('marker')

            if 'markerstyle' not in self.hist_dict:
                self.hist_dict['markerstyle'] = 'o'
            if 'linestyle' not in self.hist_dict:
                self.hist_dict['linestyle'] = ''


    def _df_binning_init(self, data, weights):
        '''Do an initial binning to get bin edges, total hist range, and break each set of data and
        weights into a dataframe (easier to handle errorbar calculation moving forward)'''

        # If bin edges are already determined, than skip initial histogramming
        self.bin_edges = None
        if isinstance(self.bins, Iterable) and not isinstance(self.bins, str):
            self.bin_edges = self.bins
            if self.bin_range is None:
                self.bin_range = (self.bin_edges[0], self.bin_edges[-1])

        # If bin edges need to be determined, there's a few different cases to consider
        else:
            if self.bin_range is None:
                xmin = np.inf
                xmax = -np.inf
                for i in xrange(self.n_data_sets):
                    if len(data[i]) > 0:
                        xmin = min(xmin, min(data[i]))
                        xmax = max(xmax, max(data[i]))
                self.bin_range = (xmin, xmax)

            # Special case for Bayesian Blocks
            if self.bins in ['block', 'blocks']:
                if self.n_data_sets == 1:
                    self.bin_edges = bayesian_blocks(t=data[0], fitness='events', p0=0.02)
                # Stacked data sets
                elif self.stacked:
                    self.bin_edges = bayesian_blocks(t=np.concatenate(data), fitness='events', p0=0.02)
                # Unstacked data
                else:
                    self.bin_edges = []
                    for i in xrange(self.n_data_sets):
                        self.bin_edges.append(bayesian_blocks(t=data[i], fitness='events', p0=0.02))
                    self.bin_edges = np.asarray(np.sort(np.concatenate(self.bin_edges)))

            else:
                _, self.bin_edges = np.histogram(data, bins=self.bins, weights=weights,
                                                 range=self.bin_range)

        self.widths = np.diff(self.bin_edges)
        self.bin_centers = self.bin_edges[:-1]+self.widths*0.5

        # Now put the data into dataframes with the weights and bins
        self.df_list = []
        for i in xrange(self.n_data_sets):
            if weights is None:
                df = pd.DataFrame({'data': data[i]})
            else:
                df = pd.DataFrame({'data': data[i], 'weights': weights[i]})
            df_bins = pd.cut(df.data, self.bin_edges, include_lowest=True)
            df['bins'] = df_bins
            self.df_list.append(df)


        # Make the initial histograms
        if self.histtype == 'marker':
            self.bin_content, _ = np.histogram(data, self.bin_edges, weights=weights, range=self.bin_range)
            self.vis_object = self.ax.plot(self.bin_centers, self.bin_content, **self.hist_dict)
        else:
            self.bin_content, _, self.vis_object = self.ax.hist(data, self.bin_edges,
                                                                weights=weights,
                                                                range=self.bin_range,
                                                                **self.hist_dict)
        if self.errorbars:
            self.calc_bin_error('default')


    def normalize(self):
        data = [df.data for df in self.df_list]
        if self.has_weights:
            weights = [df.weights for df in self.df_list]

        if self.histtype == 'marker':
            if self.n_data_sets > 1:
                raise ValueError('`histtype: marker` only supports a single input dataset.')
                # bin_content = []
                # for d in range(len(data)):
                #     if weights is None:
                #         w = None
                #     else:
                #         w = weights[d]
                #     bc, _ = np.histogram(data[d], bin_edges, weights=w, range=bin_range,
                #                          density=True)
                #     bin_content.append(bc)
            else:
                self.bin_content, _ = np.histogram(data, self.bin_edges, weights=weights,
                                                   range=self.bin_range, density=True)
        else:
            self.bin_content, _, self.vis_object = self.ax.hist(data, self.bin_edges, normed=True,
                                                                weights=weights,
                                                                range=self.bin_range,
                                                                **self.hist_dict)
        if self.errorbars:
            self.calc_bin_error('norm')


    def rescale(self, scale):
        if self.bin_content is None:
            raise ValueError('Cannot scale before histogramming')

        if isinstance(scale, Number):
            self.bin_content = np.multiply(self.bin_content, scale)
            if self.errorbars:
                self.calc_bin_error('scale', scale=scale)

        elif scale == 'binwidth':
            widths = np.diff(self.bin_edges)
            self.bin_content = np.divide(self.bin_content, widths)
            if self.errorbars:
                self.calc_bin_error('scale', scale = 1.0/widths)


    def calc_bin_error(self, err_type='gaussian', hist_mod='default', scale=None):
        data = [df.data for df in self.df_list]
        if self.has_weights:
            weights = [df.weights for df in self.df_list]

        if err_type == 'gaussian':
            if hist_mod == 'default':
                self.bin_err = np.sqrt(self.bin_content)

            elif hist_mod == 'norm':
                bin_content_no_norm, _ = np.histogram(data, self.bin_edges, weights=weights, range=self.bin_range)
                self.bin_err = np.sqrt(bin_content_no_norm)*(self.bin_content/bin_content_no_norm)

            elif hist_mod == 'scale':
                self.bin_err = np.multiply(self.bin_err, scale)

        else:
            raise KeyError('`err_type: {}` not implemented')


    def draw_errorbars(self):
        if self.n_data_sets == 1:
            bin_height = [self.bin_content]
            bin_err = [self.bin_err]
            if self.histtype != 'marker':
                vis_object = [self.vis_object]
        else:
            bin_height = self.bin_content
            bin_err = self.bin_err
            vis_object = self.vis_object

        for i in range(self.n_data_sets):
            if self.err_dict['err_color'] == 'auto':
                if self.histtype == 'marker':
                    err_color = colors.to_rgba(vis_object[i]._get_rgba_face())
                elif self.histtype in ['stepfilled', 'bar']:
                    err_color = colors.to_rgba(vis_object[i][0][0].get_facecolor())
                elif self.histtype == 'step':
                    err_color = colors.to_rgba(vis_object[i][0].get_edgecolor())

                hls_tmp = colorsys.rgb_to_hls(*err_color[:-1])
                err_color = list(colorsys.hls_to_rgb(hls_tmp[0], hls_tmp[1]*0.7, hls_tmp[2])) + \
                    [err_color[-1]]
            else:
                err_color = self.err_dict['err_color']

            if self.histtype == 'marker':
                self.ax.errorbar(self.bin_centers, bin_height[i], linestyle='', marker='',
                            yerr=bin_err[i], xerr=self.width*0.5, linewidth=2, color=err_color)
            else:
                self.ax.errorbar(self.bin_centers, bin_height[i], linestyle='', marker='',
                            yerr=bin_err[i], linewidth=2, color=err_color)


    def do_hist(self):
        '''Main plotting and rescaling function.'''

        # Special args for marker-plots
        if self.histtype == 'marker':
            if 'marker' in self.hist_dict:
                markerstyle = self.hist_dict.pop('marker')
            else:
                markerstyle = 'o'
            if 'linestyle' in self.hist_dict:
                linestyle = self.hist_dict.pop('linestyle')
            else:
                linestyle = ''
            do_marker = True
        else:
            do_marker = False

        data = [df.data for df in self.df_list]
        if 'weights' in self.df_list[0]:
            weights = [df.weights for df in self.df_list]
        else:
            weights = None

        redraw = False
        # Break into scaling cases
        if not self.normed and not self.scale:
            # Not normalized or scaled, process is straightforward.
            if do_marker:
                if len(data) > 1:
                    bin_content = []
                    for d in range(len(data)):
                        if weights is None:
                            w = None
                        else:
                            w = weights[d]
                        bc, _ = np.histogram(data[d], bin_edges, weights=w, range=bin_range)
                        bin_content.append(bc)
                else:
                    bin_content, _ = np.histogram(data, bin_edges, weights=weights, range=bin_range)
            else:
                bin_content, _, vis_object = ax.hist(data, bin_edges, weights=weights, range=bin_range,
                                                     **kwargs)
            bin_err = np.sqrt(bin_content)

        elif normed and (not scale or scale == 'binwidth'):
            # Just normalized: calculate both the normed and default hist (ratio is needed for proper
            # error bar calculation).  If 'binwdith' is selected for scale, ignore scaling (norm scales
            # by binwidth by default, as it is the correct way to renomalize a histogram with variable
            # bin sizes).
            if do_marker:
                if len(data) > 1:
                    bin_content = []
                    for d in range(len(data)):
                        if weights is None:
                            w = None
                        else:
                            w = weights[d]
                        bc, _ = np.histogram(data[d], bin_edges, weights=w, range=bin_range,
                                             density=True)
                        bin_content.append(bc)
                else:
                    bin_content, _ = np.histogram(data, bin_edges, weights=weights, range=bin_range,
                                                  density=True)
            else:
                bin_content, _, vis_object = ax.hist(data, bin_edges, normed=True, range=bin_range,
                                                     **kwargs)
            bin_content_no_norm, _ = np.histogram(data, bin_edges, density=False, range=bin_range)
            bin_err = np.sqrt(bin_content_no_norm)*(bin_content/bin_content_no_norm)

        elif not normed and isinstance(scale, Number):
            # Just scaled with a single number
            if do_marker:
                if len(data) > 1:
                    bin_content_no_scale = []
                    for d in range(len(data)):
                        if weights is None:
                            w = None
                        else:
                            w = weights[d]
                        bc, _ = np.histogram(data[d], bin_edges, weights=w, range=bin_range)
                        bin_content_no_scale.append(bc)
            else:
                bin_content_no_scale, _, vis_object = ax.hist(data, bin_edges, range=bin_range,
                                                              **kwargs)
                redraw = True
            bin_content = np.multiply(bin_content_no_scale, scale)
            bin_err = np.multiply(np.sqrt(bin_content_no_scale), scale)

        elif not normed and scale == 'binwidth':
            # Each bin should be divided by its bin width
            if do_marker:
                if len(data) > 1:
                    bin_content = []
                    for d in range(len(data)):
                        if weights is None:
                            w = None
                        else:
                            w = weights[d]
                        bc, _ = np.histogram(data[d], bin_edges, weights=w, range=bin_range)
                        bin_content_no_scale.append(bc)
                else:
                    bin_content_no_scale, _ = np.histogram(data, bin_edges, weights=weights,
                                                           range=bin_range)
            else:
                bin_content_no_scale, _, vis_object = ax.hist(data, bin_edges, weights=weights,
                                                              range=bin_range, **kwargs)
                redraw = True
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
            if do_marker:
                if len(data) > 1:
                    bin_content_norm = []
                    for d in range(len(data)):
                        if weights is None:
                            w = None
                        else:
                            w = weights[d]
                        bc, _ = np.histogram(data[d], bin_edges, weights=w, range=bin_range,
                                             density=True)
                        bin_content_norm.append(bc)
                else:
                    bin_content_norm, _ = np.histogram(data, bin_edges, weights=weights,
                                                       range=bin_range, density=True)
            else:
                bin_content_norm, _, vis_object = ax.hist(data, bin_edges, normed=True, weights=weights,
                                                          range=bin_range, **kwargs)
                redraw = True
            bin_content_no_norm, _ = np.histogram(data, bin_edges, density=False, range=bin_range)
            bin_content = np.multiply(bin_content_norm, scale)
            bin_err = np.multiply(np.sqrt(bin_content_no_norm),
                                  np.multiply(np.divide(bin_content_norm, bin_content_no_norm), scale))

        if do_marker:
            kwargs.pop('histtype')
            widths = bin_edges[1:] - bin_edges[:-1]
            bin_centers = bin_edges[:-1]+widths*0.5
            # if len(df_list) > 1:
            #     # vis_object = []
            #     # print bin_content
            #     # for i in range(len(df_list)):
            #     #     vis_object.append(ax.plot(bin_centers, bin_content[i], linestyle=linestyle,
            #     #                               marker=markerstyle, **kwargs))
            #     x_and_ys = list(itertools.chain.from_iterable(zip([bin_centers]*len(df_list),
            #                                                       bin_content)))
            #     vis_object = ax.plot(*x_and_ys, linestyle=linestyle,
            #                          marker=markerstyle, **kwargs)
            #     # print vis_object
            # else:
            vis_object = ax.plot(bin_centers, bin_content, linestyle=linestyle, marker=markerstyle,
                                 **kwargs)

        if len(df_list) == 1:
            bin_content = [bin_content]
            vis_object = [vis_object]

        if redraw:
            vis_object = _redraw_hist(bin_content, vis_object, histtype, len(df_list), ax)

        return bin_content, bin_err, vis_object





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

    # Generate a histogram object

    hist_con = HistContainer(x, bins, range, errorbars, scale, kwargs)
    hist_con.do_hist()
    hist_con.do_err_bars()

    return hist_con.bin_content, hist_con.bin_edges, hist_con.vis_object

def _do_err_bars(bin_height, bin_edges, bin_err, n_data_sets, ax, vis_object, **kwargs):
    print vis_object
    width = (bin_edges[1:]-bin_edges[:-1])
    bin_centers = bin_edges[:-1]+width*0.5
    if n_data_sets == 1:
        bin_height = [bin_height]
        bin_err = [bin_err]
        if kwargs['histtype'] != 'marker':
            vis_object = [vis_object]

    for i in range(n_data_sets):
        if kwargs['errcolor'] == 'inherit':
            if kwargs['histtype'] == 'marker':
                err_color = colors.to_rgba(vis_object[i]._get_rgba_face())
            elif kwargs['histtype'] in ['stepfilled', 'bar']:
                err_color = colors.to_rgba(vis_object[i][0][0].get_facecolor())
            elif kwargs['histtype'] == 'step':
                err_color = colors.to_rgba(vis_object[i][0].get_edgecolor())

            hls_tmp = colorsys.rgb_to_hls(*err_color[:-1])
            err_color = list(colorsys.hls_to_rgb(hls_tmp[0], hls_tmp[1]*0.7, hls_tmp[2])) + \
                [err_color[-1]]
        else:
            err_color = kwargs['errcolor']

        if kwargs['histtype'] == 'marker':
            ax.errorbar(bin_centers, bin_height[i], linestyle='', marker='',
                        yerr=bin_err[i], xerr=width*0.5, linewidth=2, color=err_color)
        else:
            ax.errorbar(bin_centers, bin_height[i], linestyle='', marker='',
                        yerr=bin_err[i], linewidth=2, color=err_color)


def _redraw_hist(bin_content, patches, histtype, n_data_sets, ax):
    for n in range(n_data_sets):
        if histtype == 'bar':
            for i, bc in enumerate(bin_content[n]):
                plt.setp(patches[n][i], 'height', bc)
        elif histtype == 'stepfilled' or histtype == 'step':
            xy = patches[n][0].get_xy()
            j = 0
            for i, bc in enumerate(bin_content[n]):
                xy[j+1, 1] = bc
                xy[j+2, 1] = bc
                j += 2
            plt.setp(patches[n][0], 'xy', xy)

    ax.relim()
    ax.autoscale_view(False, False, True)
    return patches


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
