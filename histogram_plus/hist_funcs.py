from __future__ import absolute_import
from __future__ import division
from six.moves import range
from six.moves import zip
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

from histogram_plus.bayesian_blocks_hep import bayesian_blocks
from histogram_plus.fill_between_steps import fill_between_steps

import warnings
warnings.filterwarnings('ignore')


def hist(x, bins='auto', range=None, weights=None, errorbars=False, normed=False, scale=None,
         stacked=False, histtype='stepfilled', **kwargs):
    """Enhanced histogram, based on `hist` function from matplotlib and astroML.
    The main additional features are the ability to use data-driven binning algorithms,
    the addition of errorbars, scaling options (like dividing all bin values by their
    widths), and marker-style draw options.  `hist` function wraps `HistContainer` class, which does
    the majority of work.

    Args:
        x (array_like, or list of array_like):  Array of data to be histogrammed.
        bins (int or List or str, optional):  If `int`, `bins` number of equal-width bins are
            generated.  The width is determined by either equal divison of the given `range`, or
            equal division between the first and last data point if no `range` is specified.

            If `List`, bin edges are taken directly from `List` (can be unequal width).

            If `str`, then it must be one of
                'blocks' : use bayesian blocks for dynamic bin widths.

                'auto' : use `auto` feature from numpy.histogram.

            Defaults to 'auto'.

        range (tuple or None, optional):  If specificed, data will only be considered and shown
            for the range given.  Otherwise, `range` will be between the highest and lowest
            datapoint.

            Defaults to None.

        weights (array_like or None, optional): Weights associated with each data point.  If
            specified, bin content will be equal to the sum of all relevant weights.

            Defaults to None.

        errorbars (boolean or array_like, optional):  If True, errorbars will be calculated and
            displayed based on the `err_*` arguments. The errorbars will be appropriately
            modified if `scale` and/or `normed` is True. If an array is specificed, those values
            will be used (and will not be modifed by any other methods).

            Defaults to False.

        normed (boolean, optional): If True, histogram will be normalized such that the integral
            over all bins with be equal to 1.  In general, this will NOT mean that the sum of all
            bin contents will be 1, unless all bin widths are equal to 1. If used with `scale`
            option, normalization happens before scale.

            Defaults to False

        scale (Number or 'binwidth', optional):  If Number, all bin contents are multiplied by the
            given value.  If 'binwidth', every bin content is divided by the bin width. If used with
            `normed` option, scaling occurs after normalization ('binwidth' will be ignored in this
            case, because it is handled automatically when normalizing).

            Defaults to None

        stacked (boolean, optional): If True, multiple input data sets will be layered on top of
            each other, such that the height of each bin is the sum of all the relevant dataset
            contributions.  If used with errorbars, the bars will be associated with the bin totals,
            not the individual components.

            Defaults to False.

        histtype (stepfilled', 'step', 'bar', or 'marker'): Draw options for histograms.
            'stepfilled', 'step', and 'bar' inherit from `matplotlib`.  'marker' places a single
            point at the center of each bin (best used with error bars).  'marker' will throw an
            exception if used with 'stacked' option.

            Defaults to 'stepfilled'.

        **kwargs :
            * ax : Axes instance (optional):
                Specify the Axes on which to draw the histogram.  If not specified,
                then the current active axes will be used, or a new axes instance will be generated.

        Other keyword arguments are described in `pylab.hist()`.



    Brian Pollack, 2017
    """

    # Generate a histogram object

    hist_con = HistContainer(x, bins, range, weights, errorbars, normed, scale, stacked,
                             histtype, kwargs)

    return hist_con.bin_content, hist_con.bin_edges, hist_con.vis_object


class HistContainer(object):
    """Class to hold histogram properties and members."""
    def __init__(self, x, bins, range, weights, errorbars, normed, scale, stacked, histtype,
                 kwargs):

        if weights is None:
            self.has_weights = False
        else:
            self.has_weights = True
        x, weights = self._checks_and_wrangling(x, weights)

        # Prevent hiding of builtins, define members from args
        self.bin_range = range
        del range
        self.bins = bins
        if isinstance(errorbars, Iterable):
            self.bin_err = errorbars
            self.errorbars = 'given'
        elif errorbars:
            self.errorbars = 'calc'
        else:
            self.errorbars = False
        self.normed = normed
        self.scale = scale
        self.stacked = stacked
        self.histtype = histtype
        self._arg_init(kwargs, weights)
        self._df_binning_init(x, weights)
        self.do_redraw = False

        if self.normed:
            self.normalize()
        if self.scale:
            self.rescale(self.scale)

        if self.histtype == 'marker':
            self.vis_object = self.ax.plot(self.bin_centers, self.bin_content, **self.hist_dict)

        elif self.do_redraw and not self.histtype == 'marker':
            self.redraw()

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
        # Break up the kwargs into different chunks depending on the arg, and set various defaults.

        # Scaling by `binwidth` is handled by default during normalization
        if self.normed and self.scale == 'binwidth':
            self.scale = None

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
        self.err_dict['errorbars'] = self.errorbars
        if 'err_style' in kwargs:
            self.err_dict['err_style'] = kwargs.pop('err_style')
        elif self.histtype in ['stepfilled', 'bar']:
            self.err_dict['err_style'] = 'band'
        else:
            self.err_dict['err_style'] = 'line'
        if 'err_color' in kwargs:
            self.err_dict['err_color'] = kwargs.pop('err_color')
        else:
            self.err_dict['err_color'] = 'auto'
        if 'suppress_zero' in kwargs:
            self.err_dict['suppress_zero'] = kwargs.pop('suppress_zero')
        else:
            self.err_dict['suppress_zero'] = True
        if 'err_type' in kwargs:
            self.err_dict['err_type'] = kwargs.pop('err_type')
        elif self.has_weights:
            self.err_dict['err_type'] = 'sumW2'
        else:
            self.err_dict['err_type'] = 'gaussian'

        # tweak histogram styles for `band` err_style

        if self.err_dict['err_style'] == 'band' and self.errorbars:
            if 'edgecolor' not in kwargs:
                kwargs['edgecolor'] = 'k'
            if 'linewidth' not in kwargs:
                kwargs['linewidth'] = 2

        # For data-driven binning
        self.bin_dict = {}
        if isinstance(self.bins, str):
            if 'gamma' in kwargs:
                self.bin_dict['gamma'] = kwargs.pop('gamma')
            if 'p0' in kwargs:
                self.bin_dict['p0'] = kwargs.pop('p0')

        self.hist_dict = kwargs
        if self.histtype != 'marker':
            self.hist_dict['histtype'] = self.histtype

        # set some marker defaults
        if self.histtype == 'marker':
            if 'marker' not in self.hist_dict:
                self.hist_dict['marker'] = 'o'
            if 'linestyle' not in self.hist_dict:
                self.hist_dict['linestyle'] = ''

        if 'alpha' not in self.hist_dict:
            self.hist_dict['alpha'] = 0.5

        if 'linewidth' not in self.hist_dict and self.histtype == 'step':
            self.hist_dict['linewidth'] = 2

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
            if self.stacked:
                _n_data_sets = 1
                b_data = [np.concatenate(data)]
                if self.has_weights:
                    b_weights = [np.concatenate(weights)]
                else:
                    b_weights = None
            else:
                _n_data_sets = self.n_data_sets
                b_data = data
                b_weights = weights

            if self.bin_range is None:
                xmin = np.inf
                xmax = -np.inf
                for i in range(_n_data_sets):
                    if len(data[i]) > 0:
                        xmin = min(xmin, min(b_data[i]))
                        xmax = max(xmax, max(b_data[i]))
                self.bin_range = (xmin, xmax)

            # Special case for Bayesian Blocks
            if self.bins in ['block', 'blocks']:

                # Single data-set or stacked
                if _n_data_sets == 1:

                    if self.has_weights:
                        b_weights = b_weights[0]
                    else:
                        b_weights = None
                    self.bin_edges = bayesian_blocks(data=b_data[0], weights=b_weights,
                                                     **self.bin_dict)
                else:
                    raise ValueError('Cannot use Bayesian Blocks with multiple, unstacked datasets')

            else:
                _, self.bin_edges = np.histogram(b_data, bins=self.bins, weights=b_weights,
                                                 range=self.bin_range)

        self.widths = np.diff(self.bin_edges)
        self.bin_centers = self.bin_edges[:-1]+self.widths*0.5

        # Now put the data into dataframes with the weights and bins
        self.df_list = []
        for i in range(self.n_data_sets):
            if weights is None:
                df = pd.DataFrame({'data': data[i]})
            else:
                df = pd.DataFrame({'data': data[i], 'weights': weights[i]})
            df_bins = pd.cut(df.data, self.bin_edges, include_lowest=True)
            df['bins'] = df_bins
            self.df_list.append(df)

        # Make the initial histograms
        if self.histtype == 'marker':
            self.bin_content, _ = np.histogram(data, self.bin_edges, weights=weights,
                                               range=self.bin_range)
            # self.vis_object = self.ax.plot(self.bin_centers, self.bin_content, **self.hist_dict)
        else:
            self.bin_content, _, self.vis_object = self.ax.hist(data, self.bin_edges,
                                                                weights=weights,
                                                                range=self.bin_range,
                                                                stacked=self.stacked,
                                                                **self.hist_dict)

        self.bin_content_orig = self.bin_content[:]

        if self.errorbars == 'calc' and not (self.normed or self.scale):
            self.calc_bin_error(hist_mod='default')

    def normalize(self):
        self.do_redraw = True
        data = [df.data for df in self.df_list]
        if self.has_weights:
            weights = [df.weights for df in self.df_list]
            if self.stacked:
                weights = np.concatenate(weights)
        elif self.n_data_sets == 1 or self.stacked:
            weights = None
        else:
            weights = [None for df in self.df_list]

        if self.n_data_sets == 1:
            self.bin_content, _ = np.histogram(data, self.bin_edges, weights=weights,
                                               range=self.bin_range, density=True)
        elif self.stacked:
            total_bin_content, _ = np.histogram(np.concatenate(data), self.bin_edges,
                                                weights=weights, range=self.bin_range, density=True)
            bin_scales = np.divide(total_bin_content, self.bin_content[-1])
            for i in range(self.n_data_sets):
                self.bin_content[i] = np.multiply(bin_scales, self.bin_content[i])

        else:
            self.bin_content = []
            for i, d in enumerate(data):
                self.bin_content.append(np.histogram(d, self.bin_edges, weights=weights[i],
                                                     range=self.bin_range, density=True)[0])

        if self.errorbars == 'calc':
            self.calc_bin_error(hist_mod='norm')

    def rescale(self, scale):
        self.do_redraw = True
        if self.bin_content is None:
            raise ValueError('Cannot scale before histogramming')

        if isinstance(scale, Number):
            self.bin_content = np.multiply(self.bin_content, scale)
            if self.errorbars == 'calc':
                self.calc_bin_error(hist_mod='scale', scale=scale, exist=self.normed)

        elif scale == 'binwidth':
            widths = np.diff(self.bin_edges)
            self.bin_content = np.divide(self.bin_content, widths)
            if self.errorbars == 'calc':
                self.calc_bin_error(hist_mod='scale', scale=1.0/widths, exist=self.normed)

    def calc_bin_error(self, hist_mod='default', scale=None, exist=False):

        # make new error bars if they haven't been calc'd
        if not exist:
            data = [df.data for df in self.df_list]
            if self.stacked:
                data = np.concatenate(data)

            if self.has_weights:
                weights = [df.weights for df in self.df_list]
                if self.stacked:
                    weights = np.concatenate(weights)
            elif self.n_data_sets == 1 or self.stacked:
                weights = None
            else:
                weights = [None for df in self.df_list]

            if self.n_data_sets == 1 or self.stacked:
                bin_content_no_norm, _ = np.histogram(data, self.bin_edges, weights=weights,
                                                      range=self.bin_range)
            else:
                bin_content_no_norm = []
                for i, d in enumerate(data):
                    bin_content_no_norm.append(np.histogram(d, self.bin_edges, weights=weights[i],
                                                            range=self.bin_range)[0])

            if self.err_dict['err_type'] == 'gaussian':
                bin_err_tmp = np.sqrt(bin_content_no_norm)
                # bin_err_tmp = np.sqrt(self.bin_content)

            elif self.err_dict['err_type'] == 'sumW2':
                bin_err_tmp = []
                if self.stacked:
                    df_list_tmp = [pd.concat(self.df_list, ignore_index=True)]
                else:
                    df_list_tmp = self.df_list

                for df in df_list_tmp:
                    df['weights2'] = np.square(df.weights)
                    bin_err_tmp.append(np.sqrt((
                        df.groupby('bins')['weights2'].sum().fillna(0).values)))

                bin_err_tmp = bin_err_tmp[-1]

            elif self.err_dict['err_type'] == 'poisson':
                pass

            else:
                raise KeyError('`err_type: {}` not implemented'.format(self.err_dict['err_type']))

            # Modifiy the error bars if needed (due to normalization or scaling)
            if hist_mod == 'default':
                self.bin_err = bin_err_tmp

            elif hist_mod == 'norm':
                if self.stacked:
                    bc = self.bin_content[-1]
                else:
                    bc = self.bin_content
                self.bin_err = bin_err_tmp*(np.divide(bc, bin_content_no_norm))

            elif hist_mod == 'scale':
                self.bin_err = np.multiply(bin_err_tmp, scale)

            else:
                raise KeyError('`hist_mod: {}` not implemented'.format(hist_mod))

        # if errors already exist due to norm calc
        else:
            self.bin_err = np.multiply(self.bin_err, scale)

    def draw_errorbars(self):
        if self.n_data_sets == 1:
            bin_height = [self.bin_content]
            bin_err = [self.bin_err]
            if self.histtype != 'marker':
                vis_object = [self.vis_object]
            else:
                vis_object = self.vis_object
        elif self.stacked:
            bin_height = [self.bin_content[-1]]
            bin_err = [self.bin_err]
            vis_object = [self.vis_object]
        else:
            bin_height = self.bin_content
            bin_err = self.bin_err
            vis_object = self.vis_object

        if not self.stacked:
            n_data_sets_eff = self.n_data_sets
        else:
            n_data_sets_eff = 1
        for i in range(n_data_sets_eff):
            if self.err_dict['err_color'] == 'auto' and not self.stacked:
                if self.histtype == 'marker':
                    err_color = colors.to_rgba(vis_object[i]._get_rgba_face())
                elif self.histtype in ['stepfilled', 'bar']:
                    err_color = colors.to_rgba(vis_object[i][0].get_facecolor())
                elif self.histtype == 'step':
                    err_color = colors.to_rgba(vis_object[i][0].get_edgecolor())

                hls_tmp = colorsys.rgb_to_hls(*err_color[:-1])
                err_color = list(colorsys.hls_to_rgb(hls_tmp[0], hls_tmp[1]*0.7, hls_tmp[2])) + \
                    [err_color[-1]]
            elif self.err_dict['err_color'] == 'auto' and self.stacked:
                err_color = next(self.ax._get_lines.prop_cycler)['color']
            else:
                err_color = self.err_dict['err_color']

            if self.histtype == 'marker':
                _, caps, _ = self.ax.errorbar(self.bin_centers, bin_height[i], linestyle='',
                                              marker='', yerr=bin_err[i], xerr=self.widths*0.5,
                                              linewidth=2, color=err_color)
            else:
                if self.err_dict['err_style'] == 'line':
                    self.ax.errorbar(self.bin_centers, bin_height[i], linestyle='', marker='',
                                     yerr=bin_err[i], linewidth=2, color=err_color)

                elif self.err_dict['err_style'] == 'band':
                    fill_between_steps(self.ax, self.bin_edges, bin_height[i]+bin_err[i],
                                       bin_height[i]-bin_err[i], step_where='pre', linewidth=0,
                                       color=err_color, alpha=self.hist_dict['alpha']*0.8,
                                       zorder=10)

    def redraw(self):
        self.bc_scales = np.divide(self.bin_content, self.bin_content_orig)
        self.do_redraw = False
        if self.n_data_sets == 1:
            bin_content = [self.bin_content]
            if self.histtype != 'marker':
                vis_object = [self.vis_object]
        else:
            bin_content = self.bin_content
            vis_object = self.vis_object

        for n in range(self.n_data_sets):
            if self.stacked:
                xy = vis_object[n][0].get_xy()
                j = 0
                for bcs in self.bc_scales[-1]:
                    xy[j+1, 1] *= bcs
                    xy[j+2, 1] *= bcs
                    j += 2
                if self.histtype == 'step':
                    xy[0, 1] *= self.bc_scales[-1][0]
                    xy[-1, 1] *= self.bc_scales[-1][-1]
                elif self.histtype in ['bar', 'stepfilled']:
                    for bcs in self.bc_scales[-1][::-1]:
                        xy[j+1, 1] *= bcs
                        xy[j+2, 1] *= bcs
                        j += 2
                    xy[0, 1] = xy[-1, 1]
                plt.setp(vis_object[n][0], 'xy', xy)

            else:
                if self.histtype == 'bar':
                    for i, bc in enumerate(bin_content[n]):
                        plt.setp(vis_object[n][i], 'height', bc)
                elif self.histtype == 'stepfilled' or self.histtype == 'step':
                    xy = vis_object[n][0].get_xy()
                    j = 0
                    for bc in bin_content[n]:
                        xy[j+1, 1] = bc
                        xy[j+2, 1] = bc
                        j += 2
                    plt.setp(vis_object[n][0], 'xy', xy)

        self.ax.relim()
        self.ax.autoscale_view(False, False, True)


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
