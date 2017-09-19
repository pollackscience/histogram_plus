Error Bars
==========

It is common practice in High Energy Physics (and other fields) to display error bars for each
histogram bin, corresponding to either the associated statistical or systematic errors.  While the
interpretation and meaning of the aforementioned error bars is a point of constant contention
(`Those Deceiving Error Bars <http://www.science20.com/quantum_diaries_survivor/those_deceiving_error_bars-85735>`), the entries in a histogram bin are typically seen as drawn from a Poisson distribution, and thus the standard deviation for a bin with `N` entries is `sqrt(N)`.

Drawing Error Bars
------------------

Let us produce a simple histogram with 



