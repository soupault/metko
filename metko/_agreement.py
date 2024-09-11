import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def bland_altman(m1, m2, diff_mode="unit"):
    """

    Args:
        m1: (k, ) ndarray or list
        m2: (k, ) ndarray or list
        diff_mode: {"unit", "percentage"}

    Returns:
        out: dict
    """
    if len(m1) != len(m2):
        raise ValueError("m1 does not have the same length as m2.")
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    if diff_mode == "percentage":
        diffs = diffs / means
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)
    out = {"means": means, "diffs": diffs,
           "mean_diff": mean_diff, "std_diff": std_diff}
    return out


def bland_altman_plot(m1, m2, *, sd_limit=1.96, scatter_kws=None,
                      mean_line_kws=None, limit_lines_kws=None,
                      diff_mode="unit", regression_line=False):
    """Bland-Altman Plot.

    A Bland-Altman plot is a graphical method to analyze the differences
    between two methods of measurement -- m1 and m2. The mean of the measures
    is plotted against their difference.

    Args:
        m1: ndarray or list
        m2: ndarray or list
        sd_limit : float
            The limit of agreements expressed in terms of the standard deviation of
            the differences. If `md` is the mean of the differences, and `sd` is
            the standard deviation of those differences, then the limits of
            agreement that will be plotted will be
                           md - sd_limit * sd, md + sd_limit * sd
            The default of 1.96 will produce 95% confidence intervals for the means
            of the differences.
            If sd_limit = 0, no limits will be plotted, and the ylimit of the plot
            defaults to 3 standard deviations on either side of the mean.
        scatter_kws: dict
            Options to style the scatter plot. Passed to Axes.scatter.
        mean_line_kws: dict
            Options to style the mean line plot. Passed to Axes.axhline.
        limit_lines_kws: dict
            Options to style the limit lines. Passed to Axes.axhline.
        diff_mode: {"unit", "percentage"}
            Express differences as units or proportionally to the measurement magnitude.
        regression_line: bool
            Whether to draw a regression line. Useful to detect proportional bias.

    Returns:
        fig: matplotlib Figure
        ax: matplotlib Axis
    """
    if sd_limit < 0:
        raise ValueError("sd_limit ({}) is less than 0.".format(sd_limit))

    ret = bland_altman(m1=m1, m2=m2, diff_mode=diff_mode)
    means = ret["means"]
    diffs = ret["diffs"]
    mean_diff = ret["mean_diff"]
    std_diff = ret["std_diff"]

    # Configure plotting
    xlabel = "Mean"
    ylabel = "Difference" if diff_mode == "unit" else "Difference / mean, %"
    scatter_kws = scatter_kws or {}
    if "s" not in scatter_kws:
        scatter_kws["s"] = 20
    mean_line_kws = mean_line_kws or {}
    limit_lines_kws = limit_lines_kws or {}
    for kws in [mean_line_kws, limit_lines_kws]:
        if 'color' not in kws:
            kws['color'] = 'gray'
        if 'linewidth' not in kws:
            kws['linewidth'] = 1
    if 'linestyle' not in mean_line_kws:
        kws['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kws:
        kws['linestyle'] = ':'

    # Construct plot
    fig, ax = plt.subplots()
    # Measurement pairs
    ax.scatter(means, diffs, **scatter_kws)
    # Mean line annotated with mean difference
    ax.axhline(mean_diff, **mean_line_kws)
    ax.annotate('mean diff:\n{}'.format(np.round(mean_diff, 2)),
                xy=(0.99, 0.55),
                horizontalalignment='right',
                xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (2 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kws)
        ax.annotate('-{}*SD:\n{}'.format(sd_limit, np.round(lower, 2)),
                    xy=(0.99, 0.30),
                    horizontalalignment='right',
                    xycoords='axes fraction')
        ax.annotate('+{}*SD:\n{}'.format(sd_limit, np.round(upper, 2)),
                    xy=(0.99, 0.80),
                    horizontalalignment='right',
                    xycoords='axes fraction')
    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)
    else:
        raise ValueError("sd_limit is less than 0.")

    if regression_line:
        ci = (stats.norm.cdf(sd_limit) - stats.norm.cdf(-sd_limit)) * 100
        sns.regplot(x=means, y=diffs, scatter=False, ci=ci, ax=ax)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    return fig, ax


def taffe_bias_precision():
    # TODO: Taffe method
    raise NotImplemented("taffe_bias_precision")


def taffe_bias_precision_plot():
    # TODO: taffe_bias_precision_plot
    raise NotImplemented("taffe_bias_precision_plot")
