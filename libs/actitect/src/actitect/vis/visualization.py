from dataclasses import dataclass
from datetime import datetime, timedelta, time
from itertools import cycle
from typing import Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn import metrics

from ..utils import compute_mean_std_ci

__all__ = ['draw_actigraphy_data', 'draw_roc_or_pr_curve', 'draw_cv_roc_or_pr_curve', 'draw_cv_boxplot']

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


class HandlerShadeLine(HandlerBase):
    """Legend handle: grey band + dashed mean line."""

    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        # unpack what we supplied in the tuple
        patch_obj, line_obj = orig_handle

        # pull the face-colour & alpha from that patch
        face_col = patch_obj.get_facecolor()
        alpha = patch_obj.get_alpha() or 1.0  # default if None

        band = Rectangle((x0, y0 + .25 * height),  # lower-left corner
                         width, .5 * height,  # full width, half height
                         facecolor=face_col,
                         edgecolor='none',
                         lw=0,
                         alpha=alpha,
                         transform=trans)

        ln = Line2D([x0, x0 + width],
                    [y0 + .5 * height] * 2,
                    color=line_obj.get_color(),
                    lw=line_obj.get_linewidth(),
                    linestyle=line_obj.get_linestyle(),
                    transform=trans)

        return [band, ln]


def draw_actigraphy_data(df: pd.DataFrame, _sleep_log: pd.DataFrame = None, raw_only: bool = False,
                         dark_mode: bool = False, show_sleep_bouts: bool = False):
    """Draw a multi-day actigraphy overview with image-aware, extrema-preserving downsampling.

    The signal density is derived from the horizontal resolution of the output figure.
    Local minima and maxima are retained within temporal bins, preserving short movement
    peaks much more reliably than regular striding or averaging.

    Boolean annotations are rendered as contiguous spans rather than sample-wise polygons,
    substantially reducing memory use for long recordings.
    """

    @dataclass
    class Colors:
        x: str = '#6200EE'
        y: str = '#03DAC6'
        z: str = '#bb86fc'
        sptw: str = 'gray'
        sb: str = 'gray'
        nw: str = 'r'
        dm: str = 'whitesmoke'

    def _downsample_extrema(group: pd.DataFrame, max_vertices: int) -> pd.DataFrame:
        """Retain ordered local minima/maxima from all acceleration axes."""
        n_samples = len(group)
        if n_samples <= max_vertices:
            return group

        # Each bin can contribute at most six extrema: min/max for x, y and z.
        n_bins = max(1, max_vertices // 6)
        edges = np.linspace(0, n_samples, n_bins + 1, dtype=np.int64)
        values = group[['x', 'y', 'z']].to_numpy(copy=False)

        selected = np.empty(n_bins * 6 + 2, dtype=np.int64)
        selected[0] = 0
        cursor = 1

        for start, stop in zip(edges[:-1], edges[1:]):
            if stop <= start:
                continue

            bin_values = values[start:stop]
            selected[cursor:cursor + 3] = start + np.argmin(bin_values, axis=0)
            selected[cursor + 3:cursor + 6] = start + np.argmax(bin_values, axis=0)
            cursor += 6

        selected[cursor] = n_samples - 1
        selected = np.unique(selected[:cursor + 1])

        return group.iloc[selected]

    def _draw_boolean_spans(ax, index: pd.DatetimeIndex, values, color: str, alpha: float) -> None:
        """Shade contiguous True regions without constructing sample-wise polygons."""
        mask = np.asarray(values, dtype=bool)
        if mask.size == 0 or not mask.any():
            return

        transitions = np.diff(mask.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)

        if len(index) > 1:
            sample_delta = index[-1] - index[-2]
        else:
            sample_delta = pd.Timedelta(seconds=1)

        for start, stop in zip(starts, stops):
            start_time = index[start]
            stop_time = index[stop] if stop < len(index) else index[-1] + sample_delta
            ax.axvspan(start_time, stop_time, facecolor=color, edgecolor='none', alpha=alpha, zorder=0)

    max_value = max(df[column].max() for column in ('x', 'y', 'z'))
    min_value = min(df[column].min() for column in ('x', 'y', 'z'))
    maxrange = max(max_value, -min_value)
    minrange = -maxrange

    grouped_days = df.groupby(df.index.date, sort=True)
    nrows = grouped_days.ngroups + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=False, sharey=True, figsize=(10, nrows), dpi=100)
    plt.subplots_adjust(wspace=.1)

    # The axes occupy approximately 75–85% of the figure width after labels and margins.
    # Four vertices per horizontal pixel safely oversamples the raster output while keeping
    # Matplotlib object sizes small. Extrema selection preserves peaks inside every bin.
    usable_width_px = max(1, int(fig.get_figwidth() * fig.dpi * .8))
    max_vertices_per_day = max(2_000, usable_width_px * 4)

    tick_color = Colors.dm if dark_mode else 'k'
    spine_color = Colors.dm if dark_mode else 'k'
    label_color = Colors.dm if dark_mode else 'k'
    grid_major_color = tick_color
    grid_minor_color = 'lightgrey' if dark_mode else 'grey'

    if dark_mode:
        fig.patch.set_facecolor('none')

    lbl_ax = fig.add_subplot(111, frameon=False)
    lbl_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    lbl_ax.grid(False)
    lbl_ax.set_xlabel(r'time / h', fontsize=12, labelpad=15, color=label_color)
    lbl_ax.set_ylabel(r'acceleration magnitude / $g$', fontsize=12, labelpad=20, color=label_color)

    for i, (day, group) in enumerate(grouped_days, start=1):
        ax = axes[i]
        plot_group = _downsample_extrema(group, max_vertices=max_vertices_per_day)

        plot_index = plot_group.index.to_numpy(copy=False)
        x = plot_group['x'].to_numpy(copy=False)
        y = plot_group['y'].to_numpy(copy=False)
        z = plot_group['z'].to_numpy(copy=False)

        ax.plot(plot_index, x, c=Colors.x, zorder=2000, lw=1.4, alpha=1)
        ax.plot(plot_index, y, c=Colors.y, zorder=2000, lw=1, alpha=.95)
        ax.plot(plot_index, z, c=Colors.z, zorder=2000, lw=.6, alpha=.85)

        if not raw_only:
            _draw_boolean_spans(ax, group.index, group['sptw'].to_numpy(copy=False), Colors.sptw, .3)

            if show_sleep_bouts:
                _draw_boolean_spans(ax, group.index, group['sleep_bout'].to_numpy(copy=False), Colors.sb, .5)

            _draw_boolean_spans(ax, group.index, ~group['wear'].to_numpy(copy=False), Colors.nw, .3)

        if isinstance(_sleep_log, pd.DataFrame) and not _sleep_log.empty:
            log_entries = []

            for value in _sleep_log.values.squeeze()[3:]:
                if isinstance(value, datetime) and value.date() == day:
                    log_entries.append(value)

            for log_entry in log_entries:
                if isinstance(log_entry, pd.Timestamp):
                    log_entry = log_entry.to_pydatetime()

                if dark_mode:
                    ax.axvline(log_entry, color='w', zorder=2000, lw=5)
                    ax.axvline(log_entry, color='k', zorder=2100, lw=1, alpha=.8)
                else:
                    ax.axvline(log_entry, color='k', zorder=2000, lw=5)
                    ax.axvline(log_entry, color='w', zorder=2100, lw=1, alpha=.8)

        ax.yaxis.set_label_position('right')
        ax.set_ylabel(day.strftime('%A\n%d %B'), weight='bold', ha='left', va='center', rotation=0,
                      fontsize='medium', color=label_color, labelpad=10)

        ax.get_xaxis().grid(True, which='major', color=grid_major_color, alpha=.75, lw=.75, zorder=100)
        ax.get_xaxis().grid(True, which='minor', color=grid_minor_color, alpha=.25, lw=.75, zorder=100)

        ax.set_xlim(day, day + timedelta(days=1))
        ax.set_ylim(minrange, maxrange)

        day_start = datetime.combine(day, time(0, 0))
        day_end = datetime.combine(day + timedelta(days=1), time(0, 0))
        ax.set_xticks(pd.date_range(start=day_start, end=day_end, freq='4h'))
        ax.set_xticks(pd.date_range(start=day_start, end=day_end, freq='1h'), minor=True)

        ax.tick_params(left=True, right=False, top=False, bottom=True, colors=tick_color, labelcolor=tick_color)
        ax.tick_params(which='major', direction='out', length=5, width=2, colors=tick_color)
        ax.tick_params(which='minor', direction='out', length=3, width=1, colors=tick_color)

        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color(spine_color)
        ax.spines['left'].set_zorder(1000)
        ax.spines['left'].set_linewidth(1.2)

        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color(spine_color)
        ax.spines['bottom'].set_zorder(1000)
        ax.spines['bottom'].set_linewidth(1.2)

        ax.spines['right'].set_color(spine_color)
        ax.spines['top'].set_color(spine_color)
        ax.set_facecolor('none' if dark_mode else 'whitesmoke')

    leg_ax = axes[0]
    leg_ax.axis('off')

    legend_patches = [
        mlines.Line2D([], [], visible=False, label='Legend:'),
        mlines.Line2D([], [], color=Colors.x, lw=2, label=r'$\vec{a}_{x}$'),
        mlines.Line2D([], [], color=Colors.y, lw=2, label=r'$\vec{a}_{y}$'),
        mlines.Line2D([], [], color=Colors.z, lw=2, label=r'$\vec{a}_{z}$'),
    ]

    if isinstance(_sleep_log, pd.DataFrame) and not _sleep_log.empty:
        legend_patches.append(
            mlines.Line2D([], [], color='w' if dark_mode else 'k', lw=5, label='sleep log entry')
        )

    if not raw_only:
        legend_patches.extend([
            mlines.Line2D([], [], color=Colors.sptw, lw=5, alpha=.3, label='sleep window'),
            mlines.Line2D([], [], color=Colors.nw, lw=5, label='non-wear'),
        ])

        if show_sleep_bouts:
            legend_patches.append(
                mlines.Line2D([], [], color=Colors.sb, lw=5, alpha=.5, label='sleep bout')
            )

    leg_ax.legend(
        handles=legend_patches,
        bbox_to_anchor=(0., 0., 1., 1.),
        loc='center',
        ncol=8,
        mode='best',
        borderaxespad=0,
        framealpha=.6,
        frameon=True,
        fancybox=True,
        labelcolor=label_color,
        facecolor='none' if dark_mode else None,
    )

    tick_labels = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']

    fig.autofmt_xdate()
    axes[-1].set_xticklabels(
        tick_labels, fontweight='bold', fontsize='medium', ha='center', color=label_color
    )
    axes[1].set_xticklabels(
        tick_labels, fontweight='bold', fontsize='small', ha='center', color=label_color
    )

    axes[1].xaxis.set_tick_params(pad=-4)
    axes[-1].tick_params(rotation=0, labelcolor=label_color)
    axes[1].tick_params(rotation=0, bottom=True, top=False, labeltop=True, labelcolor=label_color)

    return fig, axes


def draw_roc_or_pr_curve(x: np.ndarray, y: np.ndarray, thres: np.ndarray, mode: dict,
                         opt_thres: Union[float, dict] = None):
    assert 'curve' in mode and mode['curve'] in ['roc', 'pr'], "mode must contain 'curve' with value 'roc' or 'pr'"
    assert 'lvl' in mode and mode['lvl'] in ['night', 'patient'], \
        "mode must contain 'lvl' with value 'night' or 'patient'"
    if mode['curve'] == 'pr':
        assert 'pos_frac' in mode and isinstance(mode['pos_frac'], float), \
            "mode must contain 'pos_frac' as a float when 'curve' is 'pr'"

    _c = 'b' if mode['curve'] == 'roc' else 'deeppink'
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    plt.subplots_adjust(left=.17, right=.97, top=.95, bottom=.14)
    ax.set_title(f"{mode['lvl']} classification", fontweight='bold', pad=7, fontsize=13)

    ax.scatter(x, y, s=20, marker='o', zorder=200, color=_c, alpha=.6, edgecolor='k')

    ax.set_xlim(-.045, 1.045)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=2, top=True, right=True,
                   labelsize=15)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, top=True, right=True)

    if mode['curve'] == 'roc':
        if opt_thres:
            _opt_idx = np.where(opt_thres == thres)
            ax.scatter(x[_opt_idx], y[_opt_idx], s=70, marker='*', zorder=220, color='limegreen', alpha=1.0,
                       edgecolor='k', label=f"op. point={opt_thres:.2f}")

        ax.plot(x, y, lw=1, zorder=200, color=_c, linestyle='-', alpha=1.0,
                label=f"{r'$AUC=$'}{metrics.auc(x, y):.2f}")
        ax.plot([0, 1.], [0, 1], color='k', lw=1, linestyle='--')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=16, labelpad=20)
        ax.set_xlabel('False Positive Rate (1-Specifity)', fontsize=16, labelpad=20)
        ax.legend(loc=(.4, .1), fontsize=15, frameon=False)

    elif mode['curve'] == 'pr':

        f1_scores = np.divide(
            2 * (x * y), x + y,
            out=np.zeros_like(x),  # set default to zero, i.e. when pr. = rec. = 0
            where=(x + y) != 0  # avoid division by zero
        )

        _pos_frac = mode['pos_frac']

        if opt_thres:
            for _name, _opt_thresh in opt_thres.items():
                _opt_idx = np.where(_opt_thresh == thres)
                ax.scatter(x[_opt_idx], y[_opt_idx], s=70, marker='*', zorder=220, color='limegreen', alpha=1.0,
                           edgecolor='k', label=f"op. point ={_opt_thresh:.2f} ({_name})")

        ax.plot(x, y, lw=1, zorder=200, color=_c, linestyle='-', alpha=1.0,
                label=f"{r'$f_{1}^{ max}=$'}{np.nanmax(f1_scores):.2f}")
        ax.plot([0, 1.], [_pos_frac, _pos_frac], color='k', lw=1, linestyle='--', label=f'pos. frac={_pos_frac:.2f}')
        ax.set_ylabel('Precision', fontsize=16, labelpad=20)
        ax.set_xlabel('Recall', fontsize=16, labelpad=20)
        ax.set_ylim(_pos_frac - .045, 1.045)
        ax.legend(loc=(.4, .75), fontsize=15, frameon=False)

    return fig


def draw_cv_roc_or_pr_curve(
        curve_dict: dict, mode: str, cl: float = .95,
        *, display_op_point: bool = False,
        lodo_labels: dict = None, strip_axis: bool = False, lw: float = 1.7) -> plt.Figure:
    if mode == 'roc':
        x, y = np.array(curve_dict['fpr']), np.array(curve_dict['tpr'])
        metric, op_point = np.array(curve_dict['auc']), np.array(curve_dict['op_point'])
    elif mode == 'pr':
        x, y = np.array(curve_dict['rec']), np.array(curve_dict['prec'])
        metric = np.array(curve_dict['f1_max'])
        op_point = {'dist': np.array(curve_dict['op_point_dist']), 'f1': np.array(curve_dict['op_point_f1'])}
    else:
        raise ValueError(f"mode bust be 'roc' or 'pr', got {mode}.")

    mean_y, std_y = np.mean(y, axis=0), np.std(y, axis=0)
    _cl_std = stats.norm.ppf((1 + cl) / 2)
    y_upper = np.minimum(mean_y + _cl_std * std_y, 1)
    y_lower = np.maximum(mean_y - _cl_std * std_y, 0)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    fig.set_facecolor('none')
    if not strip_axis:
        plt.subplots_adjust(left=.17, right=.97, top=.95, bottom=.14)
    else:
        plt.subplots_adjust(left=.07, right=.98, top=.99, bottom=.04)
    ax.set_xlim(-.045, 1.045)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=2, top=True, right=True, labelsize=15)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, top=True, right=True)

    mean, std, ci_lower, ci_upper, _ = compute_mean_std_ci(metric, confidence_level=cl)

    _c = 'b' if mode == 'roc' else 'deeppink'
    # _label_line = f"{r'$AUC=$'}{np.mean(metric):.2f}±{np.std(metric):.2f}" if mode == 'roc' \
    #    else f"{r'$f1_{max}=$'}{np.mean(metric):.2f}±{np.std(metric):.2f}"

    _metric_name = 'f1_{max}' if mode == 'pr' else 'AUC'
    _line_label = rf"${_metric_name} = {mean:.2f}^{{\,+{ci_upper - mean:.2f}}}_{{\,-{mean - ci_lower:.2f}}}$"

    is_lodo = lodo_labels is not None
    if not is_lodo:
        lodo_labels = [None] * len(y)  # dummy for zip
    else:
        LINESTYLES = ['-', '--', '-.', ':']
        STYLE_CYCLER = cycle(LINESTYLES)
        style_by_dataset = {}

    custom_handles, custom_labels = [], []
    sorted_entries = sorted(zip(metric, y, lodo_labels), key=lambda t: t[0], reverse=True)

    for val, _y, ds_specs in sorted_entries:
        if lodo_labels and isinstance(ds_specs, tuple):
            ds_name, ds_color = ds_specs
            if ds_name not in style_by_dataset:
                style_by_dataset[ds_name] = next(STYLE_CYCLER)
            ds_style = style_by_dataset[ds_name]
            # label = f"{ds_name} \n(AUC = {val:.2f})" if mode == 'roc' else f"{ds_name} (F1 = {val:.2f})"
            ax.plot(x, _y, lw=lw, zorder=400, color=ds_color, ls=ds_style, alpha=.85)  # , label=label)
            line = Line2D([0], [0], color=ds_color, lw=3, linestyle=ds_style)

            ds_label = f"{ds_name}"  # left-aligned name
            auc_str = f"{val:.2f}"  # right-aligned number
            label = f"{ds_label} ({auc_str})"
            custom_labels.append(label)
            custom_handles.append(line)

        else:
            ax.plot(x, _y, lw=.7, zorder=100, color='grey', linestyle='-', alpha=.35)

    if is_lodo:
        _color = 'k'
        mean_line, = ax.plot(x, mean_y, lw=lw - 1, zorder=100, color=_color, linestyle='-', alpha=.7, label=_line_label)
        ax.fill_between(x, y_lower, y_upper, zorder=150, color=_color, alpha=.1)
        # label=rf"$ {cl * 100:.0f}\% \,CI\,(\sigma={std:.2f})$")
        # custom_handles.append(Line2D([0], [0], color='gray', lw=2, linestyle='--'))
        ci_patch = Patch(facecolor=_color, alpha=.2)
        custom_handles.append((ci_patch, mean_line))  # a *tuple*
        custom_labels.append(
            rf"{'mean'} (${mean:.2f}^{{\,+{ci_upper - mean:.2f}}}_{{\,-{mean - ci_lower:.2f}}})$")  #, \sigma={std:.2f})$")

    else:
        ax.plot(x, mean_y, lw=1, zorder=200, color=_c, linestyle='-', alpha=1.0, label=_line_label)
        ax.fill_between(x, y_lower, y_upper, zorder=150, color='grey', alpha=.2,
                        label=rf"$ {cl * 100:.0f}\% \,CI\,(\sigma={std:.2f})$")
    if display_op_point:
        _label_scatter = f"op. point={np.mean(op_point):.2f}±{np.std(op_point):.2f}" if mode == 'roc' \
            else (f"op. point={np.mean(op_point['dist']):.2f}±{np.std(op_point['dist']):.2f} "
                  f"({np.mean(op_point['f1']):.2f}±{np.std(op_point['f1']):.2f} f1)")
        ax.scatter(x, mean_y, s=20, marker='o', zorder=200, color=_c, alpha=.6, edgecolor='k', label=_label_scatter)

    if mode == 'roc':
        ax.set_ylim(-.045, 1.045)
        mean_y[-1] = 1.0
        ax.plot([0, 1.], [0, 1], color='k', lw=1, linestyle='--')
        if is_lodo:
            FIG = '\u2007'
            title_str = f"{FIG} Dataset (AUC)"

            legend = ax.legend(
                handles=custom_handles,
                labels=custom_labels,
                loc='lower right',
                fontsize=18,
                handlelength=1.8,
                borderaxespad=.4,
                frameon=False,
                handler_map={tuple: HandlerShadeLine()},
                title=title_str,
                title_fontsize=16,
                labelspacing=.5
            )

            legend.get_title().set_ha('left')
        else:
            ax.legend(loc=(.4, .1), fontsize=15, frameon=False)
        if not strip_axis:
            ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=16, labelpad=20)
            ax.set_xlabel('False Positive Rate (1-Specifity)', fontsize=16, labelpad=20)

    elif mode == 'pr':
        _mean_pos_frac = np.min(mean_y)
        ax.set_ylim(.5, 1.02)
        ax.legend(loc=(.05, .05), fontsize=15, frameon=False)
        ax.set_ylabel('Precision', fontsize=16, labelpad=20)
        ax.set_xlabel('Recall', fontsize=16, labelpad=20)
        ax.plot([0, 1], [_mean_pos_frac, _mean_pos_frac], color='k', lw=1, linestyle='--',
                label=f"{_mean_pos_frac:.2f} pos. frac.")

    return fig


def draw_cv_boxplot(history: dict, ylims=(.5, 1), cl: float = .90):
    history = {k: v for k, v in history.items() if k in (
        'accuracy', 'balanced_accuracy', 'roc_auc', 'f1', 'recall', 'precision', 'average_precision')}

    names, data = [], []
    for name, _data in history.items():
        if name == 'balanced_accuracy':
            name = 'bal. accuracy'
        elif name == 'average_precision':
            name = 'avg. precision'
        elif name == 'roc_auc':
            name = 'auroc'

        if name not in ('class_thresh', 'mean', 'std'):
            names.append(name)
            data.append(_data.flatten())

    def _pick_colors_from_colormap(colormap_name, k):
        cmap = plt.get_cmap(colormap_name)
        return np.array([cmap(j / (k - 1)) for j in range(k)])

    colors = _pick_colors_from_colormap('Greens', len(names))
    medians = np.median(data, axis=1)
    sorted_indices = np.argsort(medians)
    sorted_colors = np.empty(len(names), dtype=object)
    for i, index in enumerate(sorted_indices):
        sorted_colors[index] = colors[i]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.set_facecolor('none')
    plt.subplots_adjust(left=.04, right=.99, top=.98, bottom=.06)

    bplot = ax.boxplot(data, labels=names, notch=True, patch_artist=True, sym='k+', zorder=200)
    for patch, color in zip(bplot['boxes'], sorted_colors):
        patch.set_facecolor(color)
        patch.set_alpha(.82)
    median_color = 'k'
    for median in bplot['medians']:
        median.set_color(median_color)
        median.set_linewidth(1.5)

    _y_offset = .05
    for i, (_name, _data) in enumerate(zip(names, data)):
        lower_whisker = bplot['whiskers'][2 * i].get_ydata()[1]
        mean, std, ci_lower, ci_upper, _ = compute_mean_std_ci(_data, confidence_level=cl)
        _label = '\n'.join(names[i].split()) + '\n' \
                 + r'$' + f'{mean:.2f}^{{+{ci_upper - mean:.2f}}}_{{-{mean - ci_lower:.2f}}}' + r'$' \
                 + '\n' + r'$\sigma = ' + f'{std:.2f}' + r'$'
        ax.text(i + 1, lower_whisker - _y_offset, _label, ha='center', va='top', fontsize=15, color='k')

    ax.tick_params(axis='both', which='major', direction='in', length=6, width=2, top=True, right=True,
                   labelsize=12)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, top=True, right=True)

    ax.grid(axis='y', which='major', c='grey', alpha=.4, lw=.5)
    # ax.set_ylim(max(0, np.min(data)-.2), 1)
    ax.set_ylim(*ylims)
    ax.set_xlim(.25, len(names) + .75)
    ax.set_xticklabels([])
    # ax.set_yticks(np.linspace(.5, 1.0, 6))
    return fig
