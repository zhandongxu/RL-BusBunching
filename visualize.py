import matplotlib.pyplot as plt
from itertools import accumulate
from collections import defaultdict

fig_loc = r"D:\bus_bunching\output_fig\ "

plt.rcParams['font.family'] = 'Latin Modern Roman'


loc_color = '#FBF4F9'
traje_color = '#92C2DD'
hold_color = '#E49393'


# loc_color = '#FBF4F9'
# traje_color = '#433520'
# hold_color = '#00917C'


# plot from 0 to plot_horiz seconds
plot_horiz = 3600.0 * 2.0
norm_time = 1.0  # in seconds


def plot_time_space_diagram(buses, config):
    _, ax = plt.subplots()

    stop_xs = list(accumulate(config.link_length))
    stop_xs.insert(0, 0)
    stop_xs.pop()

    # set the right and top axis to be invisible
    ax.spines[['right', 'top']].set_visible(False)

    ax.set_xlabel('Time/s', fontsize=12)
    ax.set_ylabel('Milepost/km', fontsize=12)

    # plot scheduled times
    # stop_sched_times = config.ln_info['stop_sched_times']
    # for stop, sched_times in stop_sched_times.items():
    #     x = stop_xs[stop] / 1000.0
    #     for sched_time in sched_times:
    #         if sched_time <= plot_horiz:
    #             ax.vlines(x=sched_time, ymin=x-0.1, ymax=x +
    #                       0.1, linewidth=1.0, color='r')

    # plotting horizontal stop location lines
    # for x in stop_xs:
    #     ax.axhline(y=x/1000.0, color=loc_color, linestyle='-.',
    #                dashes=(5, 2), linewidth=1.0)
    for bus in buses:
        # plot trajectory
        trip_xs = defaultdict(list)
        trip_ys = defaultdict(list)
        for t, point in bus.traje.items():
            if t <= plot_horiz:
                trip_no = point['trip_no']
                trip_xs[trip_no].append(t/norm_time)
                trip_ys[trip_no].append(point["relat_x"] / 1000.0)
            else:
                break
        for trip, xs in trip_xs.items():
            ys = trip_ys[trip]
            ax.plot(xs, ys, color=traje_color, linewidth=1.5)

        # plot holding times
        hold_xs = defaultdict(list)
        hold_ys = {}
        for t, point in bus.traje.items():
            if t <= plot_horiz:
                trip_no = point['trip_no']
                if point['spot_type'] == 'hold':
                    hold_xs[(trip_no, point['spot_id'])].append(t/norm_time)
                    hold_ys[(trip_no, point['spot_id'])
                            ] = point['relat_x'] / 1000.0
            else:
                break

        for (trip_no, link_id), xs in hold_xs.items():
            start, end = min(xs), max(xs)
            y = hold_ys[(trip_no, link_id)]
            ax.hlines(y=y, xmin=start, xmax=end,
                      color=hold_color, linewidth=2.5)

    plt.savefig(fig_loc + 'time_space_diagram.png',
                dpi=400, bbox_inches='tight')
    # plt.show()
