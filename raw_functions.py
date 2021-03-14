import numpy as np
import matplotlib.pyplot as plt


def parse_responses(fname):
    y = []
    with open(fname) as f:
        f.readline()
        for line in f:
            if ',' not in line:
                continue
            else:
                data = float(line.split(',')[1].rstrip())
                y.append(data)
    return y


def parse_timecourse(fname):
    y = []
    with open(fname) as f:
        f.readline()
        for line in f:
            if ',' not in line:
                continue
            else:
                y.append(float(line.split(',')[1]))
    return y


def metadata(x):
    mean = np.mean(x)
    std = np.std(x)
    return mean, std


def plot_std(x, y):
    plt.title = 'Timecourse'
    plt.plot(x[:80], y[:80])
    plt.show()


def run_timecourse(fname):
    x, y = parse_timecourse(fname)
    plot_std(x, y)


def run_response(fname):
    y = parse_responses(fname)
    plt.set_title = 'Response'
    plt.boxplot(range(0, len(y) - 1), y[:80])
    plt.show()


def response_latency(data, ratio=0.3):
    res = {'Response Latency': [], 'Peak latency': []}

    baseline = np.array(data[:31])
    mean_baseline = np.mean(baseline)
    print(f'Mean baseline: {mean_baseline}')

    response_region = np.array(data[:])
    peak = np.argmax(response_region)
    max_peak = np.max(response_region)
    print(f'Peak max: {max_peak}')
    res['Peak latency'] = (peak, response_region[peak])

    # Compute response latency
    for index, value in enumerate(data[31:50], 31):
        if value > abs((1 + ratio) * mean_baseline):
            res['Response Latency'].append((index, value))
    return res['Response Latency'][0], res['Peak latency']


def run_latency(fname):
    data = parse_responses(fname)
    response, peak = response_latency(data)
    print(f'Response latency at {response[0]} frame with value {response[1]}')
    print(f'Peak latency at {peak[0]} frame with value {peak[1]}')
