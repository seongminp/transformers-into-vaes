import argparse
from pathlib import Path

import h5py
from plotter import plot_line, plot_multiple_line
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def collect(out_file, experiment_dir, metrics):
    event_acc = EventAccumulator(experiment_dir)
    event_acc.Reload()

    for metric in metrics:
        values = [s.value for s in event_acc.Scalars(metric)]
        xticks = [s.step for s in event_acc.Scalars(metric)]
        yield metric, xticks, values


def tidy_name(metric):
    if metric == "val_mi":
        return "Validation MI"
    elif metric == "val_au":
        return "Validation AU"
    elif metric == "train_unweighted_reg_loss":
        return "KL Loss During Training"
    elif metric == "val_ppl":
        return "Validation PPL"
    return "FIXFIXFIXFIXFIX"


def tidy_ylabel(metric):
    if metric == "val_mi":
        return "MI"
    elif metric == "val_au":
        return "AU"
    elif metric == "train_unweighted_reg_loss":
        return "KL"
    elif metric == "val_ppl":
        return "Validation PPL"
    return "FIXFIXFIXFIXFIX"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log-dir", help="Log directory")
    parser.add_argument("-o", "--out-file", help="Output stats directory", default=None)
    parser.add_argument("-m", "--metrics", help="Comma separated metrics to collect")
    parser.add_argument("-d", "--data", help="Already collected data", default=None)
    args = parser.parse_args()

    metrics = args.metrics.split(",")

    if not args.data:

        if not args.out_file:
            raise Exception("Must provide either args.data or args.out_file.")

        out_file = h5py.File(args.out_file, "w")

        for experiment_dir in sorted(Path(args.log_dir).glob("*/")):

            if experiment_dir.is_file():
                continue

            group = out_file.create_group(experiment_dir.name)
            for metric, xticks, values in collect(out_file, experiment_dir, metrics):
                group.create_dataset(metric, data=values)
                group.create_dataset(metric + "_xticks", data=xticks)

        out_file.close()

    # Plot.
    data_file = args.data if args.data else args.out_file
    data = h5py.File(data_file, "r")
    x_labels, y_labels, all_x_data, all_y_data = [], [], [], []
    for metric in metrics:
        x_data, y_data = None, []
        for group in data:
            if x_data is None:
                x_data = data[group][metric + "_xticks"][()]
            y_data.append((group, data[group][metric][()]))

        all_x_data.append(x_data)
        all_y_data.append(y_data)

    plot_multiple_line(
        "phase2",
        [tidy_name(m) for m in metrics],
        ["Steps"] * len(metrics),
        [tidy_ylabel(m) for m in metrics],
        all_x_data,
        all_y_data,
        save_dir="encoder_images/",
        use_markers=True,
    )
