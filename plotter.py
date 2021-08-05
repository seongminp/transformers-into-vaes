import itertools

import matplotlib.pyplot as plt

CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"
color_list = [
    CB91_Blue,
    CB91_Pink,
    CB91_Green,
    CB91_Amber,
    CB91_Purple,
    CB91_Violet,
    "#82978C",
    "#32415B",
    "#93869F",
]

import matplotlib.font_manager

fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
print([f for f in fonts if "Roman" in f])
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=color_list)
plt.rcParams["font.family"] = "Times New Roman"

markers = [".", "^", "*", "2", "+", ",", "s", "h"]


def cleanup(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        labelbottom="on",
        left="off",
        right="off",
        labelleft="on",
    )


def plot_line(
    title, x_label, y_label, x_data, y_data, save_dir=None, use_markers=False
):
    fig, ax = plt.subplots()

    # x_axis = np.arange(timings[0][1].shape[0])
    for li, (label, y) in enumerate(y_data):
        marker = markers[li] if use_markers else None
        ax.plot(x_data, y, label=label, marker=marker, markersize=4)

    ax.legend(frameon=False)
    ax.set_title(title.title(), fontweight="bold", pad=30)
    ax.set_xlabel(x_label, style="italic")
    ax.set_ylabel(y_label, style="italic")

    # Cleanup.
    cleanup(ax)

    if save_dir is not None:
        fig_name = f"{title}.png"
        fig.tight_layout()
        fig.savefig(save_dir + fig_name)
    return ax


def plot_multiple_line(
    title,
    sub_titles,
    x_labels,
    y_labels,
    all_x_data,
    all_y_data,
    save_dir="./",
    use_markers=False,
):
    fig, axs = plt.subplots(len(x_labels), 1, figsize=(8, 10))
    # fig, axs = plt.subplots(1, len(x_labels), figsize=(10, 4))
    for i in range(len(x_labels)):
        plt.gca().set_prop_cycle(None)

        sub_title = sub_titles[i]
        x_label = x_labels[i]
        y_label = y_labels[i]
        x_data = all_x_data[i]
        y_data = all_y_data[i]

        for li, (label, y) in enumerate(y_data):
            marker = markers[li] if use_markers else None
            axs[i].plot(
                x_data, y, label=label, marker=marker, c=color_list[li], markersize=15
            )

        # axs[i].legend(frameon=False)
        axs[i].set_title(sub_title, fontweight="bold", fontsize=30, pad=15)
        axs[i].set_xlabel(x_label, style="italic", fontsize=25)
        axs[i].set_ylabel(y_label, style="italic", fontsize=25)

        # Cleanup.
        # cleanup(axs[i])

    handles, labels = axs[-1].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        # bbox_to_anchor=(0.5, -0.2),
        # loc="lower center",
        bbox_to_anchor=(1.4, 0.5),
        # loc="center left",
        loc="right",
        ncol=1,
        frameon=False,
        fontsize=20,
    )

    fig.tight_layout()
    # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)

    fig_name = f"{title}.png"
    fig.savefig(save_dir + fig_name, bbox_inches="tight")
    print(f"Image saved at {save_dir + fig_name}")
