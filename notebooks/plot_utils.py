import matplotlib.pyplot as plt


def plot_setting(font_size=12):

    # plt.rcParams["text.usetex"] = True
    plt.rcParams[
        "figure.dpi"
    ] = 300  # Dots per inch, higher resolution for better print quality

    # Fonts and text
    plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"]  # Use a common serif font
    plt.rcParams["font.size"] = font_size  # Set the base font size
    plt.rcParams["axes.labelsize"] = 14  # Size of axis labels
    plt.rcParams["xtick.labelsize"] = font_size  # Size of x-axis tick labels
    plt.rcParams["ytick.labelsize"] = font_size  # Size of y-axis tick labels
    plt.rcParams["legend.fontsize"] = font_size  # Size of legend text

    # Line styles and markers
    plt.rcParams["lines.linewidth"] = 2  # Set the default line width
    plt.rcParams["lines.markersize"] = 8  # Set the default marker size

    # Grid and background
    plt.rcParams["axes.grid"] = True  # Show grid by default
    plt.rcParams[
        "axes.grid.which"
    ] = "both"  # Show grid lines on both major and minor ticks
    plt.rcParams["grid.linestyle"] = "--"  # Set grid line style
    plt.rcParams["grid.alpha"] = 0.7  # Grid transparency
