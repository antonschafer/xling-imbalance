import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class PowerLaw:
    def __init__(self, x_data, y_data, reversed=False):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)

        if reversed:
            # Fit y -> x
            self.params, self.pcov = curve_fit(self._x, self.y_transform(self.y_data), self.x_data, p0=[-0.1, 0.6, 2])
        else:
            self.params, self.pcov = curve_fit(self._y, self.x_data, self.y_transform(self.y_data))
        
    def y_transform(self, y):
        # HACK: This ensures fairer weighing of errors TODO more principled approach
        return 1/y 

    def y_transform_inv(self, y_t):
        # HACK: This ensures fairer weighing of errors TODO more principled approach
        return 1/y_t
    
    @staticmethod
    def _y(x, a, b, c):
        return -a * x**(-b) + c
    
    @staticmethod
    def _x(y, a, b, c):
        return ((y - c) / (-a))**(-1/b)

    def y(self, x):
        y = self._y(x, *self.params)
        return self.y_transform_inv(y)
    
    def x(self, y):
        y = self.y_transform(y)
        return self._x(y, *self.params)
    
    def plot(self, ax, c1="blue", c2="red", label="Fitted Curve", set_grid=True, x_left=None, label_datapoints='Data Points', markersize=None, x_factor=1.0, **kwargs):
        if x_left is None:
            x_left = min(self.x_data)
        x_line = np.linspace(x_left, max(self.x_data), 100)
        y_line = self.y(x_line)

        mask = self.x_data>=x_left
        ax.plot(x_factor * x_line, y_line, color=c2, label=label, **kwargs)
        ax.scatter(x_factor * self.x_data[mask], self.y_data[mask], color=c1, label=label_datapoints, marker="x", s=None if markersize is None else [markersize for _ in self.x_data[mask]])
        ax.legend()

        if set_grid:
            ax.grid(True, linestyle='--', color='grey', linewidth=0.5, alpha=0.6)
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    
    def relative_errors(self):
        y_errors = (self.y(self.x_data) - self.y_data) / self.y_data
        x_errors = (self.x(self.y_data) - self.x_data) / self.x_data
        return x_errors, y_errors
    
    def show(self):
        # summarize
        x_errors, y_errors = self.relative_errors()
        print(f"Params: {self.params} (a, b, c)")
        print(f"Mean abs relative error in x: {np.mean(np.abs(x_errors)):.4f} (max: {max(x_errors):.4f}, min: {min(x_errors):.4f})")
        print(f"Mean abs relative error in y: {np.mean(np.abs(y_errors)):.4f} (max: {max(y_errors):.4f}, min: {min(y_errors):.4f})")

        # plot
        ax = plt.gca()
        self.plot(ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Power Law Fit')
        plt.show()

        # details
        print("Data points")
        for x, y in zip(self.x_data, self.y_data):
            print(f"\tData: {x:.4f} -> {y:.4f}, Pred x->y: {self.y(x):.4f} (rel error {(self.y(x) - y) / y:.4f}), Pred y->x: {self.x(y):.4f} (rel error {(self.x(y) - x) / x:.4f})")


def bin_plot(x, bin_resolution=4, binsizing="log", lim_left=None):
    """
    Prepares bin plot.

    Returns to_bins, xticks, xlabels.

    To plot, run
        ax.plot(*to_bins(x, y))
        ax.set_xticks(xticks, labels=xlabels)
    """

class BinPlot:
    def __init__(self, x_all, bin_resolution=4, binsizing="log", lim_left=None):

        # compute bins that cover all x_all data
        if binsizing == "log":
            assert x_all.min() > 0
            lim_left = 0 if lim_left is None else math.floor(np.log10(lim_left))
            lim_right = math.ceil(np.log10(max(x_all)))
            exps = np.linspace(lim_left, lim_right, (lim_right - lim_left) * bin_resolution + 1)
            self.bins = np.power(10, exps)
        elif binsizing == "linear":
            lim_left = min(x_all) if lim_left is None else lim_left
            lim_right = max(x_all) 
            self.bins = np.linspace(lim_left, lim_right, math.ceil(lim_right - lim_left) * bin_resolution + 1)
        elif binsizing.startswith("steps"):
            step_size = float(binsizing.split("_")[1])
            lim_left = min(x_all) if lim_left is None else lim_left
            lim_right = max(x_all) 
            # round
            lim_left, lim_right = step_size * math.floor(lim_left / step_size), step_size * math.ceil(lim_right / step_size)
            n_steps = (lim_right - lim_left) / step_size + 1
            self.bins = np.linspace(lim_left, lim_right, int(round(n_steps)))
        else:
            raise NotImplementedError()

        self.xticks = range(len(self.bins))

        # make x labels
        if binsizing == "log":
            infty = "-\\infty"
            xlabels = [
                f"$[10^{{{exps[i-1] if i > 0 else infty}}}, 10^{{{exps[i]:.0f}}})$"
                if self.bins[i] % 1 == 0
                else ""
                for i in self.xticks
            ]
            xlabels = [f"$[0, 10^{{{exps[0]:.0f}}})$"] + xlabels[1:]
        else:
            xlabels = [
                f"[{self.bins[i-1]:.2f}, {self.bins[i]:.2f})"
                for i in self.xticks[1:]
            ]
            assert self.bins[0] >= 0 and min(x_all) >= 0, "not implemented for negative values"
            if self.bins[0] == 0:
                xlabels = [""] + xlabels
            else:
                xlabels = [f"$[0, {self.bins[0]:.2f})$"] + xlabels
        self.xlabels = xlabels

    def to_bins(self, x, y, min_count=1, return_support=False):
        """
        Bin data by x values and compute average y values per bin.

        Args:
            x: x values
            y: y values
            min_count: minimum number of elements in a bin (result considered NaN if less values present)
            return_support: whether to return support (number of elements in each bin)
        
        Returns:
            xticks, bin averages (, support if return_support=True)
        """
        bin_indices = np.digitize(x, self.bins)
        bin_averages = [y[bin_indices == i].mean() if np.sum(bin_indices == i) >= min_count else float("nan") for i in self.xticks]

        if return_support:
            return self.xticks, bin_averages, np.array([np.sum(bin_indices == i) for i in self.xticks])
        else:
            return self.xticks, bin_averages

    def get_ticks_labels(self):
        return self.xticks, self.xlabels

    @staticmethod
    def adjust_tick_lengths(ax, l_long=5, l_short=2):
        """
        Make the tick lengths long for ticks that have a label and short for those that don't.
        """
        # Set the tick parameters for the x-axis
        for label, tick in zip(ax.get_xticklabels(), ax.xaxis.get_major_ticks()):
            if label.get_text():  # Check if the tick has a label
                tick.tick1line.set_markersize(l_long)  # Adjust the tick length
            else:
                tick.tick1line.set_markersize(l_short)
