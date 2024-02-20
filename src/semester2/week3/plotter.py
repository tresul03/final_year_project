import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        # Initialize the plotter with an empty figure and axis
        self.fig, self.ax = plt.subplots()

    def plot_line(self, x, y, label=None, color=None):
        # Plot a line on the existing axis
        self.ax.plot(x, y, label=label, color=color)

    def scatter_plot(self, x, y, label=None, color=None, marker='o'):
        # Create a scatter plot on the existing axis
        self.ax.scatter(x, y, label=label, color=color, marker=marker)

    def set_title(self, title):
        # Set the title of the plot
        self.ax.set_title(title)

    def set_xlabel(self, xlabel):
        # Set the label for the x-axis
        self.ax.set_xlabel(xlabel)

    def set_ylabel(self, ylabel):
        # Set the label for the y-axis
        self.ax.set_ylabel(ylabel)

    def show_plot(self):
        # Display the plot with legend
        self.ax.legend()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Create an instance of the Plotter class
    plotter = Plotter()

    # Example data
    x_data = [1, 2, 3, 4, 5]
    y_data = [2, 4, 6, 8, 10]

    # Plot a line
    plotter.plot_line(x_data, y_data, label='Line Plot', color='blue')

    # Plot a scatter plot
    plotter.scatter_plot(x_data, y_data, label='Scatter Plot', color='red', marker='o')

    # Set plot title and axis labels
    plotter.set_title('Example Plot')
    plotter.set_xlabel('X-axis')
    plotter.set_ylabel('Y-axis')

    # Show the plot
    plotter.show_plot()
