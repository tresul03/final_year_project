import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        # Initialize the plotter with an empty figure and axis
        self.fig, self.ax = plt.subplots()

    def plot_line(self, x, y, label=None, color=None):
        # Plot a line on the existing axis using provided x and y data
        self.ax.plot(x, y, label=label, color=color)

    def plot_group(self, x, y,test_inputs, test_output):

        i = 0

        while i <24:

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            ax1.plot(x,test_output[i,0,:],color='black',lw=3,label=f'SKIRT Model: input = {test_inputs[i,0]:2.2f}, {test_inputs[i,1]:2.2f}, {test_inputs[i,2]:2.2f}')
            ax1.plot(x,y[i,:],color='darkorange',lw=3,label='Predicted Mean Model')
            ax2.plot(x,test_output[i+1,0,:],color='black',lw=3,label=f'SKIRT Model: input = {test_inputs[i+1,0]:2.2f}, {test_inputs[i+1,1]:2.2f}, {test_inputs[i+1,2]:2.2f}')
            ax2.plot(x,y[i+1,:],color='darkorange',lw=3,label='Predicted Mean Model')
            ax3.plot(x,test_output[i+2,0,:],color='black',lw=3,label=f'SKIRT Model: input = {test_inputs[i+2,0]:2.2f}, {test_inputs[i+2,1]:2.2f}, {test_inputs[i+2,2]:2.2f}')
            ax3.plot(x,y[i+2,:],color='darkorange',lw=3,label='Predicted Mean Model')
            ax4.plot(x,test_output[i+3,0,:],color='black',lw=3,label=f'SKIRT Model: input = {test_inputs[i+3,0]:2.2f}, {test_inputs[i+3,1]:2.2f}, {test_inputs[i+3,2]:2.2f}')
            ax4.plot(x,y[i+3,:],color='darkorange',lw=3,label='Predicted Mean Model')


            for ax in fig.get_axes():
                #ax.label_outer()
                ax.set(xlabel=f'Wavelength($\\mu$m)', ylabel='Sersic Index (Normalised)')
                ax.set_xscale('log')
                ax.legend()


            plt.tight_layout()

            ax.legend()
            plt.show()
            plt.close()
            
            
            i = i+4

                