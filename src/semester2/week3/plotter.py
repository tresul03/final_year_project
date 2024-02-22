import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, x, y,test_inputs, test_output):
        # # Initialize the plotter with an empty figure and axis
        # self.fig, self.ax = plt.subplots()
        self.y = y
        self.x = x
        self.test_inputs = test_inputs
        self.test_output = test_output
        self.figure = None 

        #need to add whether it is plotting n,f or r



    def plot_group_same(self, i:int =0, j:int = 16, color1:str = 'blue', color2:str = 'red'):

        while i < j:


            

            fig_group, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            ax1.plot(self.x,self.test_output[i,:],color= color1,lw=3,label=f'SKIRT Model: input = {self.test_inputs[i,0]:2.2f}, {self.test_inputs[i,1]:2.2f}, {self.test_inputs[i,2]:2.2f}')
            ax1.plot(self.x,self.y[i,:],color= color2,lw=3,label='Predicted Mean Model')
            ax2.plot(self.x,self.test_output[i+1,:],color= color1,lw=3,label=f'SKIRT Model: input = {self.test_inputs[i+1,0]:2.2f}, {self.test_inputs[i+1,1]:2.2f}, {self.test_inputs[i+1,2]:2.2f}')
            ax2.plot(self.x,self.y[i+1,:],color= color2,lw=3,label='Predicted Mean Model')
            ax3.plot(self.x,self.test_output[i+2,:],color= color1,lw=3,label=f'SKIRT Model: input = {self.test_inputs[i+2,0]:2.2f}, {self.test_inputs[i+2,1]:2.2f}, {self.test_inputs[i+2,2]:2.2f}')
            ax3.plot(self.x,self.y[i+2,:],color= color2,lw=3,label='Predicted Mean Model')
            ax4.plot(self.x,self.test_output[i+3,:],color= color1,lw=3,label=f'SKIRT Model: input = {self.test_inputs[i+3,0]:2.2f}, {self.test_inputs[i+3,1]:2.2f}, {self.test_inputs[i+3,2]:2.2f}')
            ax4.plot(self.x,self.y[i+3,:],color= color2,lw=3,label='Predicted Mean Model')

        

            for ax in fig_group.get_axes():
                #ax.label_outer()
                ax.set(xlabel=f'Wavelength($\\mu$m)', ylabel='Sersic Index (Normalised)')
                ax.set_xscale('log')
                ax.legend()
                
            fig_group.suptitle(f'Comparison of SKIRT Model and Predicted Mean Model for {i} to {i+3} input values')
            

            plt.tight_layout()


            i = i+4
            
        plt.show()
        plt.close()



    def plot_single(self, i:int =0, color1:str = 'blue', color2:str = 'red'):

        self.figure = plt.figure(figsize=(15, 5))

        plt.plot(self.x,self.test_output[i,:],color= color1,lw=3,label=f'SKIRT Model: input = {self.test_inputs[i,0]:2.2f}, {self.test_inputs[i,1]:2.2f}, {self.test_inputs[i,2]:2.2f}')
        plt.plot(self.x,self.y[i,:],color= color2,lw=3,label='Predicted Mean Model')

        plt.xlabel(f'Wavelength($\\mu$m)')
        plt.ylabel('Sersic Index (Normalised)')
        plt.xscale('log')
        plt.legend()

        plt.suptitle(f'Comparison of SKIRT Model and Predicted Mean Model for {i} input value')

        plt.tight_layout()

        plt.legend()
            
        plt.show()
        plt.close()


    def plot_same_ax(self, i:int = 0, j:int = 5, step_size:int = 1):


        self.figure, ax = plt.subplots(figsize=(15, 5))

        q = i

        while i < j:

            color = plt.cm.autumn(((i-q)+1) / (j-q))

            ax.plot(self.x,self.y[i,:],color= color,lw=3)

            i += step_size

        plt.xlabel(f'Wavelength($\\mu$m)')
        plt.ylabel('Sersic Index (Normalised)')
        plt.xscale('log')

        plt.suptitle(f'Predicted Mean Model for {q} to {j}')

        plt.tight_layout()

        plt.show()
        plt.close()


    def save_figure(self, filename):
        if self.figure is not None:
            self.figure.savefig(filename)
            print(f"Figure saved as {filename}")
        else:
            print("No figure to save. Create a plot first.")


    # def one_var_plot(self, i:int =0, color1:str = 'blue', color2:str = 'red'):

    #     ax = plt.subplots()

    #     ax.plot(self.x,self.test_output[i,0,:],color= color1,lw=3,label=f'SKIRT Model: input = {self.test_inputs[i,0]:2.2f}, {self.test_inputs[i,1]:2.2f}, {self.test_inputs[i,2]:2.2f}')
    #     ax.plot(self.x,self.y[i,:],color= color2,lw=3,label='Predicted Mean Model')

      
    #     ax.set(xlabel=f'Wavelength($\\mu$m)', ylabel='Sersic Index (Normalised)')
    #     ax.set_xscale('log')
    #     ax.legend()

    #     plt.title(f'Comparison of SKIRT Model and Predicted Mean Model for {i} input value')

    #     plt.tight_layout()

    #     ax.legend()
            
    #     plt.show()

                