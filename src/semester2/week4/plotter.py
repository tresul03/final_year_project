import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, x, y,test_inputs = None, test_output = None, output = 'n', font_size = 18):
        # # Initialize the plotter with an empty figure and axis
        # self.fig, self.ax = plt.subplots()
        self.y = y
        self.x = x
        self.test_inputs = test_inputs
        self.test_output = test_output
        self.figure = None 
        self.dict = {'n':'Sersic Index', 'f':'Flux/Wm$^{-2}$', 'r':'Half-Radius/kpc'}
        self.output = output
        self.latex_code_star_mass = r'log(M_{star})'
        self.latex_code_dust_mass = r'log(M_{star})/log(M_{dust})'
        self.latex_code_angle = r'Sin(\theta)'
        self.font_size = font_size

        #need to add whether it is plotting n,f or r


    def plot_group_same(self, i:int =0, j:int = 16, color1:str = 'blue', color2:str = 'red'):
       

        while i < j:


            fig_group, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))

            ax1.plot(self.x,self.test_output[i,:],color= color1,lw=3,label=f'SKIRT Model: input : ${self.latex_code_star_mass}$ = {self.test_inputs[i,0]:2.2f}, ${self.latex_code_dust_mass}$ = {self.test_inputs[i,1]:2.2f}, ${self.latex_code_angle}$ = {self.test_inputs[i,2]:2.2f}')
            ax1.plot(self.x,self.y[i,:],color= color2,lw=3,label='Model Predicted Mean')
            ax2.plot(self.x,self.test_output[i+1,:],color= color1,lw=3,label=f'SKIRT Model: input : ${self.latex_code_star_mass}$ = {self.test_inputs[i+1,0]:2.2f}, ${self.latex_code_dust_mass}$ = {self.test_inputs[i+1,1]:2.2f}, ${self.latex_code_angle}$ = {self.test_inputs[i+1,2]:2.2f}')
            ax2.plot(self.x,self.y[i+1,:],color= color2,lw=3,label='Model Predicted Mean')
            ax3.plot(self.x,self.test_output[i+2,:],color= color1,lw=3,label=f'SKIRT Model: input : ${self.latex_code_star_mass}$ = {self.test_inputs[i+2,0]:2.2f}, ${self.latex_code_dust_mass}$ = {self.test_inputs[i+2,1]:2.2f}, ${self.latex_code_angle}$ = {self.test_inputs[i+2,2]:2.2f}')
            ax3.plot(self.x,self.y[i+2,:],color= color2,lw=3,label='Model Predicted Mean')
            ax4.plot(self.x,self.test_output[i+3,:],color= color1,lw=3,label=f'SKIRT Model: input : ${self.latex_code_star_mass}$ = {self.test_inputs[i+3,0]:2.2f}, ${self.latex_code_dust_mass}$ = {self.test_inputs[i+3,1]:2.2f}, ${self.latex_code_angle}$ = {self.test_inputs[i+3,2]:2.2f}')
            ax4.plot(self.x,self.y[i+3,:],color= color2,lw=3,label='Model Predicted Mean')

        
            for ax in fig_group.get_axes():
                #ax.label_outer()
                ax.set(xlabel=f'Wavelength/$\\mu$m', ylabel= f'{self.dict[self.output]}')
                ax.set_xscale('log')
                ax.legend(fontsize=self.font_size)
                
            fig_group.suptitle(f'Comparison of SKIRT Model and BNN Model Predicted Mean for {i} to {i+3} input values', fontsize=self.font_size)
            

            plt.tight_layout()


            i = i+4
            
        plt.show()
        plt.close()



    def plot_single(self, i:int =0, color1:str = 'blue', color2:str = 'red'):

        self.figure, ax = plt.subplots(figsize=(15, 5))


        ax.plot(self.x,self.test_output[i,:],color= color1,lw=3,label=f'SKIRT Model: input : ${self.latex_code_star_mass}$ = {self.test_inputs[i,0]:2.2f}, ${self.latex_code_dust_mass}$ = {self.test_inputs[i,1]:2.2f}, ${self.latex_code_angle}$ = {self.test_inputs[i,2]:2.2f}')
        ax.plot(self.x,self.y[i,:],color= color2,lw=3,label='Model Predicted Mean')

        plt.xlabel(f'Wavelength/$\\mu$m', fontsize = self.font_size)
        plt.ylabel(f'{self.dict[self.output]}', fontsize = self.font_size)
        plt.xscale('log')

        plt.suptitle(f'Comparison of SKIRT Model and BNN Model Predicted Mean for {i} input value', fontsize = self.font_size)

        plt.tight_layout()

        plt.legend(fontsize=self.font_size)

        plt.xticks (fontsize=self.font_size)
        plt.yticks (fontsize=self.font_size)
            
        plt.show()
        plt.close()


    def plot_same_ax(self, i:int = 0, j:int = 5, step_size:int = 1):


        self.figure, ax = plt.subplots(figsize=(15, 5))

        q = i

        while i < j:

            color = plt.cm.autumn(((i-q)+1) / (j-q))

            ax.plot(self.x,self.y[i,:],color= color,lw=3)

            i += step_size

        plt.xlabel(f'Wavelength/$\\mu$m', fontsize=self.font_size)
        plt.ylabel(f'{self.dict[self.output]}', fontsize=self.font_size)
        plt.xscale('log')

        plt.suptitle(f'BNN Model Predicted Mean for {q} to {j}', fontsize = self.font_size)

        plt.tight_layout()

        plt.xticks (fontsize=self.font_size)
        plt.yticks (fontsize=self.font_size)

        plt.show()
        plt.close()

        
    def cost_vs_epochs(self,x,y):

        self.figure, ax = plt.subplots(figsize=(15, 5))

        ax.plot(x,y,lw=3)

        plt.xlabel(f'Epochs')
        plt.ylabel(f'Cost')

        plt.suptitle(f'Cost vs Epochs')

        plt.tight_layout()

        plt.show()
        plt.close()


    def save_figure(self, filename = "figure.png"):
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

                