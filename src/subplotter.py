import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import numpy as np



class plotter(object):
    

    def __init__(self,images, title:list=None,save_path:str=None,show_plot:bool=True,slice=10):
        self.images = images
        self.title = title
        self.save_path = save_path
        self.show_plot = show_plot
        self.fig=plt.figure(figsize=(16,8), dpi=80)


    def plotter_a(self):
    # Create a figure of size 8x6 inches, 80 dots per inch
        for idx in range(self.images[0].shape[-1]):
            self.animate(idx)
            time.sleep(0.1)
            plt.show()
            #plt.close()

    def animate(self,slic):                        
            
            for  idx,img in enumerate(self.images):
                # Create a new subplot from a grid of 1x1
                if len(self.images)>3:
                    plt.subplot(np.ceil(len(self.images)/2),len(self.images)/2,idx+1)
                else:
                    plt.subplot(1,len(self.images),idx+1)

                # Set the title and labels
                plt.title(self.title)
                plt.imshow(img[:,:,slic], cmap='gray')
                # Plot the data

                # Save the figure if a path was provided
                if self.save_path:
                    plt.savefig(self.save_path)

                # Show the figure if show_plot is True
                if True:
                    plt.show()


        

