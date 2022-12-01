import matplotlib.pyplot as plt
from toy_models.envs.utils import fig2img, save_mp4
from jupyterthemes import jtplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

jtplot.style()
plt.rcParams['axes.grid'] = False
plt.rc('axes', unicode_minus=False)

# Color maps
COLORS =['gist_earth', 'turbo' ]

class DensityPlot:
    def __init__(self, 
            x,
            y,
            X,
            Y,
            likelihood,
            density,
            reward,
            mode,
            video_name
            ):
        '''
        Generate plots in each step for the likelihood, the estimated density
        and the reward evolution.
        '''

        self.mode = mode
        self.video_name = video_name
        self.img_list = []
        self.fig, self.ax = plt.subplots(1,3, figsize=(20,4))

        self.lh_ax = self.ax[0]
        self.density_ax = self.ax[1]
        self.reward_ax = self.ax[2]

        self.cb1 = None
        self.cb2 = None
        self.cb3 = None

        self.frame = 0
        self.first_state = False

        plt.tight_layout()
              
        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, 
                            top=0.90, wspace=0.4, hspace=0)

        # Save generated points to use in each frame
        self.x = x
        self.y = y 
        self.X = X
        self.Y = Y

        self.render_lh(likelihood)
        self.render_density(density)
        self.render_reward(reward)
        
        if self.mode == 'human':
            plt.show(block=False)
        
    
    def render(self, likelihood, density, reward, history, done):
        self.render_lh(likelihood, history)
        self.render_density(density, history)
        self.render_reward(reward, history)
        self.frame += 1
        
        if self.mode == 'human': 
            plt.pause(0.00001)
        if self.mode == 'video':
            img = fig2img(self.fig)
            self.img_list.append(img)
            if done:
                save_mp4(self.img_list, self.video_name)
    

            

    def render_lh(self, likelihood, history=None):
        '''Plot fixed likelihood'''
        self.lh_ax.clear()
        if self.cb1 is not None:
            self.cb1.remove()
        self.lh_ax.title.set_text('Likelihood')
        cp1 = self.lh_ax.scatter(self.X, self.Y, c=likelihood, cmap=COLORS[0])
        divider = make_axes_locatable(self.lh_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cb1 = self.fig.colorbar(cp1, cax=cax, orientation='vertical')
 
        if history is not None:
            self.lh_ax.scatter(history[:,0], history[:,1], color='orangered', s=20)

    def render_density(self, density, history=None):
        '''Plot initial density'''
        self.density_ax.clear()
        if self.cb2 is not None:
            self.cb2.remove()
        self.density_ax.title.set_text('Density')
        cp1 = self.density_ax.scatter(self.X, self.Y, c=density, cmap=COLORS[0])
        divider = make_axes_locatable(self.density_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cb2 = self.fig.colorbar(cp1, cax=cax, orientation='vertical')

    def render_reward(self, reward, history=None):
        '''Plot initial reward'''
        self.reward_ax.clear()
        if self.cb3 is not None:
            self.cb3.remove()
        self.reward_ax.title.set_text('Reward')
        cp1 = self.reward_ax.scatter(self.X, self.Y, c=reward, cmap=COLORS[1])
        divider = make_axes_locatable(self.reward_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cb3 = self.fig.colorbar(cp1, cax=cax, orientation='vertical')



    def add_colorbar(self, ax, cb, plot):
        if cb is not None:
            cb.remove()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = self.fig.colorbar(plot, cax=cax, orientation='vertical')





