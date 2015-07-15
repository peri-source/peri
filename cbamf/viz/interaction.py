import numpy as np
import matplotlib as mpl
import matplotlib.pylab as pl
from matplotlib.gridspec import GridSpec

class OrthoManipulator(object):
    def __init__(self, state, cmap_abs='bone', cmap_diff='RdBu', vmin=0, vmax=1):
        self.mode = 'view'
        self.state = state
        self.cmap_abs = cmap_abs
        self.cmap_diff = cmap_diff

        self.fig = pl.figure(figsize=(20,10))
        self.gl = GridSpec(2,2)
        self.gr = GridSpec(2,2)

        self.gl.update(left=0.05, right=0.50, bottom=0.01, top=0.95, hspace=0.00, wspace=0.05)
        self.gr.update(left=0.51, right=0.94, bottom=0.01, top=0.95, hspace=0.00, wspace=0.05)

        self.gl.xy = pl.subplot(self.gl[0,0])
        self.gl.yz = pl.subplot(self.gl[0,1])
        self.gl.xz = pl.subplot(self.gl[1,0])

        self.gr.xy = pl.subplot(self.gr[0,0])
        self.gr.yz = pl.subplot(self.gr[0,1])
        self.gr.xz = pl.subplot(self.gr[1,0])

        self.slices = (np.array(self.state.image.shape)/2).astype('int')

        #self.vmin = min([self.state.image.min(), self.state.get_model_image().min()])
        #self.vmax = max([self.state.image.max(), self.state.get_model_image().max()])
        self.vmin = vmin
        self.vmax = vmax
        self.draw()
        self.register_events()

    def _format_ax(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])

    def draw(self):
        self.draw_ortho(self.state.image, self.gl)
        self.draw_ortho(self.state.model_image, self.gr)

    def draw_ortho(self, im, g, cmap=None):
        slices = self.slices
        vmin, vmax = self.vmin, self.vmax

        g.xy.cla()
        g.yz.cla()
        g.xz.cla()

        g.xy.imshow(im[slices[0],:,:], vmin=vmin, vmax=vmax, cmap=self.cmap_abs)
        g.xy.hlines(slices[1], 0, im.shape[2], colors='y', linestyles='dashed', lw=1)
        g.xy.vlines(slices[2], 0, im.shape[1], colors='y', linestyles='dashed', lw=1)
        self._format_ax(g.xy)

        g.yz.imshow(im[:,:,slices[2]].T, vmin=vmin, vmax=vmax, cmap=self.cmap_abs)
        g.yz.hlines(slices[1], 0, im.shape[0], colors='y', linestyles='dashed', lw=1)
        g.yz.vlines(slices[0], 0, im.shape[1], colors='y', linestyles='dashed', lw=1)
        self._format_ax(g.yz)

        g.xz.imshow(im[:,slices[1],:], vmin=vmin, vmax=vmax, cmap=self.cmap_abs)
        g.xz.hlines(slices[0], 0, im.shape[2], colors='y', linestyles='dashed', lw=1)
        g.xz.vlines(slices[2], 0, im.shape[0], colors='y', linestyles='dashed', lw=1)
        self._format_ax(g.xz)

        pl.draw()

    def register_events(self):
        self._calls = []

        self._calls.append(self.fig.canvas.mpl_connect('key_press_event', self.key_press_event))

        if self.mode == 'view':
            self._calls.append(self.fig.canvas.mpl_connect('button_press_event', self.mouse_press_view))
        if self.mode == 'add':
            self._calls.append(self.fig.canvas.mpl_connect('button_press_event', self.mouse_press_add))
        if self.mode == 'remove':
            self._calls.append(self.fig.canvas.mpl_connect('button_press_event', self.mouse_press_remove))
        if self.mode == 'grab':
            self._calls.append(self.fig.canvas.mpl_connect('button_press_event', self.mouse_press_grab))

    def _pt_xyz(self, event):
        x0 = event.xdata
        y0 = event.ydata

        f = False
        for g in [self.gl, self.gr]:
            if event.inaxes == g.xy:
                z = self.slices[0]
                x = x0
                y = y0
                f = True
            if event.inaxes == g.yz:
                z = x0
                x = self.slices[2]
                y = y0
                f = True
            if event.inaxes == g.xz:
                y = self.slices[1]
                z = y0
                x = x0
                f = True

        if f:
            return np.array((z,y,x))
        return None

    def mouse_press_view(self, event):
        self.event = event

        p = self._pt_xyz(event)
        if p is not None:
            self.slices = p
        self.draw()

    def mouse_press_add(self, event):
        self.event = event

        p = self._pt_xyz(event)
        if p is not None:
            print "Adding particle at", p
            self.state.add_particle(p, self.state.obj.rad.mean())
        self.draw()

    def mouse_press_remove(self, event):
        self.event = event

        p = self._pt_xyz(event)
        if p is not None:
            print "Removing particle near", p
            self.state.remove_closest_particle(p)
        self.draw()

    def key_press_event(self, event):
        self.event = event

        if event.key == 'v':
            self.mode = 'view'
        if event.key == 'a':
            self.mode = 'add'
        if event.key == 'r':
            self.mode = 'remove'
        if event.key == 'g':
            self.mode = 'grab'

        print "Switching mode to", self.mode

        for c in self._calls:
            self.fig.canvas.mpl_disconnect(c)

        self.register_events()
