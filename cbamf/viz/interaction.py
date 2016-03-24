import numpy as np
import matplotlib as mpl
import matplotlib.pylab as pl
from matplotlib.gridspec import GridSpec

from cbamf import runner

class OrthoManipulator(object):
    def __init__(self, state, cmap_abs='bone', cmap_diff='RdBu', incsize=18.0):
        self.incsize = incsize
        self.modes = ['view', 'add', 'remove', 'grab', 'optimize']
        self.mode = self.modes[0]

        self.insets = ['exposure']
        self.inset = 'none'

        self.views = ['field', 'diff', 'cropped']
        self.view = self.views[2]

        self.modifiers = ['none', 'fft']
        self.modifier = self.modifiers[0]

        self.state = state
        self.cmap_abs = cmap_abs
        self.cmap_diff = cmap_diff

        sh = self.state.image.shape
        q = float(sh[1]) / (sh[0]+sh[1])

        self.fig = pl.figure(figsize=(16,8))

        h = 0.5
        self.gl = {}
        self.gl['xy'] = self.fig.add_axes((h*0.0, 1-q, h*q,     q))
        self.gl['yz'] = self.fig.add_axes((h*q,   1-q, h*(1-q), q))
        self.gl['xz'] = self.fig.add_axes((h*0.0, 0.0, h*q,     1-q))
        self.gl['in'] = self.fig.add_axes((h*q,   0.0, h*(1-q), 1-q))

        self.gr = {}
        self.gr['xy'] = self.fig.add_axes((h+h*0.0, 1-q, h*q,     q))
        self.gr['yz'] = self.fig.add_axes((h+h*q,   1-q, h*(1-q), q))
        self.gr['xz'] = self.fig.add_axes((h+h*0.0, 0.0, h*q,     1-q))
        self.gr['in'] = self.fig.add_axes((h+h*q,   0.0, h*(1-q), 1-q))

        self.slices = (np.array(self.state.image.shape)/2).astype('int')

        self._grab = None
        self.set_field()
        self.draw()
        self.register_events()

    def _format_ax(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])

    def set_field(self):
        if self.view == 'field':
            out = self.state.model_image
            vmin, vmax = 0.0, 1.0
            cmap = self.cmap_abs

        if self.view == 'diff':
            out = (self.state.image - self.state.get_model_image())
            vmin, vmax = -0.2, 0.2
            cmap = self.cmap_diff

        if self.view == 'cropped':
            out = self.state.get_model_image()
            vmin, vmax = 0.0, 1.0
            cmap = self.cmap_abs

        if self.modifier == 'fft':
            out = np.real(np.abs(np.fft.fftn(out)))**0.2
            out[0,0,0] = 1.0
            out = np.fft.fftshift(out)
            vmin = out.min()
            vmax = out.max()
            cmap = self.cmap_abs

        self.field = out
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap

    def draw(self):
        self.draw_ortho(self.state.image, self.gl, cmap=self.cmap_abs, vmin=0.0, vmax=1.0)
        self.draw_ortho(self.field, self.gr, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)

    def draw_ortho(self, im, g, cmap=None, vmin=0, vmax=1):
        slices = self.slices

        g['xy'].cla()
        g['yz'].cla()
        g['xz'].cla()
        g['in'].cla()

        g['xy'].imshow(im[slices[0],:,:], vmin=vmin, vmax=vmax, cmap=cmap)
        g['xy'].hlines(slices[1], 0, im.shape[2], colors='y', linestyles='dashed', lw=1)
        g['xy'].vlines(slices[2], 0, im.shape[1], colors='y', linestyles='dashed', lw=1)
        self._format_ax(g['xy'])

        g['yz'].imshow(im[:,:,slices[2]].T, vmin=vmin, vmax=vmax, cmap=cmap)
        g['yz'].hlines(slices[1], 0, im.shape[0], colors='y', linestyles='dashed', lw=1)
        g['yz'].vlines(slices[0], 0, im.shape[1], colors='y', linestyles='dashed', lw=1)
        self._format_ax(g['yz'])

        g['xz'].imshow(im[:,slices[1],:], vmin=vmin, vmax=vmax, cmap=cmap)
        g['xz'].hlines(slices[0], 0, im.shape[2], colors='y', linestyles='dashed', lw=1)
        g['xz'].vlines(slices[2], 0, im.shape[0], colors='y', linestyles='dashed', lw=1)
        self._format_ax(g['xz'])

        if self.inset == 'exposure':
            m = im*self.state.image_mask
            self.pix = np.r_[m[slices[0],:,:], m[:,:,slices[2]], m[:,slices[1],:]].ravel()
            self.pix = self.pix[self.pix != 0.]
            g['in'].hist(self.pix, bins=300, histtype='step')
            g['in'].semilogy()

            g['in'].set_xlim(0, 1)
            g['in'].set_ylim(9e-1, 1e3)

            if self.view == 'diff' and g == self.gr:
                g['in'].set_xlim(-0.3, 0.3)

        self._format_ax(g['in'])
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
        if self.mode == 'optimize':
            self._calls.append(self.fig.canvas.mpl_connect('button_press_event', self.mouse_press_optimize))
        if self.mode == 'grab':
            self._calls.append(self.fig.canvas.mpl_connect('button_press_event', self.mouse_press_grab))
            self._calls.append(self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_move_grab))
            self._calls.append(self.fig.canvas.mpl_connect('button_release_event', self.mouse_release_grab))
            self._calls.append(self.fig.canvas.mpl_connect('scroll_event', self.mouse_scroll_grab))

    def mouse_press_grab(self, event):
        self.event = event
        p = self._pt_xyz(event)

        if p is not None:
            self._grab = self.state.closest_particle(p)
            self._pos = self.state.obj.pos[self._grab]
        else:
            self._grab = None

    def mouse_release_grab(self, event):
        self.event = event
        self._grab = None

    def mouse_move_grab(self, event):
        self.event = event
        p = self._pt_xyz(event)

        if p is not None and self._grab is not None:
            b = self.state.block_particle_pos(self._grab)
            self.state.update(b, p)
            self.set_field()
            self.draw()

    def mouse_scroll_grab(self, event):
        self.event = event
        p = self._pt_xyz(event)

        if p is not None:
            n = self.state.closest_particle(p)
            b = self.state.block_particle_rad(n)
            self.state.update(b, self.state.obj.rad[n]+event.step/self.incsize)
            self.set_field()
            self.draw()

    def _pt_xyz(self, event):
        x0 = event.xdata
        y0 = event.ydata

        f = False
        for g in [self.gl, self.gr]:
            if event.inaxes == g['xy']:
                z = self.slices[0]
                x = x0
                y = y0
                f = True
            if event.inaxes == g['yz']:
                z = x0
                x = self.slices[2]
                y = y0
                f = True
            if event.inaxes == g['xz']:
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
            print "Moving view to %r" % p
            self.slices = p
        self.draw()

    def mouse_press_add(self, event):
        self.event = event

        p = self._pt_xyz(event)
        if p is not None:
            p = np.array(p)
            r = self.state.obj.rad[self.state.obj.typ==1.].mean()

            print "Adding particle at", p, r
            self.state.add_particle(p, r)
        self.set_field()
        self.draw()

    def mouse_press_remove(self, event):
        self.event = event

        p = self._pt_xyz(event)
        if p is not None:
            print "Removing particle near", p
            self.state.remove_closest_particle(p)
        self.set_field()
        self.draw()

    def mouse_press_optimize(self, event):
        self.event = event
        p = self._pt_xyz(event)

        if p is not None:
            print "Optimizing particle near", p
            n = self.state.closest_particle(p)
            bl = self.state.blocks_particle(n)
            runner.sample_state(self.state, bl, stepout=0.1, doprint=True, N=3)

        self.set_field()
        self.draw()

    def cycle(self, c, clist):
        return clist[(clist.index(c)+1) % len(clist)]

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
        if event.key == 'e':
            self.mode = 'optimize'
        if event.key == 'q':
            self.view = self.cycle(self.view, self.views)
            self.set_field()
            self.draw()
            return
        if event.key == 'w':
            self.modifier = self.cycle(self.modifier, self.modifiers)
            self.set_field()
            self.draw()
            return

        print "Switching mode to", self.mode

        for c in self._calls:
            self.fig.canvas.mpl_disconnect(c)

        self.register_events()

#=============================================================================
# A simpler version for a single 3D field viewer
#=============================================================================
class OrthoViewer(object):
    def __init__(self, field, onesided=True, vmin=None, vmax=None, cmap='bone'):
        """ Easy interactive viewing of 3D ndarray with view selection """
        self.vmin = vmin
        self.vmax = vmax
        self.field = field
        self.onesided = onesided

        if self.onesided is None:
            self.cmap = cmap
        else:
            self.cmap = 'bone' if self.onesided else 'RdBu_r'

        sh = self.field.shape
        q = float(sh[1]) / (sh[0]+sh[1])

        self.fig = pl.figure(figsize=(14,14))

        self.g = {}
        self.g['xy'] = self.fig.add_axes((0.0, 1-q, q,     q))
        self.g['yz'] = self.fig.add_axes((q,   1-q, (1-q), q))
        self.g['xz'] = self.fig.add_axes((0.0, 0.0, q,     1-q))
        self.g['in'] = self.fig.add_axes((q,   0.0, (1-q), 1-q))

        self.slices = (np.array(self.field.shape)/2).astype('int')

        self.draw()
        self.register_events()

    def _format_ax(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])

    def draw(self):
        self.draw_ortho(self.field, self.g, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)

    def draw_ortho(self, im, g, cmap=None, vmin=0, vmax=1):
        slices = self.slices
        if vmin is None:
            vmin = im.min()
        if vmax is None:
            vmax = im.max()

        if self.cmap == 'RdBu_r':
            val = np.max(np.abs([vmin, vmax]))
            vmin = -val
            vmax = val

        g['xy'].cla()
        g['yz'].cla()
        g['xz'].cla()
        g['in'].cla()

        g['xy'].imshow(im[slices[0],:,:], vmin=vmin, vmax=vmax, cmap=cmap)
        g['xy'].hlines(slices[1], 0, im.shape[2], colors='y', linestyles='dashed', lw=1)
        g['xy'].vlines(slices[2], 0, im.shape[1], colors='y', linestyles='dashed', lw=1)
        self._format_ax(g['xy'])

        g['yz'].imshow(im[:,:,slices[2]].T, vmin=vmin, vmax=vmax, cmap=cmap)
        g['yz'].hlines(slices[1], 0, im.shape[0], colors='y', linestyles='dashed', lw=1)
        g['yz'].vlines(slices[0], 0, im.shape[1], colors='y', linestyles='dashed', lw=1)
        self._format_ax(g['yz'])

        g['xz'].imshow(im[:,slices[1],:], vmin=vmin, vmax=vmax, cmap=cmap)
        g['xz'].hlines(slices[0], 0, im.shape[2], colors='y', linestyles='dashed', lw=1)
        g['xz'].vlines(slices[2], 0, im.shape[0], colors='y', linestyles='dashed', lw=1)
        self._format_ax(g['xz'])
        self._format_ax(g['in'])
        pl.draw()

    def register_events(self):
        self._calls = []
        self._calls.append(self.fig.canvas.mpl_connect('button_press_event', self.mouse_press_view))

    def _pt_xyz(self, event):
        x0 = event.xdata
        y0 = event.ydata

        f = False
        if event.inaxes == self.g['xy']:
            z = self.slices[0]
            x = x0
            y = y0
            f = True
        if event.inaxes == self.g['yz']:
            z = x0
            x = self.slices[2]
            y = y0
            f = True
        if event.inaxes == self.g['xz']:
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
