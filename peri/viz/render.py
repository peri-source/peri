import os
import vtk
import numpy
import numpy as np

import peri.comp.comp
import peri.comp.objs

import matplotlib as mpl
import matplotlib.pyplot as pl


def norm(field, vmin=0, vmax=255):
    """Truncates field to 0,1; then normalizes to a uin8 on [0,255]"""
    field = 255*np.clip(field, 0, 1)
    field = field.astype('uint8')
    return field


def roll(field):
    return np.rollaxis(field, 0, 2)


def clip(field):
    return np.clip(field, 0, 1)


def extract_field(state, field='exp-particles'):
    """
    Given a state, extracts a field. Extracted value depends on the value
    of field:
        'exp-particles' : The inverted data in the regions of the particles,
                zeros otherwise -- i.e. particles + noise.
        'exp-platonic'  : Same as above, but nonzero in the region of the
                entire platonic image -- i.e. platonic + noise.
        'sim-particles' : Just the particles image; no noise from the data.
        'sim-platonic'  : Just the platonic image; no noise from the data.
    """
    es, pp = field.split('-')  # exp vs sim, particles vs platonic
    # 1. The weights for the field, based off the platonic vs particles
    if pp == 'particles':
        o = state.get('obj')
        if isinstance(o, peri.comp.comp.ComponentCollection):
            wts = 0*o.get()[state.inner]
            for c in o.comps:
                if isinstance(c, peri.comp.objs.PlatonicSpheresCollection):
                    wts += c.get()[state.inner]
        else:
            wts = o.get()[state.inner]
    elif pp == 'platonic':
        wts = state.get('obj').get()[state.inner]
    else:
        raise ValueError('Not a proper field.')
    # 2. Exp vs sim-like data
    if es == 'exp':
        out = (1-state.data) * (wts > 1e-5)
    elif es == 'sim':
        out = wts
    else:
        raise ValueError('Not a proper field.')
    return norm(clip(roll(out)))


def cmap2colorfunc(cmap='bone'):
    values = np.arange(255)
    colors = mpl.cm.__dict__.get(cmap)(mpl.colors.Normalize()(values))

    colorFunc = vtk.vtkColorTransferFunction()
    for v, c in zip(values, colors):
        colorFunc.AddRGBPoint(v, *c[:-1])
    return colorFunc


def volume_render(field, outfile, maxopacity=1.0, cmap='bone', size=600,
                  elevation=45, azimuth=45, bkg=(0.0, 0.0, 0.0),
                  opacitycut=0.35, offscreen=False, rayfunction='smart'):
    """
    Uses vtk to make render an image of a field, with control over the
    camera angle and colormap.

    Input Parameters
    ----------------
        field : np.ndarray
            3D array of the field to render.
        outfile : string
            The save name of the image.
        maxopacity : Float
            Default is 1.0
        cmap : matplotlib colormap string
            Passed to cmap2colorfunc. Default is bone.
        size : 2-element list-like of ints or Int
            The size of the final rendered image.
        elevation : Numeric
            The elevation of the camera angle, in degrees. Default is 45
        azimuth : Numeric
            The azimuth of the camera angle, in degrees. Default is 45
        bkg : Tuple of floats
            3-element tuple of floats on [0,1] of the background image color.
            Default is (0., 0., 0.).
    """
    sh = field.shape

    dataImporter = vtk.vtkImageImport()
    dataImporter.SetDataScalarTypeToUnsignedChar()
    data_string = field.tostring()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))

    dataImporter.SetDataExtent(0, sh[2]-1, 0, sh[1]-1, 0, sh[0]-1)
    dataImporter.SetWholeExtent(0, sh[2]-1, 0, sh[1]-1, 0, sh[0]-1)

    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.0)
    alphaChannelFunc.AddPoint(int(255*opacitycut), maxopacity)

    volumeProperty = vtk.vtkVolumeProperty()
    colorFunc = cmap2colorfunc(cmap)
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)

    volumeMapper = vtk.vtkVolumeRayCastMapper()
    if rayfunction == 'mip':
        comp = vtk.vtkVolumeRayCastMIPFunction()
        comp.SetMaximizeMethodToOpacity()
    elif rayfunction == 'avg':
        comp = vtk.vtkVolumeRayCastCompositeFunction()
    elif rayfunction == 'iso':
        comp = vtk.vtkVolumeRayCastIsosurfaceFunction()
        comp.SetIsoValue(maxopacity/2)
    else:
        comp = vtk.vtkVolumeRayCastIsosurfaceFunction()
    volumeMapper.SetSampleDistance(0.1)
    volumeMapper.SetVolumeRayCastFunction(comp)

    if rayfunction == 'smart':
        volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    light = vtk.vtkLight()
    light.SetLightType(vtk.VTK_LIGHT_TYPE_HEADLIGHT)
    light.SetIntensity(5.5)
    light.SwitchOn()

    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderWin.SetOffScreenRendering(1)

    if not hasattr(size, '__iter__'):
        size = (size, size)

    renderer.AddVolume(volume)
    renderer.AddLight(light)
    renderer.SetBackground(*bkg)
    renderWin.SetSize(*size)

    if offscreen:
        renderWin.SetOffScreenRendering(1)

    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    renderWin.AddObserver("AbortCheckEvent", exitCheck)

    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.Initialize()
    renderWin.Render()
    renderInteractor.Start()

    writer = vtk.vtkPNGWriter()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renderWin)
    writer.SetInputConnection(w2i.GetOutputPort())

    renderWin.Render()
    ac = renderer.GetActiveCamera()
    ac.Elevation(elevation)
    ac.Azimuth(azimuth)
    renderer.ResetCameraClippingRange()
    renderWin.Render()
    w2i.Modified()
    writer.SetFileName(outfile)
    writer.Write()

