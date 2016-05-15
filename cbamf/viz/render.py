import os
import vtk
import numpy
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as pl

from cbamf.initializers import normalize

def norm(field, vmin=0, vmax=255):
    field = 255*normalize(field)
    field = field.astype('uint8')
    return field

def roll(field):
    return np.rollaxis(field, 0, 2)

def clip(field):
    return np.clip(field, 0, 1)

def extract_field(state, field='exp-particles'):
    if field == 'exp-particles':
        out = ((1-state.image)*(state.obj.get_field() > 1e-5))[state.inner]
    elif field == 'exp-platonic':
        out = ((1-state.image)*(state._platonic_image() > 1e-5))[state.inner]
    elif field == 'sim-particles':
        out = state.obj.get_field()[state.inner]
    elif field == 'sim-platonic':
        out = state._platonic_image()[state.inner]
    else:
        raise AttributeError("Not a field")
    return norm(clip(roll(out)))

def cmap2colorfunc(cmap='bone'):
    values = np.arange(255)
    colors = mpl.cm.__dict__.get(cmap)(mpl.colors.Normalize()(values))

    colorFunc = vtk.vtkColorTransferFunction()
    for v, c in zip(values, colors):
        colorFunc.AddRGBPoint(v, *c[:-1])
    return colorFunc

def volume_render(field, outfile, maxopacity=1.0, cmap='bone', vmin=None, vmax=None,
        mip=True, size=600, elevation=45, azimuth=45, bkg=(0.0, 0.0, 0.0),
        opacitycut=0.35, offscreen=False):
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
    if mip:
        mipFunction = vtk.vtkVolumeRayCastMIPFunction()
        mipFunction.SetMaximizeMethodToOpacity()
        volumeMapper.SetVolumeRayCastFunction(mipFunction)
    else:
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)

    renderer.AddVolume(volume)
    renderer.SetBackground(*bkg)
    renderWin.SetSize(size, size)

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

    #writer = vtk.vtkFFMPEGWriter()
    #writer.SetQuality(2)
    #writer.SetRate(24)
    #w2i = vtk.vtkWindowToImageFilter()
    #w2i.SetInput(renderWin)
    #writer.SetInputConnection(w2i.GetOutputPort())
    #writer.SetFileName('movie.avi')
    #writer.Start()
    #writer.End()

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

