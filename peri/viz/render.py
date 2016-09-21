import os
import vtk
import numpy
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as pl

def norm(field, vmin=0, vmax=255):
    field = 255*np.clip(field, 0, 1)
    field = field.astype('uint8')
    return field

def roll(field):
    return np.rollaxis(field, 0, 2)

def clip(field):
    return np.clip(field, 0, 1)

def extract_field(state, field='exp-particles'):
    if field == 'exp-particles':
        out = ((1-state.data)*(state.get('obj').get() > 1e-5))[state.inner]
    elif field == 'exp-platonic':
        out = ((1-state.data)*(state._platonic_image() > 1e-5))[state.inner]
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
        size=600, elevation=45, azimuth=45, bkg=(0.0, 0.0, 0.0),
        opacitycut=0.35, offscreen=False, rayfunction='smart'):
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
    renderWin.SetOffScreenRendering(1);

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

