from vtk import *
import numpy as np
from vtk.util import numpy_support


def read_vtk(filename) :
    reader=vtkGenericDataObjectReader()
    reader.SetFileName(filename)
    reader.Update()
    image = reader.GetOutput()

    x,y,z = image.GetDimensions()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    vtk_data = image.GetPointData().GetScalars()
    numpy_array = numpy_support.vtk_to_numpy(vtk_data)
    numpy_array = np.reshape(numpy_array,[z,y,x])
    image_infos = {'spacing' : spacing, 'origin' : origin, 'shape' : (x,y,z)}
    return numpy_array, image_infos

def write_vtk(filename, numpy_array, image_info):

    x,y,z = numpy_array.shape
    raveled = numpy_array.ravel()
    vtk_data = numpy_support.numpy_to_vtk(raveled, deep=True)
    image = vtkImageData();
    image.SetDimensions([z,y,x])
    image.SetSpacing(image_info['spacing'])
    image.SetOrigin(image_info['origin'])
    image.GetPointData().SetScalars(vtk_data)

    writer=vtkGenericDataObjectWriter()
    writer.SetInputData(image)
    writer.SetFileName(filename)
    writer.SetFileTypeToBinary()
    writer.Write()
