%exception
{
    _herrno = 0;

    $action

    if (_herrno != 0)
    {
        switch(_herrno)
        {
            case EPERM:
                PyErr_Format(PyExc_IndexError, "Index out of range");
                break;
            case ENOMEM:
                PyErr_Format(PyExc_MemoryError, "Failed malloc()");
                break;
            default:
                PyErr_Format(PyExc_Exception, "Unknown exception");
        }
        SWIG_fail;
    }
}

