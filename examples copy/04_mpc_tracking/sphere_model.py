import LibsControl
import numpy

class SphereDS(LibsControl.DynamicalSystem):

    def __init__(self):

        self.dt = 1.0/100.0

        mat_a = numpy.zeros((4, 4))
        mat_b = numpy.zeros((4, 2))

        mat_a[0][1] = 1.0
        mat_a[2][3] = 1.0

        mat_b[1][0] = 1.0
        mat_b[3][1] = 1.0

        LibsControl.DynamicalSystem.__init__(self, mat_a, mat_b, None, self.dt)
