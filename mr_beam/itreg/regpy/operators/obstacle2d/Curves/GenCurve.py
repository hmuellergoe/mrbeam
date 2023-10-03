import numpy as np


class GenCurve:
    """ parameterized smooth closed curve in R^2 without self-crossing
        parametrization by function name(t), 0<=t<=2*pi (counter-clockwise)
    """
    def __init__(self, **kwargs):
        self.name=None
        """ values of z(t) and its derivatives at equidistant grid"""
        self.z=None
        self.zp=None
        self.zpp=None
        self.zppp=None
        self.zpabs=None   # |z'(t)|
        self.normal=None # outer normal vector (not normalized)






#    def GenCurve_func(self, name):
#        self.name=name

    def bd_eval(self, n, der) :

        t=2*np.pi*np.linspace(0, n-1, n)/n
        """feval ?"""
        self.z = eval(self.name)(t,0)
#        self.z=kite(t, 0)
        if der>=1:
            """feval ?"""
            self.zp = eval(self.name)(t,1)
            self.zpabs = np.sqrt(self.zp[0,:]**2 + self.zp[1,:]**2)
            #outer normal vector
            self.normal = [self.zp[1,:]-self.zp[1,:]]

        if der>=2:
            """feval ?"""
            self.zpp = eval(self.name)(t,2)

        if der>=3:
            """feval ?"""
            self.zppp = eval(self.name)(t,3)

        if der>3:
            raise ValueError('only derivatives up to order 3 implemented')



def kite(t, der):
    res=np.zeros((2,t.shape[0]))
    n=t.shape[0]
    if der==0:
        res= np.append(np.cos(t)+0.65*np.cos(2*t)-0.65,   1.5*np.sin(t)).reshape(2, n)
    elif der==1:
        res = np.append(-np.sin(t)-1.3*np.sin(2*t)    ,   1.5*np.cos(t)).reshape(2, n)
    elif der==2:
        res = np.append(-np.cos(t)-2.6*np.cos(2*t)    ,   -1.5*np.sin(t)).reshape(2, n)
    elif der==3:
        res = np.append(np.sin(t)+5.2*np.sin(2*t)     ,   -1.5*np.cos(t)).reshape(2, n)
    else:
        raise ValueError('derivative not implemented')
    return res
