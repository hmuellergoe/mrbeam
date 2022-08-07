import numpy as np

from regpy.operators import Operator
from regpy.discrs import Discretization

from ehtim.imaging.imager_utils import chisqdata_camp

class ClosureTracePol(Operator):
    def __init__(self, wrapper):
        self.wrapper = wrapper
        domain = Discretization(self.wrapper.xtuple.shape)
        
        _, _, self.A4 = chisqdata_camp(self.wrapper.Obsdata, self.wrapper.Prior, self.wrapper.embed_mask)
        codomain = Discretization(self.A4[0].shape[0], dtype=complex)

        super().__init__(domain, codomain)

    def _eval(self, x, differentiate=False):

        iimage = x[0]
        qimage = x[1]
        uimage = x[2]
        
        brightness_matrix = np.asarray([[iimage, qimage+1j*uimage],[qimage-1j*uimage, iimage]])
        
        vismatrix1 = np.asarray([[ self.A4[0] @ brightness_matrix[0,0], self.A4[0] @ brightness_matrix[0,1]], [self.A4[0] @ brightness_matrix[1,0], self.A4[0] @ brightness_matrix[1,1]]])
        vismatrix2 = np.asarray([[ self.A4[1] @ brightness_matrix[0,0], self.A4[1] @ brightness_matrix[0,1]], [self.A4[1] @ brightness_matrix[1,0], self.A4[1] @ brightness_matrix[1,1]]])
        vismatrix3 = np.asarray([[ self.A4[2] @ brightness_matrix[0,0], self.A4[2] @ brightness_matrix[0,1]], [self.A4[2] @ brightness_matrix[1,0], self.A4[2] @ brightness_matrix[1,1]]])
        vismatrix4 = np.asarray([[ self.A4[3] @ brightness_matrix[0,0], self.A4[3] @ brightness_matrix[0,1]], [self.A4[3] @ brightness_matrix[1,0], self.A4[3] @ brightness_matrix[1,1]]])
        
        self.final_matrix1 = np.transpose(vismatrix1, axes=[2,0,1])
        self.final_matrix2 = np.linalg.inv(np.transpose(vismatrix2, axes=[2,0,1]))
        self.final_matrix3 = np.transpose(vismatrix3, axes=[2,0,1])
        self.final_matrix4 = np.linalg.inv(np.transpose(vismatrix4, axes=[2,0,1]))
        
        final_matrix = self.final_matrix1 @ self.final_matrix2 @ self.final_matrix3 @ self.final_matrix4
                
        return np.trace(final_matrix, axis1=1, axis2=2)

    def _derivative(self, h):
        iimage = h[0]
        qimage = h[1]
        uimage = h[2]
        
        brightness_matrix = np.asarray([[iimage, qimage+1j*uimage],[qimage-1j*uimage, iimage]])
        
        vismatrix1 = np.asarray([[ self.A4[0] @ brightness_matrix[0,0], self.A4[0] @ brightness_matrix[0,1]], [self.A4[0] @ brightness_matrix[1,0], self.A4[0] @ brightness_matrix[1,1]]])
        vismatrix2 = np.asarray([[ self.A4[1] @ brightness_matrix[0,0], self.A4[1] @ brightness_matrix[0,1]], [self.A4[1] @ brightness_matrix[1,0], self.A4[1] @ brightness_matrix[1,1]]])
        vismatrix3 = np.asarray([[ self.A4[2] @ brightness_matrix[0,0], self.A4[2] @ brightness_matrix[0,1]], [self.A4[2] @ brightness_matrix[1,0], self.A4[2] @ brightness_matrix[1,1]]])
        vismatrix4 = np.asarray([[ self.A4[3] @ brightness_matrix[0,0], self.A4[3] @ brightness_matrix[0,1]], [self.A4[3] @ brightness_matrix[1,0], self.A4[3] @ brightness_matrix[1,1]]])
    
        final_matrix1h = np.transpose(vismatrix1, axes=[2,0,1])
        final_matrix2h = np.transpose(vismatrix2, axes=[2,0,1])
        final_matrix3h = np.transpose(vismatrix3, axes=[2,0,1])
        final_matrix4h = np.transpose(vismatrix4, axes=[2,0,1])
    
        #derivative of non-linear part    
        final_matrix = -self.final_matrix1 @ self.final_matrix2 @ self.final_matrix3 @ self.final_matrix4 @ final_matrix4h @ self.final_matrix4 \
            + self.final_matrix1 @ self.final_matrix2 @ final_matrix3h @ self.final_matrix4 \
            - self.final_matrix1 @ self.final_matrix2 @ final_matrix2h @ self.final_matrix2 @ self.final_matrix3 @ self.final_matrix4 \
            + final_matrix1h @ self.final_matrix2 @ self.final_matrix3 @ self.final_matrix4
          
    
        return np.trace(final_matrix, axis1=1, axis2=2)
    
    def _adjoint(self, h):
        final_matrix = np.zeros((len(h),2,2))
        final_matrix[:,0,0] = h
        final_matrix[:,1,1] = h
        
        final_matrix1 = np.conjugate(self.final_matrix1)
        final_matrix2 = np.conjugate(self.final_matrix2)
        final_matrix3 = np.conjugate(self.final_matrix3)
        final_matrix4 = np.conjugate(self.final_matrix4)
        
        vismatrix1 = final_matrix @ np.transpose(final_matrix2 @ final_matrix3 @ final_matrix4, axes=[0,2,1])
        vismatrix2 = -np.transpose(final_matrix1 @ final_matrix2, axes=[0,2,1]) @ final_matrix @ np.transpose(final_matrix2 @ final_matrix3 @ final_matrix4, axes=[0,2,1])
        vismatrix3 = np.transpose(final_matrix1 @ final_matrix2, axes=[0,2,1]) @ final_matrix @ np.transpose(final_matrix4, axes=[0,2,1])
        vismatrix4 = -np.transpose(final_matrix1 @ final_matrix2 @ final_matrix3 @ final_matrix4, axes=[0,2,1]) @ final_matrix @ np.transpose(final_matrix4, axes=[0,2,1])
           
        A4 = np.conjugate(np.asarray(self.A4))
        
        brightness_matrix = np.zeros((2, 2, self.domain.shape[1]), dtype=complex)
        brightness_matrix[0,0] = A4[0].transpose() @ vismatrix1[:,0,0] \
                            + A4[1].transpose() @ vismatrix2[:,0,0] \
                            + A4[2].transpose() @ vismatrix3[:,0,0] \
                            + A4[3].transpose() @ vismatrix4[:,0,0]
        brightness_matrix[0,1] = A4[0].transpose() @ vismatrix1[:,0,1] \
                    + A4[1].transpose() @ vismatrix2[:,0,1] \
                    + A4[2].transpose() @ vismatrix3[:,0,1] \
                    + A4[3].transpose() @ vismatrix4[:,0,1]
        brightness_matrix[1,0] = A4[0].transpose() @ vismatrix1[:,1,0] \
                    + A4[1].transpose() @ vismatrix2[:,1,0] \
                    + A4[2].transpose() @ vismatrix3[:,1,0] \
                    + A4[3].transpose() @ vismatrix4[:,1,0]
        brightness_matrix[1,1] = A4[0].transpose() @ vismatrix1[:,1,1] \
                    + A4[1].transpose() @ vismatrix2[:,1,1] \
                    + A4[2].transpose() @ vismatrix3[:,1,1] \
                    + A4[3].transpose() @ vismatrix4[:,1,1]
                    
        toret = np.zeros(self.domain.shape)
        
        toret[0] = brightness_matrix[0,0] + brightness_matrix[1,1]
        toret[1] = brightness_matrix[0,1] + brightness_matrix[1,0]
        toret[2] = -1j*brightness_matrix[0,1] + 1j*brightness_matrix[1,0]
        
        return np.real(toret)
    
    
    
    
    
    