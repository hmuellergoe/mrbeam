from imagingbase.ehtim_wrapper import EhtimWrapper
from regpy.functionals import Functional
import ehtim.scattering as so
import ehtim.const_def as ehc
import numpy as np
import pdb

class ScatteringWrapper(EhtimWrapper):
    def __init__(self, Obsdata, InitIm, Prior, flux, d='vis', **kwargs):
        super().__init__(Obsdata, InitIm, Prior, flux, d=d, **kwargs)
        
        # Parameters related to scattering
        self.epsilon_list_next = []
        self.scattering_model = self.kwargs.get('scattering_model', None)
        self._sqrtQ = None
        self._ea_ker = None
        self._ea_ker_gradient_x = None
        self._ea_ker_gradient_y = None
        self._alpha_phi_list = []
        #self.alpha_phi_next = self.kwargs.get('alpha_phi', 1e4)
        self.alpha_phi_next = self.kwargs.get('alpha_phi', 1)
        
        if self.scattering_model is None:
            self.scattering_model = so.ScatteringModel()

        # First some preliminary definitions
        wavelength = ehc.C/self.Obsdata.rf*100.0  # Observing wavelength [cm]
        N = self.Prior.xdim

        # Field of view, in cm, at the scattering screen
        FOV = self.Prior.psize * N * self.scattering_model.observer_screen_distance

        # The ensemble-average convolution kernel and its gradients
        self._ea_ker = self.scattering_model.Ensemble_Average_Kernel(self.Prior, wavelength_cm=wavelength)
        ea_ker_gradient = so.Wrapped_Gradient(self._ea_ker/(FOV/N))
        self._ea_ker_gradient_x = -ea_ker_gradient[1]
        self._ea_ker_gradient_y = -ea_ker_gradient[0]

        # The power spectrum
        # Note: rotation is not currently implemented;
        # the gradients would need to be modified slightly
        self._sqrtQ = np.real(self.scattering_model.sqrtQ_Matrix(self.Prior, t_hr=0.0))

        # Generate the initial image+screen vector.
        # By default, the screen is re-initialized to zero each time.
        if len(self.epsilon_list_next) == 0:
            self._xinit = np.concatenate((self._xinit, np.zeros(N**2-1)))
        else:
            self._xinit = np.concatenate((self._xinit, self.epsilon_list_next))
            
class ScatteringFunctional(Functional):
    def __init__(self, handler, domain):
        self.handler = handler
        super().__init__(domain)
        
    def _eval(self, minvec):
        N = self.handler.Prior.xdim

        imvec = minvec[:N**2]
        EpsilonList = minvec[N**2:]
        if self.handler.logim:
            imvec = np.exp(imvec)

        IM = self.handler.formatoutput(imvec)
        # The scattered image vector
        screen = so.MakeEpsilonScreenFromList(EpsilonList, N)
        scatt_im = self.handler.scattering_model.Scatter(IM, Epsilon_Screen=screen,
                                                 ea_ker=self.handler._ea_ker, sqrtQ=self.handler._sqrtQ,
                                                 Linearized_Approximation=True)
        scatt_im = scatt_im.imvec
        if self.handler.dataterm:
            return self.handler._chisq(scatt_im)
        else:
            if self.handler.d == 'rgauss':
                return self.handler._reg(scatt_im)
            elif self.handler.d == 'epsilon':
                # Scattering screen regularization term
                chisq_epsilon = sum(EpsilonList*EpsilonList)/((N*N-1.0)/2.0)
                return self.handler.alpha_phi_next * (chisq_epsilon - 1.0)
            else:
                return self.handler._reg(imvec)
        
    def _gradient(self, minvec):
        wavelength = ehc.C/self.handler.Obsdata.rf*100.0  # Observing wavelength [cm]
        wavelengthbar = wavelength/(2.0*np.pi)     # lambda/(2pi) [cm]
        N = self.handler.Prior.xdim

        # Field of view, in cm, at the scattering screen
        FOV = self.handler.Prior.psize * N * self.handler.scattering_model.observer_screen_distance
        rF = self.handler.scattering_model.rF(wavelength)

        imvec = minvec[:N**2]
        EpsilonList = minvec[N**2:]
        if self.handler.logim:
            imvec = np.exp(imvec)

        IM = self.handler.formatoutput(imvec)

        # The scattered image vector
        screen = so.MakeEpsilonScreenFromList(EpsilonList, N)
        scatt_im = self.handler.scattering_model.Scatter(IM, Epsilon_Screen=screen,
                                                 ea_ker=self.handler._ea_ker, sqrtQ=self.handler._sqrtQ,
                                                 Linearized_Approximation=True)
        scatt_im = scatt_im.imvec

        EA_Image = self.handler.scattering_model.Ensemble_Average_Blur(IM, ker=self.handler._ea_ker)
        EA_Gradient = so.Wrapped_Gradient((EA_Image.imvec/(FOV/N)).reshape(N, N))

        # The gradient signs don't actually matter, but let's make them match intuition
        # (i.e., right to left, bottom to top)
        EA_Gradient_x = -EA_Gradient[1]
        EA_Gradient_y = -EA_Gradient[0]

        Epsilon_Screen = so.MakeEpsilonScreenFromList(EpsilonList, N)
        phi_scr = self.handler.scattering_model.MakePhaseScreen(Epsilon_Screen, IM,
                                                        obs_frequency_Hz=self.handler.Obsdata.rf,
                                                        sqrtQ_init=self.handler._sqrtQ)
        phi = phi_scr.imvec.reshape((N, N))
        phi_Gradient = so.Wrapped_Gradient(phi/(FOV/N))
        phi_Gradient_x = -phi_Gradient[1]
        phi_Gradient_y = -phi_Gradient[0]
        
        gradterm = np.zeros(imvec.shape)
        gradterm_epsilon = np.zeros(N**2-1)
        if self.handler.dataterm:
            daterm = self.handler._chisqgrad(scatt_im)
            dchisq_dIa = daterm.reshape((N, N))

            # Now the chain rule factor to get the chi^2 gradient wrt the unscattered image
            gx = so.Wrapped_Convolve(self.handler._ea_ker_gradient_x[::-1, ::-1], phi_Gradient_x * (dchisq_dIa))
            gx = (rF**2.0 * gx).flatten()

            gy = so.Wrapped_Convolve(self.handler._ea_ker_gradient_y[::-1, ::-1], phi_Gradient_y * (dchisq_dIa))
            gy = (rF**2.0 * gy).flatten()

            gradterm = so.Wrapped_Convolve(
                self.handler._ea_ker[::-1, ::-1], (dchisq_dIa)).flatten() + gx + gy
            
            # Gradient of the data chi^2 wrt to the epsilon screen
            # Preliminary Definitions
            chisq_grad_epsilon = np.zeros(N**2-1)
            i_grad = 0
            ell_mat = np.zeros((N, N))
            m_mat = np.zeros((N, N))
            for ell in range(0, N):
                for m in range(0, N):
                    ell_mat[ell, m] = ell
                    m_mat[ell, m] = m

            # Real part; top row
            for t in range(1, (N+1)//2):
                s = 0
                grad_term = (wavelengthbar/FOV*self.handler._sqrtQ[s][t] *
                             2.0*np.cos(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
                grad_term = so.Wrapped_Gradient(grad_term)
                grad_term_x = -grad_term[1]
                grad_term_y = -grad_term[0]

                cge_term = (EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y)
                chisq_grad_epsilon[i_grad] = np.sum(dchisq_dIa * rF**2 * cge_term)

                i_grad = i_grad + 1

            # Real part; remainder
            for s in range(1, (N+1)//2):
                for t in range(N):
                    grad_term = (wavelengthbar/FOV*self.handler._sqrtQ[s][t] *
                                 2.0*np.cos(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
                    grad_term = so.Wrapped_Gradient(grad_term)
                    grad_term_x = -grad_term[1]
                    grad_term_y = -grad_term[0]

                    cge_term = (EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y)
                    chisq_grad_epsilon[i_grad] = np.sum(dchisq_dIa * rF**2 * cge_term)

                    i_grad = i_grad + 1

            # Imaginary part; top row
            for t in range(1, (N+1)//2):
                s = 0
                grad_term = (-wavelengthbar/FOV*self.handler._sqrtQ[s][t] *
                             2.0*np.sin(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
                grad_term = so.Wrapped_Gradient(grad_term)
                grad_term_x = -grad_term[1]
                grad_term_y = -grad_term[0]

                cge_term = (EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y)
                chisq_grad_epsilon[i_grad] = np.sum(dchisq_dIa * rF**2 * cge_term)

                i_grad = i_grad + 1

            # Imaginary part; remainder
            for s in range(1, (N+1)//2):
                for t in range(N):
                    grad_term = (-wavelengthbar/FOV*self.handler._sqrtQ[s][t] *
                                 2.0*np.sin(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
                    grad_term = so.Wrapped_Gradient(grad_term)
                    grad_term_x = -grad_term[1]
                    grad_term_y = -grad_term[0]

                    cge_term = (EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y)
                    chisq_grad_epsilon[i_grad] = np.sum(dchisq_dIa * rF**2 * cge_term)
                    i_grad = i_grad + 1

            gradterm_epsilon = chisq_grad_epsilon.flatten()
            
        else:
            if self.handler.d == 'rgauss':
                # Get gradient of the scattered image vector
                gaussterm = self.handler._reggrad(scatt_im)
                dgauss_dIa = gaussterm.reshape((N, N))

                # Now the chain rule factor to get the gauss gradient wrt the unscattered image
                gx = so.Wrapped_Convolve(
                    self.handler._ea_ker_gradient_x[::-1, ::-1], phi_Gradient_x * (dgauss_dIa))
                gx = (rF**2.0 * gx).flatten()

                gy = so.Wrapped_Convolve(
                    self.handler._ea_ker_gradient_y[::-1, ::-1], phi_Gradient_y * (dgauss_dIa))
                gy = (rF**2.0 * gy).flatten()

                # Now we add the gradient for the unscattered image
                gradterm = so.Wrapped_Convolve(self.handler._ea_ker[::-1, ::-1], (dgauss_dIa)).flatten() + gx + gy

            elif self.handler.d == 'epsilon':
                # Gradient of the chi^2 regularization term for the epsilon screen
                gradterm_epsilon = self.handler.alpha_phi_next * 2.0*EpsilonList/((N*N-1)/2.0)

            else:
                gradterm = self.handler._reggrad(imvec).flatten()

        # Chain rule term for change of variables
        if self.handler.logim:
            gradterm *= imvec
            gradterm_epsilon *= imvec

        return np.concatenate((gradterm, gradterm_epsilon))
            
