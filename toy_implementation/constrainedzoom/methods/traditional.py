import copy
import numpy as np
import numpy.testing
from ..fft_wrapper import in_real_space, FFTArray, complex_dot
from . import UnfilteredZoomConstrained


class TraditionalZoomConstrained(UnfilteredZoomConstrained):
    description = "Hahn/Abel"
    realspace_convolution = True
    constrain_noise_directly = True # i.e. constraints are applied directly to independent noise vectors
    convolve_lores_at_hires = True # i.e. bring the lores noise in the centre of W into nW BEFORE convolving; otherwise interpolation errors will be greater

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.realspace_convolution:
            self.set_Chigh_realspace()

    def get_default_plot_padding(self):
        """The default plot padding (in coarse pixels) to hide from the high-res region"""
        return self.n2 // (self.pixel_size_ratio) // 4

    @property
    def _B_window_slice(self):
        B_window_size = self.n2 // (self.pixel_size_ratio) // 2
        offset_B = self.offset + self.n2 // self.pixel_size_ratio // 2 - B_window_size // 2
        return slice(offset_B,offset_B+B_window_size)


    @in_real_space
    def _modify_whitenoise(self, delta_low: FFTArray, delta_high: FFTArray):
        assert not delta_high.fourier
        assert not delta_low.fourier

        delta_high = self._remove_lr_pixel_info(delta_high)

        delta_low_zeroed = copy.copy(delta_low)

        if self.convolve_lores_at_hires:
            delta_low_zeroed[self._B_window_slice] = 0


        self._delta_low_residual = (delta_low-delta_low_zeroed).view(FFTArray)
        self._delta_low_residual.fourier = False

        lo_modes_for_hi_window = self.upsample_zeroorder(self._delta_low_residual)

        delta_high += lo_modes_for_hi_window

        return delta_low_zeroed, delta_high

    def _remove_lr_pixel_info(self, hr_vector):
        return hr_vector - self._only_lr_pixel_info(hr_vector)

    def _only_lr_pixel_info(self, hr_vector):
        return self.upsample_zeroorder(self.downsample(hr_vector))

    def _separate_fields(self, delta_low: FFTArray, delta_high: FFTArray):
        return delta_low, delta_high, None

    @in_real_space
    def _recombine_fields(self, delta_low, delta_high, _):
        delta_high += self.upsample_cubic(delta_low)
        delta_low += self._apply_transfer_function(self._delta_low_residual.in_fourier_space()).in_real_space()

        return delta_low, delta_high

    @in_real_space
    def _covector_to_vector(self, lr_covec, hr_covec):
        return lr_covec*self.pixel_size_ratio, self._remove_lr_pixel_info(hr_covec)

    @in_real_space
    def _covector_vector_inner_product(self, lr1, hr1, lr2, hr2):
        return np.dot(lr1,lr2) + np.dot(hr1,hr2)

    @in_real_space
    def _covector_covector_inner_product(self, lr_covec1, hr_covec1, lr_covec2, hr_covec2):
        return self._covector_vector_inner_product(lr_covec1, hr_covec1,
                                                   *self._covector_to_vector(lr_covec2, hr_covec2))

    def _covector_norm(self, lr_covec, hr_covec):
        return self._covector_covector_inner_product(lr_covec, hr_covec, lr_covec, hr_covec)

    def add_constraint(self, val=0.0, hr_covec=None):
        if len(self.constraints)>0:
            raise RuntimeError("Gram-Schmidt not yet implemented for traditional constraints")

        if hr_covec is None:
            hr_covec = self._default_constraint_hr_vec()

        # Unable to constrain in the 'pad' region
        np.testing.assert_allclose(hr_covec[:self.n2//4], 0, atol=1e-2)
        np.testing.assert_allclose(hr_covec[(self.n2*3)//4:], 0, atol=1e-2)

        # filter is (C_P^(1/2) P W^+/m)
        lr_covec = FFTArray(self.downsample(hr_covec, in_window=True))
        lr_covec.in_fourier_space()
        lr_covec*=self.C_low**0.5/self.pixel_size_ratio
        lr_covec.in_real_space()

        # filter is (I-WP^+PW^+) XC^{1/2}X+
        hr_covec = FFTArray(copy.copy(hr_covec))

        hr_covec.in_fourier_space()
        hr_covec*=self.C_high**0.5
        hr_covec.in_real_space()
        hr_covec = self._remove_lr_pixel_info(hr_covec)

        if np.allclose(lr_covec, 0, atol=1e-7):
            print("NOTE: pure HR covector")

        norm = self._covector_norm(lr_covec, hr_covec)

        lr_covec/=np.sqrt(norm)
        hr_covec/=np.sqrt(norm)
        val/=np.sqrt(norm)

        self.constraints.append((lr_covec, hr_covec))
        self.constraints_val.append(val)

    @in_real_space
    def _apply_constraints(self, noise_P, noise_W, verbose):
        for (P_covec, W_covec), val in zip(self.constraints, self.constraints_val):
            scale = val - self._covector_vector_inner_product(P_covec, W_covec, noise_P, noise_W)
            P_vec, W_vec = self._covector_to_vector(P_covec, W_covec)
            #np.testing.assert_allclose(W_vec, 0, atol=1e-8) # temporary while debugging - work only with LR covectors, vectors
            noise_P+=scale*P_vec
            noise_W+=scale*W_vec
            #print("Inf constrained value --> ",np.dot(P_covec, P_vec) + np.dot(W_covec, W_vec))

        return noise_P, noise_W


class BertschingerZoomConstrained(TraditionalZoomConstrained):
    description = "Bertschinger"
    realspace_convolution = False