pub extern crate num_complex;
extern crate rustfft;

use num_complex::Complex;

pub mod rfft;
pub use rfft::{RFFTImpl, RIFFTImpl};
pub use rustfft::{FFTnum, Length};

pub trait RFFT<T: FFTnum>: Length + Sync + Send {
    fn process(&self, input: &mut [T], output: &mut [Complex<T>]);
}

pub trait RIFFT<T: FFTnum>: Length + Sync + Send {
    fn process(&self, input: &mut [Complex<T>], output: &mut [T]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::FFT;
    use rustfft::num_complex::Complex;
    use rustfft::num_traits::Zero;
    use std::f64::consts::PI;
    use rustfft::algorithm::Radix4;


    fn check_rfft(sz: usize) {
        let mut src: Vec<f64> = vec![Zero::zero(); sz];
        for i in 0..sz {
            src[i] = ((PI * (i as f64) / 4096 as f64).cos()) as f64;
        }
        let ref_value = src.clone();

        let mut input2: Vec<Complex<f64>> = vec![Zero::zero(); src.len()];
        for i in 0..src.len() {
            input2[i].re = src[i];
            input2[i].im = 0.0f64;
        }

        let mut spectrum1: Vec<Complex<f64>> = vec![Zero::zero(); src.len() / 2];
        let mut spectrum2: Vec<Complex<f64>> = vec![Zero::zero(); src.len()];
        let end = spectrum1.len();

        let ref_fft = Radix4::new(src.len(), false);
        ref_fft.process(&mut input2, &mut spectrum2);

        let rfft = RFFTImpl::new(src.len());

        rfft.process(&mut src, &mut spectrum1);

        assert_eq!(spectrum1.len(), src.len() / 2);
        let mut fake_zero: Complex<f64> = Zero::zero();
        fake_zero.re = spectrum2[0].re;
        fake_zero.im = spectrum2[end].re;
        let mut t = (spectrum1[0] - fake_zero).norm();
        for i in 1..src.len() / 2 {
            t += (spectrum1[i] - spectrum2[i]).norm();
        }
        assert!((t/spectrum1.len() as f64) < 0.000000001f64);

        let mut tmp: Vec<f64> = vec![Zero::zero(); src.len()];

        let rifft = RIFFTImpl::new(src.len());
        rifft.process(&mut spectrum1, &mut tmp);

        t = 0f64;
        for i in 0..src.len() {
            t += (ref_value[i] - (tmp[i]/(sz >> 1) as f64)).abs();
        }
        assert!((t/src.len() as f64) < 0.00000001f64);
    }

    #[test]
    fn rfft_test_4096() {
        check_rfft(4096);
    }

    #[test]
    fn rfft_test_2048() {
        check_rfft(2048);
    }

    #[test]
    fn rfft_test_1024() {
        check_rfft(1024);
    }

    #[test]
    fn rfft_test_512() {
        check_rfft(512);
    }

    #[test]
    fn rfft_test_16() {
        check_rfft(16);
    }

    #[test]
    fn rifft_test_dc() {
        let sz: usize = 16;
        let val: f64 = 16.0f64;
        let mut spectrum1: Vec<Complex<f64>> = vec![Zero::zero(); sz/2];
        let mut tmp: Vec<f64> = vec![Zero::zero(); sz];
        spectrum1[0].re = val;

        let rifft = RIFFTImpl::new(sz);
        rifft.process(&mut spectrum1, &mut tmp);
        for i in 0..tmp.len() {
            assert!((tmp[i]/(sz >> 1) as f64 - val / sz as f64).abs() < 0.000000001f64);
        }
    }
}
