pub extern crate num_complex;
extern crate rustfft;
extern crate num;

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
    use self::num::complex::Complex;
    use rustfft::num_traits::Zero;
    use std::f64::consts::PI;
    use rustfft::algorithm::Radix4;


    fn check_rfft(mut src: Vec<f64>) {
        let ref_value = src.clone();
        let mut input2: Vec<Complex<f64>> = vec![Zero::zero(); src.len()];
        for i in 0..src.len() {
            input2[i].re = src[i];
        }
        let mut spectrum1: Vec<Complex<f64>> = vec![Zero::zero(); src.len() / 2];
        let mut spectrum2: Vec<Complex<f64>> = vec![Zero::zero(); src.len()];

        let ref_fft = Radix4::new(src.len(), false);
        ref_fft.process(&mut input2, &mut spectrum2);

        let rfft = RFFTImpl::new(src.len());
        rfft.process(&mut src, &mut spectrum1);

        for i in 0..src.len() / 2 {
            assert_eq!((10000f64 * spectrum1[i].re).round(), (10000f64 * spectrum2[i].re).round());
            assert_eq!((10000f64 * spectrum1[i].im).round(), (10000f64 * spectrum2[i].im).round());
        }

        let mut tmp: Vec<f64> = vec![Zero::zero(); src.len()];
        let rifft = RIFFTImpl::new(src.len());
        rifft.process(&mut spectrum1, &mut tmp);
        for i in 0..src.len() {
            assert_eq!((10000f64 * ref_value[i]).round(), (10000f64 * tmp[i]/2048.0).round());
        }
    }
    #[test]
    fn rfft_test() {
        let mut src: Vec<f64> = vec![Zero::zero(); 4096];
        for i in 0..4096 {
            src[i] = ((PI * (i as f64) / 512 as f64).cos()) as f64;
        }
        check_rfft(src);
    }
}


