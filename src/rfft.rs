extern crate num;

use std::f64::consts::PI;
use std::slice;

use rustfft::{FFT, FFTnum};
use rustfft::algorithm::Radix4;
use rustfft::num_traits::{FromPrimitive};
use rfft::num::complex::Complex;

use ::{RFFT, RIFFT, Length};

pub struct RFFTImpl<T> {
    len: usize,
    cos: Box<[T]>,
    fft: Box<FFT<T>>
}

pub struct RIFFTImpl<T> {
    len: usize,
    cos: Box<[T]>,
    fft: Box<FFT<T>>
}

#[inline]
fn cast<T: FFTnum>(arr: &mut [T]) -> &mut [Complex<T>] {
    assert!(arr.len() % 2 == 0);
    let new_len = arr.len() >> 1;
    unsafe {
        let ptr = arr.as_mut_ptr() as *mut Complex<T>;
        (slice::from_raw_parts_mut(ptr, new_len))
    }
}

fn gen_cos<T: FFTnum>(len: usize) -> Vec<T> {
    let mut cos_table = Vec::with_capacity(len);
    let n = len * 2;
    for i in 0..len {
        cos_table.push(FromPrimitive::from_f64((PI * (i as f64) / n as f64).cos()).unwrap());
    }
    return cos_table;
}

impl<T: FFTnum> RFFTImpl<T> {
    pub fn new(len: usize) -> Self {
        let cos = gen_cos(len / 4);
        RFFTImpl {
            len: len,
            cos: cos.into_boxed_slice(),
            fft: Box::new(Radix4::new(len / 2, false)),
        }
    }
}

impl<T: FFTnum> RFFT<T> for RFFTImpl<T> {

    fn process(&self, input: &mut [T], mut output: &mut [Complex<T>]) {

        self.fft.process(cast(input), &mut output);

        let dc = output[0].re;
        output[0].re = dc + output[0].im;
        output[0].im = dc - output[0].im;

        let end_table = self.len / 4;
        let m: T = FromPrimitive::from_f64(0.5f64).unwrap();

        for i in 1..end_table {
            let cos_idx = i;
            let sin_idx = end_table - i;
            let j = self.len / 2 - i;

            let a_re = m * (output[i].re + output[j].re);
            let b_im = m * (output[j].re - output[i].re);
            let a_im = m * (output[i].im - output[j].im);
            let b_re = m * (output[i].im + output[j].im);
            let sum_re = b_re * self.cos[cos_idx] + b_im * self.cos[sin_idx];
            let sum_im = b_im * self.cos[cos_idx] - b_re * self.cos[sin_idx];
            output[i].re = a_re + sum_re;
            output[i].im = a_im + sum_im;
            output[j].re = a_re - sum_re;
            output[j].im = sum_im - a_im;
        }
    }
}

impl<T> Length for RFFTImpl<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<T: FFTnum> RIFFTImpl<T> {
    pub fn new(len: usize) -> Self {
        let cos = gen_cos(len / 4);
        RIFFTImpl {
            len: len,
            cos: cos.into_boxed_slice(),
            fft: Box::new(Radix4::new(len / 2, true)),
        }
    }
}

impl<T: FFTnum> RIFFT<T> for RIFFTImpl<T> {
    fn process(&self, mut input: &mut [Complex<T>], output: &mut [T]) {

        let dc = input[0].re;
        let m: T = FromPrimitive::from_f64(0.5f64).unwrap();
        input[0].re = m * (dc + input[0].im);
        input[0].im = m * (dc - input[0].im);

        let end_table = self.len / 4;

        for i in 1..end_table {
            let cos_idx = i;
            let sin_idx = end_table - i;
            let j = self.len / 2 - i;

            let a_re = m * (input[i].re + input[j].re);
            let b_im = m * (input[i].re - input[j].re);
            let a_im = m * (input[i].im - input[j].im);
            let b_re = m * (-input[i].im - input[j].im);

            let sum_re = b_re * self.cos[cos_idx] - b_im * self.cos[sin_idx];
            let sum_im = b_im * self.cos[cos_idx] + b_re * self.cos[sin_idx];
            input[i].re = a_re + sum_re;
            input[i].im = a_im + sum_im;
            input[j].re = a_re - sum_re;
            input[j].im = sum_im - a_im;
        }

        self.fft.process(&mut input, cast(output));
    }
}

impl<T> Length for RIFFTImpl<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

