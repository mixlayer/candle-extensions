use candle::cuda::CudaDType;
pub use cudarc::cublaslt::Activation;
use cudarc::driver::DevicePtr;
use cudarc::driver::DeviceRepr;
use float8::F8E4M3;

use candle::backend::BackendStorage;
use candle::{CpuStorage, DType, Layout, Result, Shape, Storage, Tensor};
use half::{bf16, f16};
use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Fp8Matmul, MatmulConfig, ScaleMode};

use crate::CublasLt;

// pub enum Fp8Scale {
//     Scalar(f32),
//     Row(candle::Tensor),
//     Block(candle::Tensor),
// }

pub struct FP8CublasLTMatmul {
    pub cublaslt: Arc<CudaBlasLT>,
    pub act: Option<Activation>,
    pub c: Option<Tensor>,
    pub alpha: Option<f32>,
    pub beta: Option<f32>,

    pub fp8_a_scale: f32,
    pub fp8_b_scale: f32,

    pub out_dtype: DType,
}

impl FP8CublasLTMatmul {
    pub fn fwd_f8e4m3<T>(
        &self,
        a: &candle::CudaStorage,
        a_l: &Layout,
        b: &candle::CudaStorage,
        b_l: &Layout,
        bias: Option<&candle::CudaStorage>,
        bias_l: Option<&Layout>,
    ) -> Result<(candle::CudaStorage, Shape)>
    where
        T: CudaDType + DeviceRepr,
        CudaBlasLT: Fp8Matmul<T>,
    {
        let dev = a.device();

        // Assume TN
        let (m, k) = a_l.shape().dims2()?;
        let (n, b_1) = b_l.shape().dims2()?;

        if b_1 != k {
            candle::bail!("This layer only supports TN layout");
        }

        let lda = k;
        let ldb = k;
        let ldc = m;

        let out_shape = Shape::from((n, m));

        let a = a.as_cuda_slice::<F8E4M3>()?.slice(a_l.start_offset()..);
        let b = b.as_cuda_slice::<F8E4M3>()?.slice(b_l.start_offset()..);

        let bias = if let (Some(bias), Some(bias_l)) = (bias, bias_l) {
            if bias_l.shape().dims1()? != m {
                candle::bail!("Bias does not have the correct shape");
            }

            Some(bias.as_cuda_slice::<T>()?.slice(bias_l.start_offset()..))
        } else {
            None
        };

        // Coerce to trait object expected by fp8_matmul
        let bias_ptr: Option<&dyn DevicePtr<T>> = bias.as_ref().map(|b| b as &dyn DevicePtr<T>);

        let mut out = if let Some(c) = &self.c {
            let (c, c_l) = c.storage_and_layout();

            let c = match &*c {
                Storage::Cuda(storage) => (*storage.as_cuda_slice::<T>()?).clone(),
                _ => candle::bail!("`c` must be a cuda tensor"),
            };

            match c_l.contiguous_offsets() {
                Some((o1, o2)) => {
                    if o1 != 0 {
                        candle::bail!("`c` start offset must be 0");
                    }
                    if o2 != out_shape.elem_count() {
                        candle::bail!("`c` end offset must be {}", out_shape.elem_count())
                    }
                }
                None => candle::bail!("`c` has to be contiguous"),
            };

            if c_l.shape().dims2()? != (n, m) {
                candle::bail!("`c` does not have the correct shape");
            }

            c.clone()
        } else {
            // Allocate out tensor
            unsafe { dev.alloc::<T>(out_shape.elem_count())? }
        };

        let config = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: m as u64,
            n: n as u64,
            k: k as u64,
            alpha: self.alpha.unwrap_or(1.0),
            lda: lda as i64,
            ldb: ldb as i64,
            beta: self.beta.unwrap_or(0.0),
            ldc: ldc as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };

        unsafe {
            let stream = dev.cuda_stream();

            let a_scale_dev = stream.memcpy_stod(&[self.fp8_a_scale]).unwrap();
            let b_scale_dev = stream.memcpy_stod(&[self.fp8_b_scale]).unwrap();

            match self.cublaslt.fp8_matmul(
                config,
                &a,
                &a_scale_dev,
                ScaleMode::Scalar32f,
                &b,
                &b_scale_dev,
                ScaleMode::Scalar32f,
                &mut out,
                bias_ptr,
                self.act.as_ref(),
            ) {
                Ok(()) => {}
                Err(e) => {
                    candle::bail!("cublaslt fp8_matmul failed: {:?}", e)
                }
            }
        }

        let out = candle::CudaStorage::wrap_cuda_slice(out, dev.clone());

        Ok((out, out_shape))
    }
}

impl candle::CustomOp2 for FP8CublasLTMatmul {
    fn name(&self) -> &'static str {
        "cublaslt-matmul-fp8-scalar"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for cublaslt-matmul")
    }

    fn cuda_fwd(
        &self,
        a: &candle::CudaStorage,
        a_l: &Layout,
        b: &candle::CudaStorage,
        b_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match self.out_dtype {
            DType::BF16 => self.fwd_f8e4m3::<bf16>(a, a_l, b, b_l, None, None),
            DType::F16 => self.fwd_f8e4m3::<f16>(a, a_l, b, b_l, None, None),
            dt => candle::bail!(
                "cublaslt-matmul-fp8-scalar is only supported for bf16/f16 output dtypes: ({dt:?})"
            ),
        }
    }
}

/// Fused fp8 matmul w/ scalar scale + add + Relu/Gelu activation using CublasLt
///
/// # Arguments
///
/// * `a` - Input tensor of size MxK
/// * `b` - Input tensor of size NxK
/// * `a_scale` - Scale factor for A used to dequantize A
/// * `b_scale` - Scale factor for B used to dequantize B
/// * `out` - Optional Output tensor of size NxK.
///           If set and beta != 0, will be added to the end result of A*B before `act`
/// * `alpha` - Optional scaling factor for A*B
/// * `beta` - Optional scaling factor for C
/// * `bias` - Optional bias tensor of size M
/// * `act` - Optional Gelu or Relu activation. If set, will be added to the end result
/// * `cublaslt` - CublasLt handle
///
/// The resulting tensor is of shape NxM
#[allow(clippy::too_many_arguments)]
pub fn fp8_scalar_fused_matmul(
    a: &Tensor,
    a_scale: f32,
    b: &Tensor,
    b_scale: f32,
    out: Option<&Tensor>,
    out_dtype: DType,
    alpha: Option<f32>,
    beta: Option<f32>,
    bias: Option<&Tensor>,
    act: Option<Activation>,
    cublaslt: CublasLt,
) -> Result<Tensor> {
    let op = FP8CublasLTMatmul {
        act,
        cublaslt: cublaslt.0,
        c: out.cloned(),
        alpha,
        beta,
        fp8_a_scale: a_scale,
        fp8_b_scale: b_scale,
        out_dtype,
    };

    if a.dtype() != DType::F8E4M3 {
        candle::bail!("a tensor must be of type f8e4m3");
    }

    if b.dtype() != DType::F8E4M3 {
        candle::bail!("b tensor must be of type f8e4m3");
    }

    if let Some(out) = out {
        if out.dtype() != out_dtype {
            candle::bail!("output tensor must match of type out_dtype: {out_dtype:?}");
        }
    }

    if let Some(bias) = bias {
        if bias.dtype() != out_dtype {
            candle::bail!("bias tensor must match of type out_dtype: {out_dtype:?}");
        }

        todo!()
        //a.apply_op3(b, bias, op)
    } else {
        a.apply_op2(b, op)
    }
}
