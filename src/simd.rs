//! Guard against the AVX -> SSE transition penalty.
//!
//! faer's runtime-dispatched AVX/AVX-512 kernels can return with the upper
//! halves of the vector registers still "dirty". Until a `vzeroupper` runs,
//! every legacy-SSE instruction on that thread (this crate is built for the
//! portable x86-64 baseline, and so is glibc's libm) pays a transition penalty:
//! we measured a 16x slowdown of exp/log-heavy kernels on the affected threads,
//! persisting indefinitely. Call `clean_simd_state` after faer-backed work.

/// Clear upper vector-register state on the calling thread and all rayon workers.
pub fn clean_simd_state() {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            #[target_feature(enable = "avx")]
            unsafe fn zeroupper() {
                std::arch::x86_64::_mm256_zeroupper()
            }
            unsafe { zeroupper() };
            rayon::broadcast(|_| unsafe { zeroupper() });
        }
    }
}
