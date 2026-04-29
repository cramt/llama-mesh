//! Minimal FFI bindings for ggml backend + RPC functions.
//!
//! Links against libggml, libggml-base, libggml-rpc from nixpkgs' llama-cpp.
//! Only the functions we actually call are declared here.

use std::ffi::{c_char, c_void};

/// Opaque device handle — pointer to `ggml_backend_dev` in C.
pub type GgmlBackendDev = *mut c_void;

// Device type constants (from enum ggml_backend_dev_type)
pub const GGML_BACKEND_DEVICE_TYPE_CPU: i32 = 0;
pub const GGML_BACKEND_DEVICE_TYPE_GPU: i32 = 1;

extern "C" {
    // ---- backend device enumeration (ggml-backend.h) ----

    pub fn ggml_backend_dev_count() -> usize;
    pub fn ggml_backend_dev_get(index: usize) -> GgmlBackendDev;
    pub fn ggml_backend_dev_name(device: GgmlBackendDev) -> *const c_char;
    pub fn ggml_backend_dev_description(device: GgmlBackendDev) -> *const c_char;
    pub fn ggml_backend_dev_type(device: GgmlBackendDev) -> i32;
    pub fn ggml_backend_dev_memory(device: GgmlBackendDev, free: *mut usize, total: *mut usize);

    // ---- RPC (ggml-rpc.h) ----

    /// Start an RPC server. **Blocks forever.**
    pub fn ggml_backend_rpc_start_server(
        endpoint: *const c_char,
        cache_dir: *const c_char,
        n_threads: usize,
        n_devices: usize,
        devices: *mut GgmlBackendDev,
    );

    /// Query memory on a remote RPC endpoint.
    pub fn ggml_backend_rpc_get_device_memory(
        endpoint: *const c_char,
        device: u32,
        free: *mut usize,
        total: *mut usize,
    );
}

// ---- safe helpers ----

pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub description: String,
    pub dev_type: i32,
    pub vram_free_mb: u64,
    pub vram_total_mb: u64,
}

impl DeviceInfo {
    pub fn is_gpu(&self) -> bool {
        self.dev_type == GGML_BACKEND_DEVICE_TYPE_GPU
    }
}

/// Enumerate all ggml backend devices.
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    unsafe {
        let count = ggml_backend_dev_count();
        (0..count)
            .map(|i| {
                let dev = ggml_backend_dev_get(i);
                let name = std::ffi::CStr::from_ptr(ggml_backend_dev_name(dev))
                    .to_string_lossy()
                    .into_owned();
                let description = std::ffi::CStr::from_ptr(ggml_backend_dev_description(dev))
                    .to_string_lossy()
                    .into_owned();
                let dev_type = ggml_backend_dev_type(dev);
                let mut free = 0usize;
                let mut total = 0usize;
                ggml_backend_dev_memory(dev, &mut free, &mut total);

                DeviceInfo {
                    index: i,
                    name,
                    description,
                    dev_type,
                    vram_free_mb: (free / (1024 * 1024)) as u64,
                    vram_total_mb: (total / (1024 * 1024)) as u64,
                }
            })
            .collect()
    }
}

/// Start the RPC server on `endpoint` exposing `device_indices`.
/// This **blocks forever** — call from a dedicated thread.
pub fn run_rpc_server(endpoint: &str, device_indices: &[usize]) {
    let endpoint_c = std::ffi::CString::new(endpoint).expect("invalid endpoint string");
    let mut handles: Vec<GgmlBackendDev> = device_indices
        .iter()
        .map(|&i| unsafe { ggml_backend_dev_get(i) })
        .collect();

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    unsafe {
        ggml_backend_rpc_start_server(
            endpoint_c.as_ptr(),
            std::ptr::null(),
            n_threads,
            handles.len(),
            handles.as_mut_ptr(),
        );
    }
}
