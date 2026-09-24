//! Energy per decision from NVML's total energy counter, as spec/13-benchmarks.md asks.
//!
//! NVML comes with the driver, so like the driver it is loaded at run time and a machine without
//! it only loses the energy numbers. The counter is the board's energy since the driver loaded, in
//! millijoules, so a measurement is the difference between two reads around a run.

use std::ffi::c_void;
use std::os::raw::{c_int, c_uint};
use std::time::Instant;

use kime_tensor::{Error, Result};
use libloading::Library;

type Device = *mut c_void;
type Init = unsafe extern "C" fn() -> c_int;
type Handle = unsafe extern "C" fn(c_uint, *mut Device) -> c_int;
type Energy = unsafe extern "C" fn(Device, *mut u64) -> c_int;
type Power = unsafe extern "C" fn(Device, *mut c_uint) -> c_int;

/// One GPU's energy and power counters.
pub struct Meter {
    device: Device,
    energy: Energy,
    power: Power,
    _lib: Library,
}

impl std::fmt::Debug for Meter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Meter").finish_non_exhaustive()
    }
}

fn nvml(what: &str, code: c_int) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(Error::Device(format!("NVML {what} returned {code}"))) }
}

impl Meter {
    /// Opens NVML and the GPU at `ordinal`, in NVML's order, which is the PCI order that
    /// `CUDA_DEVICE_ORDER=PCI_BUS_ID` gives CUDA as well.
    pub fn open(ordinal: u32) -> Result<Self> {
        let name = if cfg!(windows) { "nvml.dll" } else { "libnvidia-ml.so.1" };
        // SAFETY: loading NVML runs only its library constructors, which the driver package
        // ships to be loaded by any process.
        let lib =
            unsafe { Library::new(name) }.map_err(|e| Error::Device(format!("{name}: {e}")))?;
        let sym = |s: &str| Error::Device(format!("NVML has no {s}"));
        // SAFETY: each type is the C signature NVML's header gives the symbol, and the pointers
        // are only called while `_lib` keeps the library loaded.
        let (init, handle, energy, power) = unsafe {
            (
                *lib.get::<Init>(b"nvmlInit_v2\0").map_err(|_| sym("nvmlInit_v2"))?,
                *lib.get::<Handle>(b"nvmlDeviceGetHandleByIndex_v2\0")
                    .map_err(|_| sym("nvmlDeviceGetHandleByIndex_v2"))?,
                *lib.get::<Energy>(b"nvmlDeviceGetTotalEnergyConsumption\0")
                    .map_err(|_| sym("nvmlDeviceGetTotalEnergyConsumption"))?,
                *lib.get::<Power>(b"nvmlDeviceGetPowerUsage\0")
                    .map_err(|_| sym("nvmlDeviceGetPowerUsage"))?,
            )
        };
        let mut device: Device = std::ptr::null_mut();
        // SAFETY: init takes nothing, and handle writes one pointer through a valid `&mut`.
        unsafe {
            nvml("init", init())?;
            nvml("device handle", handle(ordinal, &mut device))?;
        }
        Ok(Self { device, energy, power, _lib: lib })
    }

    /// Millijoules the board has used since the driver loaded.
    pub fn millijoules(&self) -> Result<u64> {
        let mut mj = 0u64;
        // SAFETY: `device` came from NVML for a loaded library, and the call writes one u64.
        nvml("total energy", unsafe { (self.energy)(self.device, &mut mj) })?;
        Ok(mj)
    }

    /// The board's power draw right now, in watts.
    pub fn watts(&self) -> Result<f64> {
        let mut mw: c_uint = 0;
        // SAFETY: as in `millijoules`, with one u32 written.
        nvml("power", unsafe { (self.power)(self.device, &mut mw) })?;
        Ok(f64::from(mw) / 1e3)
    }

    /// Runs `f` and returns its seconds and joules, along with what it returns.
    pub fn measure<T>(&self, f: impl FnOnce() -> T) -> Result<(T, f64, f64)> {
        let (e, t) = (self.millijoules()?, Instant::now());
        let out = f();
        let s = t.elapsed().as_secs_f64();
        Ok((out, s, (self.millijoules()? - e) as f64 / 1e3))
    }

    /// The board's draw with nothing of ours running, from the counter over `secs`.
    pub fn idle_watts(&self, secs: f64) -> Result<f64> {
        let (_, s, j) =
            self.measure(|| std::thread::sleep(std::time::Duration::from_secs_f64(secs)))?;
        Ok(j / s)
    }
}
