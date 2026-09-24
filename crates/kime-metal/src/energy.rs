//! Energy per decision on Apple silicon from the SMC's `PSTR` key, as spec/13-benchmarks.md asks.
//!
//! `PSTR` is the power the whole machine draws, in watts, and reading it needs no root. It is a
//! rate and not a counter, so a measurement samples it on a thread every few milliseconds and adds
//! up the samples. It covers everything the machine runs, so a busy machine shows up as a higher
//! idle draw, and the idle draw should be read in the same session.

use std::ffi::{c_char, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use kime_tensor::{Error, Result};

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(name: *const c_char) -> *mut c_void;
    fn IOServiceGetMatchingService(main_port: u32, matching: *mut c_void) -> u32;
    fn IOServiceOpen(service: u32, task: u32, kind: u32, connect: *mut u32) -> i32;
    fn IOServiceClose(connect: u32) -> i32;
    fn IOObjectRelease(object: u32) -> i32;
    fn IOConnectCallStructMethod(
        connect: u32,
        selector: u32,
        input: *const c_void,
        input_size: usize,
        output: *mut c_void,
        output_size: *mut usize,
    ) -> i32;
}

unsafe extern "C" {
    static mach_task_self_: u32;
}

/// The SMC's call argument, laid out as the AppleSMC user client expects.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct Param {
    key: u32,
    version: [u8; 6],
    limits: [u32; 4],
    size: u32,
    kind: u32,
    attributes: u8,
    /// The C struct for size, type and attributes is padded to 12 bytes.
    _pad: [u8; 3],
    result: u8,
    status: u8,
    command: u8,
    data32: u32,
    bytes: [u8; 32],
}

const _: () = assert!(size_of::<Param>() == 80);
const READ_INFO: u8 = 9;
const READ: u8 = 5;

/// An open connection to the SMC, reading the machine's total power.
#[derive(Debug)]
pub struct Meter {
    connect: u32,
    info: Param,
}

impl Meter {
    /// Opens the SMC and checks that it has `PSTR` as a float.
    pub fn open() -> Result<Self> {
        let mut connect = 0;
        // SAFETY: the name is a NUL terminated literal, `IOServiceGetMatchingService` takes the
        // dictionary's reference, and `IOServiceOpen` writes one u32 through a valid `&mut`.
        unsafe {
            let service = IOServiceGetMatchingService(0, IOServiceMatching(c"AppleSMC".as_ptr()));
            if service == 0 {
                return Err(Error::Device("no AppleSMC service".into()));
            }
            let r = IOServiceOpen(service, mach_task_self_, 0, &mut connect);
            IOObjectRelease(service);
            if r != 0 {
                return Err(Error::Device(format!("opening the SMC returned {r:#x}")));
            }
        }
        let mut m = Self { connect, info: Param::default() };
        let key = u32::from_be_bytes(*b"PSTR");
        let out = m.call(Param { key, command: READ_INFO, ..Param::default() })?;
        if out.kind != u32::from_be_bytes(*b"flt ") || out.size != 4 {
            return Err(Error::Device("the SMC's PSTR is not a float".into()));
        }
        m.info = Param { key, size: out.size, kind: out.kind, command: READ, ..Param::default() };
        Ok(m)
    }

    fn call(&self, input: Param) -> Result<Param> {
        let mut out = Param::default();
        let mut n = size_of::<Param>();
        // SAFETY: both pointers are to live `Param`s of the size passed, which is the size the
        // AppleSMC user client's selector 2 reads and writes.
        let r = unsafe {
            IOConnectCallStructMethod(
                self.connect,
                2,
                (&raw const input).cast(),
                size_of::<Param>(),
                (&raw mut out).cast(),
                &mut n,
            )
        };
        if r != 0 || out.result != 0 {
            return Err(Error::Device(format!("SMC call returned {r:#x}, result {}", out.result)));
        }
        Ok(out)
    }

    /// The machine's power draw right now, in watts.
    pub fn watts(&self) -> Result<f64> {
        let out = self.call(self.info)?;
        let b = out.bytes;
        Ok(f64::from(f32::from_ne_bytes([b[0], b[1], b[2], b[3]])))
    }

    /// Runs `f` and returns its seconds and joules, along with what it returns. A second
    /// connection samples the power every 5 ms while `f` runs.
    pub fn measure<T>(&self, f: impl FnOnce() -> T) -> Result<(T, f64, f64)> {
        let stop = Arc::new(AtomicBool::new(false));
        let sampler = {
            let (stop, meter) = (Arc::clone(&stop), Meter::open()?);
            std::thread::spawn(move || -> Result<f64> {
                let (mut joules, mut last, mut w) = (0.0, Instant::now(), meter.watts()?);
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(5));
                    let (now, next) = (Instant::now(), meter.watts()?);
                    joules += (w + next) / 2.0 * (now - last).as_secs_f64();
                    (last, w) = (now, next);
                }
                Ok(joules)
            })
        };
        let t = Instant::now();
        let out = f();
        let s = t.elapsed().as_secs_f64();
        stop.store(true, Ordering::Relaxed);
        let j =
            sampler.join().map_err(|_| Error::Device("the power sampler panicked".into()))??;
        Ok((out, s, j))
    }

    /// The machine's draw with nothing of ours running, averaged over `secs`.
    pub fn idle_watts(&self, secs: f64) -> Result<f64> {
        let (_, s, j) = self.measure(|| std::thread::sleep(Duration::from_secs_f64(secs)))?;
        Ok(j / s)
    }
}

impl Drop for Meter {
    fn drop(&mut self) {
        // SAFETY: `connect` came from `IOServiceOpen` and is closed once, here.
        unsafe {
            IOServiceClose(self.connect);
        }
    }
}
