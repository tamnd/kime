//! The email cleaner on any text: it must not panic, and it must never give back more than it
//! was given or more than `max_chars` characters.
//!
//!     cargo +nightly fuzz run email -- -max_total_time=3600

#![no_main]

use kime_core::email::clean_email_body;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((&first, rest)) = data.split_first() else { return };
    let Ok(body) = std::str::from_utf8(rest) else { return };
    let max = usize::from(first) * 16;
    let out = clean_email_body(body, max);
    let n = out.chars().count();
    assert!(n <= max, "{n} characters out, max_chars {max}");
    assert!(n <= body.chars().count(), "{n} characters out of {}", body.chars().count());
});
