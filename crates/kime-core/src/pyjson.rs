//! Python's `json.dumps`, byte for byte, for the values a request can hold.
//!
//! Laya renders structured states, criteria and instructions with `json.dumps`, and the compat
//! models were trained on exactly that text. A different float spelling (`1e-5` against Python's
//! `1e-05`) or a different escape (`\u00e9` against `é`) changes the token ids, so this writer copies
//! Python's choices rather than serde_json's.
//!
//! Integers larger than `u64` arrive from serde_json as floats and print as floats, where Python
//! would print them exactly. No real request has hit that yet.

use serde_json::Value;

/// The options Laya uses. Every call site in Laya keeps Python's default separators, `", "` and
/// `": "`, so only `ensure_ascii` varies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dumps {
    /// Escape everything outside printable ASCII as `\uXXXX`, as `json.dumps` does by default.
    pub ensure_ascii: bool,
}

impl Dumps {
    /// `json.dumps(value, ensure_ascii=...)`.
    #[must_use]
    pub fn to_string(self, value: &Value) -> String {
        let mut out = String::new();
        self.write(value, &mut out);
        out
    }

    /// Appends `json.dumps(value)` to `out`.
    pub fn write(self, value: &Value, out: &mut String) {
        match value {
            Value::Null => out.push_str("null"),
            Value::Bool(true) => out.push_str("true"),
            Value::Bool(false) => out.push_str("false"),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    out.push_str(&i.to_string());
                } else if let Some(u) = n.as_u64() {
                    out.push_str(&u.to_string());
                } else {
                    float_repr(n.as_f64().unwrap_or(f64::NAN), out);
                }
            }
            Value::String(s) => self.string(s, out),
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.write(item, out);
                }
                out.push(']');
            }
            Value::Object(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.string(k, out);
                    out.push_str(": ");
                    self.write(v, out);
                }
                out.push('}');
            }
        }
    }

    fn string(self, s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 => push_u(out, c as u32),
                // With ensure_ascii Python escapes everything outside space to tilde, DEL included,
                // and writes chars above the BMP as a surrogate pair.
                c if self.ensure_ascii && !(' '..='~').contains(&c) => {
                    let mut buf = [0u16; 2];
                    for unit in c.encode_utf16(&mut buf) {
                        push_u(out, u32::from(*unit));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
}

fn push_u(out: &mut String, unit: u32) {
    use std::fmt::Write;
    let _ = write!(out, "\\u{unit:04x}");
}

/// Python's `repr(float)`: the shortest digits that round trip, in fixed notation when the decimal
/// exponent is from -4 to 15 and in scientific notation otherwise, with a sign and at least two
/// exponent digits. Integral values keep a `.0`.
pub fn float_repr(f: f64, out: &mut String) {
    if f.is_nan() {
        out.push_str("NaN");
        return;
    }
    if f.is_infinite() {
        out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
        return;
    }
    // `{:e}` gives the shortest round trip digits as `d.ddde-x`, which has everything we need.
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("{:e} writes a decimal exponent");
    let (neg, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mantissa),
    };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp < 0 {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        } else {
            let point = exp as usize + 1;
            if digits.len() <= point {
                out.push_str(&digits);
                for _ in digits.len()..point {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..point]);
                out.push('.');
                out.push_str(&digits[point..]);
            }
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.abs()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn repr(f: f64) -> String {
        let mut s = String::new();
        float_repr(f, &mut s);
        s
    }

    #[test]
    fn floats_print_like_python() {
        // Each right hand side is what CPython 3.12 prints for repr(float).
        let cases = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (1.5, "1.5"),
            (0.1, "0.1"),
            (100.0, "100.0"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (0.00012, "0.00012"),
            (1.5e-7, "1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e16, "1.2345e+16"),
            (123456789.125, "123456789.125"),
            (1e100, "1e+100"),
            (-2.5e-300, "-2.5e-300"),
            (f64::MAX, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
        ];
        for (f, want) in cases {
            assert_eq!(repr(f), want, "{f:e}");
        }
    }

    #[test]
    fn dumps_matches_python() {
        let v = json!({"b": [1, 2.0, true, null], "é": "a\"b\\c\n\u{1}\u{7f}ü😀", "n": -3});
        assert_eq!(
            Dumps { ensure_ascii: false }.to_string(&v),
            "{\"b\": [1, 2.0, true, null], \"é\": \"a\\\"b\\\\c\\n\\u0001\u{7f}ü😀\", \"n\": -3}"
        );
        assert_eq!(
            Dumps { ensure_ascii: true }.to_string(&v),
            "{\"b\": [1, 2.0, true, null], \"\\u00e9\": \"a\\\"b\\\\c\\n\\u0001\\u007f\\u00fc\\ud83d\\ude00\", \"n\": -3}"
        );
    }
}
