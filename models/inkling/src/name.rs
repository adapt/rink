//! Fixed-size names keep weight lookup formatting off the heap.
use std::fmt::{self, Write};
pub(crate) struct Name {
    bytes: [u8; 192],
    len: usize,
}
impl Name {
    pub fn new(args: fmt::Arguments<'_>) -> Self {
        let mut name = Self {
            bytes: [0; 192],
            len: 0,
        };
        name.write_fmt(args)
            .expect("internal tensor name exceeds capacity");
        name
    }
}
impl Write for Name {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self
            .len
            .checked_add(s.len())
            .filter(|&n| n <= self.bytes.len())
            .ok_or(fmt::Error)?;
        self.bytes[self.len..end].copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}
impl std::ops::Deref for Name {
    type Target = str;
    fn deref(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).expect("formatted UTF-8")
    }
}
impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self)
    }
}
macro_rules! name { ($($arg:tt)*) => { $crate::name::Name::new(format_args!($($arg)*)) }; }
pub(crate) use name;
