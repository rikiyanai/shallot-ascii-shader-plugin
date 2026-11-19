use crate::Result;

pub(crate) struct Reader<'a> {
    pub bytes: &'a [u8],
    pub offset: usize,
}
impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
    pub fn take(&mut self, size: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(size)
            .ok_or("Draco range overflow")?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or("truncated Draco stream")?;
        self.offset = end;
        Ok(bytes)
    }
    pub fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn varint(&mut self, width: u32) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..width).step_by(7) {
            let byte = self.byte()?;
            let part = u64::from(byte & 127);
            value |= part.wrapping_shl(shift);
            if byte & 128 == 0 {
                return Ok(if width == 32 {
                    u64::from(value as u32)
                } else {
                    value
                });
            }
        }
        Err("Draco varint overflow")
    }
    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn count(&mut self) -> Result<usize> {
        usize::try_from(self.varint(32)?).map_err(|_| "Draco count overflow")
    }
}

pub(crate) struct Bits<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Bits<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    pub fn read(&mut self, count: u32) -> Result<u32> {
        if count > 32
            || self
                .offset
                .checked_add(count as usize)
                .is_none_or(|end| end > self.bytes.len().saturating_mul(8))
        {
            return Err("truncated Draco bits");
        }
        // Least-significant bit first, across byte boundaries.
        let first = self.offset / 8;
        let value = match self.bytes.get(first..first + 8) {
            Some(window) => u64::from_le_bytes(window.try_into().unwrap()),
            None => {
                let mut value = 0u64;
                let last = (self.offset + count as usize).div_ceil(8);
                for (i, &byte) in self.bytes[first..last].iter().enumerate() {
                    value |= u64::from(byte) << (8 * i);
                }
                value
            }
        };
        let value = (value >> (self.offset % 8)) & ((1u64 << count) - 1);
        self.offset += count as usize;
        Ok(value as u32)
    }
    pub fn consumed(&self) -> usize {
        self.offset.div_ceil(8)
    }
}
