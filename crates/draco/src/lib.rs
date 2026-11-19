//! Draco 2.2 triangle-mesh decoding, bit-equal to Draco 1.5.9's decoder.
//!
//! `decode` returns faces and attributes; `decode_faces` stops after connectivity.
mod allocation;
mod attributes;
mod buffer;
mod edgebreaker;
mod entropy;
mod prediction;
mod traversal;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use attributes::{Attribute, AttributeType, DataType};

use allocation::{copy, push};
use buffer::Reader;
pub type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    Sequential,
    Edgebreaker,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Metadata {
    pub entries: Vec<(Vec<u8>, Vec<u8>)>,
    pub children: Vec<(Vec<u8>, Metadata)>,
}

#[derive(Debug)]
pub struct Header {
    pub method: Method,
    pub attributes: Vec<(u32, Metadata)>,
    pub metadata: Option<Metadata>,
    pub size: usize,
}

fn name(r: &mut Reader<'_>) -> Result<Vec<u8>> {
    let size = usize::from(r.byte()?);
    copy(r.take(size)?)
}
fn metadata_node(r: &mut Reader<'_>) -> Result<(Metadata, usize)> {
    let count = r.count()?;
    if count > r.remaining() / 3 {
        return Err("invalid Draco metadata entry count");
    }
    let mut ordered = Vec::new();
    for index in 0..count {
        let key = name(r)?;
        let size = r.count()?;
        if size == 0 {
            return Err("empty Draco metadata value");
        }
        let value = copy(r.take(size)?)?;
        push(&mut ordered, (key, index, value))?;
    }
    // The pinned decoder keeps the last value for each duplicate key.
    ordered.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for (key, _, value) in ordered {
        if let Some(last) = entries.last_mut().filter(|last| last.0 == key) {
            last.1 = value;
        } else {
            push(&mut entries, (key, value))?;
        }
    }
    let count = r.count()?;
    if count > r.remaining() {
        return Err("invalid Draco submetadata count");
    }
    Ok((
        Metadata {
            entries,
            children: Vec::new(),
        },
        count,
    ))
}
fn metadata(r: &mut Reader<'_>) -> Result<Metadata> {
    let (root, pending) = metadata_node(r)?;
    let mut stack = Vec::new();
    push(&mut stack, (Vec::new(), root, pending))?;
    loop {
        if stack.last().unwrap().2 == 0 {
            let (key, mut node, _) = stack.pop().unwrap();
            node.children.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            if node.children.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err("duplicate Draco submetadata name");
            }
            if let Some((_, parent, _)) = stack.last_mut() {
                push(&mut parent.children, (key, node))?;
            } else {
                return Ok(node);
            }
        } else {
            // The pinned decoder counts the first child at level zero.
            if stack.len() > 1001 {
                return Err("Draco metadata nesting exceeds 1000");
            }
            let key = name(r)?;
            stack.last_mut().unwrap().2 -= 1;
            let (node, count) = metadata_node(r)?;
            push(&mut stack, (key, node, count))?;
        }
    }
}

/// A decoded mesh: faces over `points` points, and each attribute's values.
pub struct Mesh {
    pub faces: Vec<[u32; 3]>,
    pub points: usize,
    pub attributes: Vec<Attribute>,
}

impl Mesh {
    /// The first attribute with `unique_id`, as `GetAttributeByUniqueId`.
    pub fn attribute(&self, unique_id: u32) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.unique_id == unique_id)
    }
}

/// Decodes a mesh only when its counts match the caller's trusted geometry
/// counts, keeping the attributes whose unique ids are `attributes` (in glTF,
/// the extension's `attributes` values) and refusing an id the stream lacks.
pub fn decode(
    bytes: &[u8],
    expected_faces: usize,
    expected_points: usize,
    attributes: &[u32],
) -> Result<Mesh> {
    let header = parse_header(bytes)?;
    let mut r = Reader::new(bytes);
    r.take(header.size)?;
    let (faces, connectivity) = match header.method {
        Method::Sequential => (sequential(&mut r, expected_faces, expected_points)?, None),
        Method::Edgebreaker => {
            let mut connectivity = edgebreaker::decode(&mut r, expected_faces, expected_points)?;
            (std::mem::take(&mut connectivity.faces), Some(connectivity))
        }
    };
    let attributes = attributes::decode(
        &mut r,
        &faces,
        expected_points,
        connectivity.as_ref(),
        attributes,
    )?;
    Ok(Mesh {
        faces,
        points: expected_points,
        attributes,
    })
}

/// Decodes connectivity only when its counts match the caller's trusted geometry counts.
pub fn decode_faces(
    bytes: &[u8],
    expected_faces: usize,
    expected_points: usize,
) -> Result<Vec<[u32; 3]>> {
    let header = parse_header(bytes)?;
    let mut r = Reader::new(bytes);
    r.take(header.size)?;
    match header.method {
        Method::Sequential => sequential(&mut r, expected_faces, expected_points),
        Method::Edgebreaker => {
            Ok(edgebreaker::decode(&mut r, expected_faces, expected_points)?.faces)
        }
    }
}

fn sequential(
    r: &mut Reader<'_>,
    expected_faces: usize,
    expected_points: usize,
) -> Result<Vec<[u32; 3]>> {
    let count = r.count()?;
    if count != expected_faces {
        return Err("Draco face count mismatch");
    }
    let points = r.varint(32)? as u32;
    if points as usize != expected_points {
        return Err("Draco point count mismatch");
    }
    if count > u32::MAX as usize / 3 || count > r.remaining() / 3 {
        return Err("invalid Draco sequential face count");
    }
    let compressed = r.byte()? == 0;
    let mut faces = Vec::new();
    if compressed {
        let values = entropy::symbols(r, count * 3, 1)?;
        let mut previous = 0i32;
        for triangle in values.as_chunks::<3>().0 {
            let mut face = [0; 3];
            for (out, encoded) in face.iter_mut().zip(triangle) {
                let difference = (encoded >> 1) as i32;
                previous = if encoded & 1 != 0 {
                    previous.checked_sub(difference)
                } else {
                    previous.checked_add(difference)
                }
                .ok_or("Draco sequential index overflow")?;
                if previous < 0 || previous as u32 >= points {
                    return Err("Draco sequential index out of range");
                }
                *out = previous as u32;
            }
            push(&mut faces, face)?;
        }
    } else {
        for _ in 0..count {
            let mut face = [0; 3];
            for out in &mut face {
                *out = if points < 256 {
                    u32::from(r.byte()?)
                } else if points < 1 << 16 {
                    u32::from(u16::from_le_bytes(r.take(2)?.try_into().unwrap()))
                } else if points < 1 << 21 {
                    r.varint(32)? as u32
                } else {
                    u32::from_le_bytes(r.take(4)?.try_into().unwrap())
                };
                if *out >= points {
                    return Err("Draco sequential index out of range");
                }
            }
            push(&mut faces, face)?;
        }
    }
    Ok(faces)
}

pub fn parse_header(bytes: &[u8]) -> Result<Header> {
    let mut r = Reader::new(bytes);
    if r.take(5)? != b"DRACO" {
        return Err("invalid Draco signature");
    }
    if r.byte()? != 2 || r.byte()? != 2 {
        return Err("unsupported Draco version: expected 2.2");
    }
    match r.byte()? {
        1 => (),
        0 => return Err("Draco point clouds are not supported"),
        _ => return Err("invalid Draco geometry type"),
    }
    let method = match r.byte()? {
        0 => Method::Sequential,
        1 => Method::Edgebreaker,
        _ => return Err("unsupported Draco mesh encoding method"),
    };
    let flags = u16::from_le_bytes(r.take(2)?.try_into().unwrap());
    let mut attributes = Vec::new();
    let mut file_metadata = None;
    if flags & 0x8000 != 0 {
        let count = r.count()?;
        if count > r.remaining() {
            return Err("invalid Draco attribute metadata count");
        }
        for _ in 0..count {
            let id = r.varint(32)? as u32;
            let value = metadata(&mut r)?;
            push(&mut attributes, (id, value))?;
        }
        file_metadata = Some(metadata(&mut r)?);
    }
    Ok(Header {
        method,
        attributes,
        metadata: file_metadata,
        size: r.offset,
    })
}
