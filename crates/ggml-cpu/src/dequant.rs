//! Dequantização de blocos GGML para f32.
#![allow(clippy::indexing_slicing)]

use gguf::GgmlType;

use crate::error::DequantError;

/// Converte `bytes` brutos de um tensor para `Vec<f32>`.
/// Suporta F32, F16, Q8_0, Q4_0, Q3_K, Q4_K, Q5_K, Q6_K e IQ3_XXS.
pub fn dequant_to_f32(bytes: &[u8], ty: GgmlType) -> Result<Vec<f32>, DequantError> {
    match ty {
        GgmlType::F32 => dequant_f32(bytes),
        GgmlType::F16 => dequant_f16(bytes),
        GgmlType::Q8_0 => dequant_q8_0(bytes),
        GgmlType::Q4_0 => dequant_q4_0(bytes),
        GgmlType::Q3_K => dequant_q3_k(bytes),
        GgmlType::IQ3_XXS => dequant_iq3_xxs(bytes),
        GgmlType::Q4_K => dequant_q4_k(bytes),
        GgmlType::Q5_K => dequant_q5_k(bytes),
        GgmlType::Q6_K => dequant_q6_k(bytes),
        other => Err(DequantError::UnsupportedType(format!("{other:?}"))),
    }
}

fn dequant_f32(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(DequantError::BadSize {
            ty: "F32",
            block_bytes: 4,
            got: bytes.len(),
        });
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn dequant_f16(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(DequantError::BadSize {
            ty: "F16",
            block_bytes: 2,
            got: bytes.len(),
        });
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect())
}

fn dequant_q8_0(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 34; // 2 (f16) + 32 (i8)
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(DequantError::BadSize {
            ty: "Q8_0",
            block_bytes: BLOCK,
            got: bytes.len(),
        });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for b in bytes.as_chunks::<BLOCK>().0 {
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        for &q in &b[2..34] {
            out.push(q.cast_signed() as f32 * d);
        }
    }
    Ok(out)
}
fn dequant_q4_0(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 18; // 2 (f16) + 16 (nibbles)
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(DequantError::BadSize {
            ty: "Q4_0",
            block_bytes: BLOCK,
            got: bytes.len(),
        });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 32];
    for (bi, b) in bytes.as_chunks::<BLOCK>().0.iter().enumerate() {
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let base = bi * 32;
        for j in 0..16 {
            let q = b[2 + j];
            let x0 = i32::from(q & 0x0F) - 8;
            let x1 = i32::from(q >> 4) - 8;
            out[base + j] = x0 as f32 * d;
            out[base + j + 16] = x1 as f32 * d;
        }
    }
    Ok(out)
}
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

fn dequant_q4_k(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 144; // 2+2+12+128
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(DequantError::BadSize {
            ty: "Q4_K",
            block_bytes: BLOCK,
            got: bytes.len(),
        });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 256];

    for (bi, b) in bytes.as_chunks::<BLOCK>().0.iter().enumerate() {
        let d_val = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let min_val = half::f16::from_bits(u16::from_le_bytes([b[2], b[3]])).to_f32();
        let scales = &b[4..16];
        let qs = &b[16..144];
        let base = bi * 256;
        let mut qs_off = 0usize;
        let mut is = 0usize;

        for j_step in [0usize, 64, 128, 192] {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d_val * f32::from(sc1);
            let m1f = min_val * f32::from(m1);
            let d2 = d_val * f32::from(sc2);
            let m2f = min_val * f32::from(m2);
            for l in 0..32 {
                let q = qs[qs_off + l];
                out[base + j_step + l] = d1 * f32::from(q & 0xF) - m1f;
                out[base + j_step + l + 32] = d2 * f32::from(q >> 4) - m2f;
            }
            qs_off += 32;
            is += 2;
        }
    }
    Ok(out)
}
/// Q5_K: superbloco de 256 elementos em 176 bytes —
/// `d(f16) | dmin(f16) | scales[12] | qh[32] | qs[128]`.
///
/// Mesmo esquema de escalas do Q4_K (6 bits por sub-bloco de 32, via `get_scale_min_k4`),
/// acrescido de um 5º bit por elemento vindo de `qh`: o bit `l` de `qh[l]` pertence ao
/// primeiro grupo de 32 e o bit seguinte ao segundo, avançando 2 bits a cada 64 elementos.
fn dequant_q5_k(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 176; // 2+2+12+32+128
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(DequantError::BadSize {
            ty: "Q5_K",
            block_bytes: BLOCK,
            got: bytes.len(),
        });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 256];

    for (bi, b) in bytes.as_chunks::<BLOCK>().0.iter().enumerate() {
        let d_val = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let min_val = half::f16::from_bits(u16::from_le_bytes([b[2], b[3]])).to_f32();
        let scales = &b[4..16];
        let qh = &b[16..48];
        let qs = &b[48..176];
        let base = bi * 256;
        let mut qs_off = 0usize;
        let mut is = 0usize;
        // Máscaras do bit alto: avançam duas posições a cada grupo de 64 elementos.
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for j_step in [0usize, 64, 128, 192] {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d_val * f32::from(sc1);
            let m1f = min_val * f32::from(m1);
            let d2 = d_val * f32::from(sc2);
            let m2f = min_val * f32::from(m2);
            for l in 0..32 {
                let q = qs[qs_off + l];
                let hi1 = u8::from(qh[l] & u1 != 0) * 16;
                let hi2 = u8::from(qh[l] & u2 != 0) * 16;
                out[base + j_step + l] = d1 * f32::from((q & 0xF) + hi1) - m1f;
                out[base + j_step + l + 32] = d2 * f32::from((q >> 4) + hi2) - m2f;
            }
            qs_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

/// Q3_K: superbloco de 256 elementos em 110 bytes — `hmask[32] | qs[64] | scales[12] | d(f16)`.
///
/// 16 escalas de 6 bits com offset 32: os 4 bits baixos estão em `scales[0..8]` (nibble baixo
/// para as escalas 0–7, alto para 8–15) e os 2 altos em `scales[8..12]`. Cada elemento tem 2 bits
/// em `qs` e um 3º em `hmask`; o 3º bit **zerado** subtrai 4. Mesma ordem de
/// `dequantize_row_q3_K` do llama.cpp.
fn dequant_q3_k(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 110; // 32+64+12+2
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(DequantError::BadSize {
            ty: "Q3_K",
            block_bytes: BLOCK,
            got: bytes.len(),
        });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 256];

    for (bi, b) in bytes.as_chunks::<BLOCK>().0.iter().enumerate() {
        let hm = &b[0..32];
        let qs = &b[32..96];
        let sc = &b[96..108];
        let d_all = half::f16::from_bits(u16::from_le_bytes([b[108], b[109]])).to_f32();

        let mut scales = [0i8; 16];
        for (j, s) in scales.iter_mut().enumerate() {
            let lo = if j < 8 { sc[j] & 0xF } else { sc[j - 8] >> 4 };
            let hi = (sc[8 + j % 4] >> (2 * (j / 4))) & 3;
            *s = (lo | (hi << 4)).cast_signed() - 32;
        }

        let mut y = bi * 256;
        let mut is = 0usize;
        let mut m: u8 = 1;
        for q in qs.as_chunks::<32>().0 {
            for shift in [0u8, 2, 4, 6] {
                for metade in [0usize, 16] {
                    let dl = d_all * f32::from(scales[is]);
                    is += 1;
                    for l in metade..metade + 16 {
                        let v = ((q[l] >> shift) & 3).cast_signed()
                            - if hm[l] & m != 0 { 0 } else { 4 };
                        out[y] = dl * f32::from(v);
                        y += 1;
                    }
                }
                m = m.wrapping_shl(1);
            }
        }
    }
    Ok(out)
}

/// IQ3_XXS: superbloco de 256 elementos em 98 bytes — `d(f16) | qs[64] | escalas_e_sinais[32]`.
///
/// Cada `qs` indexa uma entrada de 4 magnitudes em `IQ3XXS_GRID`. Por grupo de 32 elementos, uma
/// palavra de 32 bits traz a escala (4 bits altos) e quatro índices de sinal de 7 bits; o 8º bit
/// de sinal é a paridade dos outros 7 (é o `ksigns_iq2xs` do llama.cpp, calculado em vez de
/// tabelado). Mesma ordem de `dequantize_row_iq3_xxs`.
fn dequant_iq3_xxs(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 98; // 2+64+32
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(DequantError::BadSize {
            ty: "IQ3_XXS",
            block_bytes: BLOCK,
            got: bytes.len(),
        });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 256];

    for (bi, b) in bytes.as_chunks::<BLOCK>().0.iter().enumerate() {
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let qs = &b[2..66];
        let escalas_e_sinais = &b[66..98];
        let mut y = bi * 256;

        for ib32 in 0..8 {
            let p = &escalas_e_sinais[4 * ib32..4 * ib32 + 4];
            let aux = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
            #[allow(clippy::cast_precision_loss)]
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4 {
                let s7 = ((aux >> (7 * l)) & 127) as u8;
                let sinais = s7 | (((s7.count_ones() & 1) as u8) << 7);
                let g1 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l])].to_le_bytes();
                let g2 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l + 1])].to_le_bytes();
                for j in 0..4 {
                    let sinal = |bit: usize| if sinais & (1 << bit) != 0 { -1.0 } else { 1.0 };
                    out[y + j] = db * f32::from(g1[j]) * sinal(j);
                    out[y + 4 + j] = db * f32::from(g2[j]) * sinal(j + 4);
                }
                y += 8;
            }
        }
    }
    Ok(out)
}

/// `iq3xxs_grid` de `ggml-common.h`: 256 entradas de 4 magnitudes de 8 bits.
#[rustfmt::skip]
const IQ3XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];

fn dequant_q6_k(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 210; // 128+64+16+2
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(DequantError::BadSize {
            ty: "Q6_K",
            block_bytes: BLOCK,
            got: bytes.len(),
        });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 256];

    for (bi, b) in bytes.as_chunks::<BLOCK>().0.iter().enumerate() {
        let ql_full = &b[0..128];
        let qh_full = &b[128..192];
        let sc_full = &b[192..208]; // [i8; 16] stored as u8
        let d_val = half::f16::from_bits(u16::from_le_bytes([b[208], b[209]])).to_f32();
        let base = bi * 256;

        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;

        for n in [0usize, 128] {
            let ql = &ql_full[ql_off..ql_off + 64];
            let qh = &qh_full[qh_off..qh_off + 32];
            let sc = &sc_full[sc_off..sc_off + 8];
            for l in 0..32usize {
                let is = l / 16;
                let q1 = i32::from(((ql[l] & 0xF) | ((qh[l] & 3) << 4)).cast_signed()) - 32;
                let q2 =
                    i32::from(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)).cast_signed()) - 32;
                let q3 = i32::from(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)).cast_signed()) - 32;
                let q4 =
                    i32::from(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)).cast_signed()) - 32;
                let s0 = f32::from(sc[is].cast_signed());
                let s2 = f32::from(sc[is + 2].cast_signed());
                let s4 = f32::from(sc[is + 4].cast_signed());
                let s6 = f32::from(sc[is + 6].cast_signed());
                #[allow(clippy::cast_precision_loss)]
                {
                    out[base + n + l] = d_val * s0 * q1 as f32;
                    out[base + n + l + 32] = d_val * s2 * q2 as f32;
                    out[base + n + l + 64] = d_val * s4 * q3 as f32;
                    out[base + n + l + 96] = d_val * s6 * q4 as f32;
                }
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }

    #[test]
    fn f32_passthrough() {
        let bytes: Vec<u8> = 1.5f32.to_le_bytes().to_vec();
        let out = dequant_to_f32(&bytes, GgmlType::F32).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.5).abs() < 1e-7);
    }

    #[test]
    fn f16_conversion() {
        let bits = f16_bytes(0.5);
        let out = dequant_to_f32(&bits, GgmlType::F16).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn unsupported_type_returns_error() {
        assert!(dequant_to_f32(&[], GgmlType::Q2_K).is_err());
    }

    fn make_q8_0_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(34);
        b.extend_from_slice(&f16_bytes(d));
        b.extend(qs.iter().map(|&q| q.cast_unsigned()));
        b
    }

    #[test]
    fn q8_0_single_block() {
        // d=1.0; qs=[1, -1, 0×30] → out=[1.0, -1.0, 0×30]
        let mut qs = [0i8; 32];
        qs[0] = 1;
        qs[1] = -1;
        let block = make_q8_0_block(1.0, &qs);
        let out = dequant_to_f32(&block, GgmlType::Q8_0).unwrap();
        assert_eq!(out.len(), 32);
        assert!((out[0] - 1.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - (-1.0)).abs() < 1e-5, "out[1]={}", out[1]);
        assert!(out[2..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q8_0_scale_applied() {
        // d=2.0; qs=[3, 0×31] → out[0]=6.0
        let mut qs = [0i8; 32];
        qs[0] = 3;
        let block = make_q8_0_block(2.0, &qs);
        let out = dequant_to_f32(&block, GgmlType::Q8_0).unwrap();
        assert!((out[0] - 6.0).abs() < 1e-4, "out[0]={}", out[0]);
    }

    #[test]
    fn q8_0_bad_size_returns_error() {
        assert!(dequant_to_f32(&[0u8; 33], GgmlType::Q8_0).is_err());
    }

    fn make_q4_0_block(d: f32, qs: &[u8; 16]) -> Vec<u8> {
        let mut b = Vec::with_capacity(18);
        b.extend_from_slice(&f16_bytes(d));
        b.extend_from_slice(qs);
        b
    }

    #[test]
    fn q4_0_single_block() {
        // d=1.0; qs[0]=0x89 → lower=9,x0=1 → out[0]=1.0; upper=8,x1=0; resto 0x88→zeros
        let mut qs = [0x88u8; 16];
        qs[0] = 0x89;
        let block = make_q4_0_block(1.0, &qs);
        let out = dequant_to_f32(&block, GgmlType::Q4_0).unwrap();
        assert_eq!(out.len(), 32);
        assert!((out[0] - 1.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!(
            out[1..16].iter().all(|&v| v == 0.0),
            "out[1..16] deve ser zero"
        );
        assert!(
            out[16..].iter().all(|&v| v == 0.0),
            "out[16..] deve ser zero"
        );
    }

    #[test]
    fn q4_0_bad_size_returns_error() {
        assert!(dequant_to_f32(&[0u8; 17], GgmlType::Q4_0).is_err());
    }

    fn make_q4_k_block(d: f32, dmin: f32, scales: &[u8; 12], qs: &[u8; 128]) -> Vec<u8> {
        let mut b = Vec::with_capacity(144);
        b.extend_from_slice(&f16_bytes(d));
        b.extend_from_slice(&f16_bytes(dmin));
        b.extend_from_slice(scales);
        b.extend_from_slice(qs);
        b
    }

    #[test]
    fn q4_k_two_active_sub_blocks() {
        // d=1.0, dmin=1.0
        // scales=[8, 4, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0]
        //   is=0: get(0)→sc=8,mn=0; get(1)→sc=4,mn=0
        //   is=2...: →sc=0,mn=0 (all zero)
        // qs = [0x22 × 128]: nibbles both = 2
        // out[0..32]   = 8*2-0 = 16.0
        // out[32..64]  = 4*2-0 = 8.0
        // out[64..256] = 0.0
        let scales: [u8; 12] = [8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let qs = [0x22u8; 128];
        let block = make_q4_k_block(1.0, 1.0, &scales, &qs);
        let out = dequant_to_f32(&block, GgmlType::Q4_K).unwrap();
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            let expected = if i < 32 {
                16.0f32
            } else if i < 64 {
                8.0
            } else {
                0.0
            };
            assert!(
                (v - expected).abs() < 1e-4,
                "out[{i}]={v} esperado={expected}"
            );
        }
    }

    #[test]
    fn q4_k_bad_size_returns_error() {
        assert!(dequant_to_f32(&[0u8; 143], GgmlType::Q4_K).is_err());
    }

    fn make_q6_k_block(d: f32, scales: &[i8; 16], ql: &[u8; 128], qh: &[u8; 64]) -> Vec<u8> {
        let mut b = Vec::with_capacity(210);
        b.extend_from_slice(ql);
        b.extend_from_slice(qh);
        b.extend(scales.iter().map(|&s| s.cast_unsigned()));
        b.extend_from_slice(&f16_bytes(d));
        b
    }

    #[test]
    fn q6_k_all_zero_ql_qh() {
        // ql=zeros, qh=zeros → all 6-bit quants = 0-32 = -32
        // d=1.0, scales=[1,0,1,0,1,0,1,0, 1,0,1,0,1,0,1,0]
        // For l=0..15 (is=0): out=sc[0]*(-32)=-32; sc[2]*(-32)=-32; sc[4]*(-32)=-32; sc[6]*(-32)=-32
        // For l=16..31(is=1): out=sc[1]*(-32)=0; sc[3]*(-32)=0; sc[5]*(-32)=0; sc[7]*(-32)=0
        let scales: [i8; 16] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let ql = [0u8; 128];
        let qh = [0u8; 64];
        let block = make_q6_k_block(1.0, &scales, &ql, &qh);
        let out = dequant_to_f32(&block, GgmlType::Q6_K).unwrap();
        assert_eq!(out.len(), 256);
        for chunk_base in [0usize, 128] {
            for l in 0..16usize {
                assert!(
                    (out[chunk_base + l] - (-32.0)).abs() < 1e-4,
                    "chunk={chunk_base} l={l} off=0"
                );
                assert!(
                    (out[chunk_base + l + 32] - (-32.0)).abs() < 1e-4,
                    "chunk={chunk_base} l={l} off=32"
                );
                assert!(
                    (out[chunk_base + l + 64] - (-32.0)).abs() < 1e-4,
                    "chunk={chunk_base} l={l} off=64"
                );
                assert!(
                    (out[chunk_base + l + 96] - (-32.0)).abs() < 1e-4,
                    "chunk={chunk_base} l={l} off=96"
                );
            }
            for l in 16..32usize {
                assert_eq!(
                    out[chunk_base + l],
                    0.0,
                    "chunk={chunk_base} l={l} off=0 deve=0"
                );
                assert_eq!(
                    out[chunk_base + l + 32],
                    0.0,
                    "chunk={chunk_base} l={l} off=32 deve=0"
                );
                assert_eq!(
                    out[chunk_base + l + 64],
                    0.0,
                    "chunk={chunk_base} l={l} off=64 deve=0"
                );
                assert_eq!(
                    out[chunk_base + l + 96],
                    0.0,
                    "chunk={chunk_base} l={l} off=96 deve=0"
                );
            }
        }
    }

    #[test]
    fn q6_k_bad_size_returns_error() {
        assert!(dequant_to_f32(&[0u8; 209], GgmlType::Q6_K).is_err());
    }
}

#[cfg(test)]
mod q5k_tests {
    use super::*;

    // Primeiro superbloco de `token_embd.weight` do Qwen2.5-32B-Instruct-Q5_K_M, e os
    // valores correspondentes produzidos pelo `gguf-py` do llama.cpp — o oráculo do
    // formato. Fixture pequeno de propósito: pega qualquer erro de manipulação de bits
    // (nibble trocado, bit alto no grupo errado, escala mal desempacotada) sem depender
    // do modelo estar presente na máquina.
    const RAW: [u8; 176] = [
        60, 6, 128, 24, 217, 160, 160, 90, 213, 88, 91, 81, 255, 251, 235, 6, 222, 227, 191, 124,
        150, 29, 213, 59, 219, 255, 241, 30, 63, 145, 57, 124, 125, 249, 241, 221, 55, 146, 83,
        162, 124, 217, 252, 247, 247, 222, 117, 127, 55, 187, 228, 65, 208, 180, 35, 129, 127, 108,
        196, 99, 69, 252, 126, 159, 119, 251, 254, 205, 146, 62, 9, 159, 125, 184, 201, 15, 42, 8,
        10, 17, 214, 240, 97, 144, 166, 101, 29, 95, 52, 147, 233, 198, 44, 143, 174, 246, 7, 131,
        232, 20, 231, 252, 88, 238, 184, 91, 170, 8, 236, 187, 169, 30, 231, 32, 16, 75, 214, 171,
        246, 8, 230, 5, 204, 181, 104, 109, 93, 170, 55, 57, 28, 217, 103, 218, 220, 78, 37, 248,
        31, 99, 249, 10, 40, 33, 195, 21, 143, 105, 11, 176, 65, 126, 82, 68, 105, 177, 190, 92,
        217, 240, 193, 134, 65, 251, 28, 47, 3, 13, 216, 38, 161, 1, 31, 2, 149, 170,
    ];

    const SPOTS: [(usize, f32); 24] = [
        (0, -2.9495e-02),
        (1, 1.8069506e-02),
        (2, 1.4219284e-03),
        (3, -4.3764353e-02),
        (4, -4.6142578e-02),
        (5, 1.4219284e-03),
        (6, -9.562969e-04),
        (7, -5.7127476e-03),
        (124, -2.7265549e-03),
        (125, 2.9427052e-02),
        (126, -1.2619972e-02),
        (127, 4.693508e-03),
        (128, -5.8579445e-04),
        (129, -1.3842773e-01),
        (130, -4.253769e-02),
        (131, 2.3386717e-02),
        (248, -7.949352e-03),
        (249, 2.5148392e-03),
        (250, 1.9257545e-02),
        (251, -1.6708374e-03),
        (252, 4.2200089e-04),
        (253, -1.6708374e-03),
        (254, -1.6320705e-02),
        (255, -1.4227867e-02),
    ];

    /// Soma dos 256 valores, para detectar erro em posições fora das amostradas.
    const SOMA: f64 = 1.0147691e-01;

    #[test]
    fn q5_k_bate_com_o_oraculo_do_llama_cpp() {
        let got = dequant_to_f32(&RAW, GgmlType::Q5_K).expect("dequant Q5_K");
        assert_eq!(got.len(), 256, "um superbloco = 256 elementos");

        for (i, esperado) in SPOTS {
            let d = (got[i] - esperado).abs();
            assert!(
                d < 1e-6,
                "indice {i}: obtido {}, esperado {esperado}",
                got[i]
            );
        }

        let soma: f64 = got.iter().map(|&v| f64::from(v)).sum();
        assert!(
            (soma - SOMA).abs() < 1e-5,
            "soma dos 256 valores: obtida {soma:.9e}, esperada {SOMA:.9e}"
        );
    }

    #[test]
    fn q5_k_rejeita_tamanho_invalido() {
        assert!(dequant_to_f32(&RAW[..175], GgmlType::Q5_K).is_err());
    }
}

#[cfg(test)]
mod q3k_tests {
    use super::*;

    // Primeiro superbloco de `token_embd.weight` do Qwen3.8-35B-A3B (APEX-I-MiniPlus), e os
    // valores do `gguf-py` do llama.cpp para ele — o oráculo do formato.
    #[rustfmt::skip]
    const RAW: [u8; 110] = [
        245, 11, 119, 238, 241, 171, 184, 230, 152, 70, 252, 122, 234, 190, 169, 249, 76, 236, 207, 47,
        235, 205, 19, 198, 216, 93, 41, 55, 50, 155, 210, 255, 216, 4, 194, 4, 61, 50, 133, 182,
        63, 241, 192, 51, 47, 70, 126, 156, 119, 2, 68, 4, 56, 41, 172, 0, 3, 76, 116, 208,
        247, 32, 127, 36, 0, 59, 222, 0, 4, 1, 121, 65, 49, 79, 1, 192, 23, 120, 51, 53,
        202, 66, 29, 242, 1, 64, 190, 31, 14, 204, 48, 241, 240, 140, 12, 8, 245, 115, 230, 105,
        182, 196, 14, 65, 37, 213, 33, 254, 220, 141,
    ];

    const SPOTS: [(usize, f32); 24] = [
        (0, 0.0),
        (1, 0.0),
        (2, 0.007867813),
        (3, -0.015735626),
        (4, 0.0039339066),
        (5, 0.007867813),
        (6, -0.01180172),
        (7, -0.007867813),
        (124, 0.0060796738),
        (125, -0.0),
        (126, 0.018239021),
        (127, -0.0),
        (128, -0.0),
        (129, 0.005364418),
        (130, -0.010728836),
        (131, 0.021457672),
        (248, -0.0),
        (249, 0.0071525574),
        (250, 0.02861023),
        (251, 0.0071525574),
        (252, 0.0071525574),
        (253, -0.014305115),
        (254, -0.0),
        (255, -0.0),
    ];

    /// Soma dos 256 valores, para detectar erro em posições fora das amostradas.
    const SOMA: f64 = -0.031828880310058594;

    #[test]
    fn q3_k_bate_com_o_oraculo_do_llama_cpp() {
        let got = dequant_to_f32(&RAW, GgmlType::Q3_K).expect("dequant Q3_K");
        assert_eq!(got.len(), 256, "um superbloco = 256 elementos");
        for (i, esperado) in SPOTS {
            assert!(
                (got[i] - esperado).abs() < 1e-7,
                "indice {i}: obtido {}, esperado {esperado}",
                got[i]
            );
        }
        let soma: f64 = got.iter().map(|&v| f64::from(v)).sum();
        assert!(
            (soma - SOMA).abs() < 1e-6,
            "soma dos 256 valores: obtida {soma:.9e}, esperada {SOMA:.9e}"
        );
    }

    #[test]
    fn q3_k_rejeita_tamanho_invalido() {
        assert!(dequant_to_f32(&RAW[..RAW.len() - 1], GgmlType::Q3_K).is_err());
    }
}

#[cfg(test)]
mod iq3xxs_tests {
    use super::*;

    // Primeiro superbloco de `blk.10.ffn_down_exps.weight` do Qwen3.8-35B-A3B (APEX-I-MiniPlus), e os
    // valores do `gguf-py` do llama.cpp para ele — o oráculo do formato.
    #[rustfmt::skip]
    const RAW: [u8; 98] = [
        196, 4, 38, 121, 13, 134, 107, 81, 84, 148, 70, 110, 119, 196, 56, 63, 185, 128, 7, 199,
        27, 73, 148, 201, 60, 106, 248, 210, 151, 223, 156, 47, 47, 98, 132, 18, 47, 89, 246, 83,
        71, 109, 200, 109, 144, 195, 47, 13, 31, 247, 100, 130, 176, 147, 109, 243, 31, 182, 49, 47,
        52, 239, 199, 158, 123, 174, 102, 58, 22, 242, 97, 177, 42, 145, 183, 187, 223, 221, 206, 206,
        57, 174, 4, 130, 57, 241, 111, 73, 148, 121, 244, 137, 172, 150, 169, 111, 220, 154,
    ];

    const SPOTS: [(usize, f32); 24] = [
        (0, 0.011271238),
        (1, -0.006762743),
        (2, -0.015779734),
        (3, 0.0022542477),
        (4, 0.011271238),
        (5, -0.011271238),
        (6, -0.011271238),
        (7, 0.011271238),
        (124, -0.007635355),
        (125, -0.001527071),
        (126, -0.001527071),
        (127, 0.007635355),
        (128, 0.015779734),
        (129, 0.0022542477),
        (130, -0.024796724),
        (131, 0.011271238),
        (248, 0.0013816357),
        (249, -0.012434721),
        (250, -0.0069081783),
        (251, 0.0069081783),
        (252, -0.012434721),
        (253, 0.0013816357),
        (254, -0.0013816357),
        (255, 0.012434721),
    ];

    /// Soma dos 256 valores, para detectar erro em posições fora das amostradas.
    const SOMA: f64 = -0.0716632604598999;

    #[test]
    fn iq3_xxs_bate_com_o_oraculo_do_llama_cpp() {
        let got = dequant_to_f32(&RAW, GgmlType::IQ3_XXS).expect("dequant IQ3_XXS");
        assert_eq!(got.len(), 256, "um superbloco = 256 elementos");
        for (i, esperado) in SPOTS {
            assert!(
                (got[i] - esperado).abs() < 1e-7,
                "indice {i}: obtido {}, esperado {esperado}",
                got[i]
            );
        }
        let soma: f64 = got.iter().map(|&v| f64::from(v)).sum();
        assert!(
            (soma - SOMA).abs() < 1e-6,
            "soma dos 256 valores: obtida {soma:.9e}, esperada {SOMA:.9e}"
        );
    }

    #[test]
    fn iq3_xxs_rejeita_tamanho_invalido() {
        assert!(dequant_to_f32(&RAW[..RAW.len() - 1], GgmlType::IQ3_XXS).is_err());
    }
}
