//! Kernels f32 puros do forward Llama. Layout token-major: `x[t*dim + d]`.
#![allow(clippy::indexing_slicing)]

use crate::error::ModelError;
use rayon::prelude::*;
use wide::{bytemuck, f32x8};

/// GET_ROWS: para cada token, copia a linha de `embd` ({vocab, n_embd}).
/// Saída token-major [n_tok * n_embd].
pub(crate) fn embedding_lookup(
    embd: &[f32],
    tokens: &[u32],
    n_embd: usize,
) -> Result<Vec<f32>, ModelError> {
    let mut out = Vec::with_capacity(tokens.len() * n_embd);
    for &tok in tokens {
        let t = usize::try_from(tok).map_err(|_| ModelError::Overflow)?;
        let start = t * n_embd;
        let row = embd
            .get(start..start + n_embd)
            .ok_or_else(|| ModelError::Config(format!("token {t} fora do vocab")))?;
        out.extend_from_slice(row);
    }
    Ok(out)
}

/// RMSNorm por linha (sem peso): `x / sqrt(mean(x^2) + eps)`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn rmsnorm(x: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let ss: f32 = row_in.iter().map(|&v| v * v).sum();
        let scale = 1.0 / (ss / dim as f32 + eps).sqrt();
        for (o, &i) in row_out.iter_mut().zip(row_in.iter()) {
            *o = i * scale;
        }
    }
    out
}

/// Multiplicação elementwise por peso broadcast por dimensão (MUL com {dim}).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn mul_rows(x: &[f32], weight: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        for (idx, (o, &i)) in row_out.iter_mut().zip(row_in.iter()).enumerate() {
            *o = i * weight[idx];
        }
    }
    out
}

/// Otimização 5 — RMSNorm + escala elementwise em um único passe sobre os dados.
/// Equivale a `mul_rows(rmsnorm(x, dim, eps), weight, dim)` sem alocar buffer intermediário.
pub(crate) fn rmsnorm_and_scale(x: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let ss: f32 = row_in.iter().map(|&v| v * v).sum();
        let scale = 1.0 / (ss / dim as f32 + eps).sqrt();
        for (idx, (o, &i)) in row_out.iter_mut().zip(row_in.iter()).enumerate() {
            *o = i * scale * weight[idx];
        }
    }
    out
}

/// MUL_MAT: `W{in,out}` (out linhas de comprimento in) × `x` token-major [n_tok*in].
/// Saída token-major [n_tok*out]: `out[t*out+j] = Σ_i W[j*in+i] * x[t*in+i]`.
/// Threshold below which rayon thread-pool overhead exceeds benefit.
const PAR_MIN_N_OUT: usize = 512;
/// Lower threshold for Q8_0 actq path (kernel é mais rápido, vale paralelizar mais cedo).
const PAR_MIN_N_OUT_Q8: usize = 256;
/// Linhas por tarefa rayon. Valor menor aumenta paralelismo para matrizes menores
/// (ex.: attn_q n_out=896 com 28 threads: 64→14 tasks (metade idle), 32→28 tasks (todos ativos)).
const PAR_CHUNK: usize = 32;

pub(crate) fn matmul(w: &[f32], x: &[f32], n_in: usize, n_out: usize, n_tok: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tok * n_out];
    for t in 0..n_tok {
        let xrow = &x[t * n_in..(t + 1) * n_in];
        let orow = &mut out[t * n_out..(t + 1) * n_out];
        if n_out >= PAR_MIN_N_OUT {
            orow.par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    for (k, o) in chunk.iter_mut().enumerate() {
                        let j = ci * PAR_CHUNK + k;
                        let wrow = &w[j * n_in..(j + 1) * n_in];
                        *o = wrow.iter().zip(xrow.iter()).map(|(&a, &b)| a * b).sum();
                    }
                });
        } else {
            for (j, o) in orow.iter_mut().enumerate() {
                let wrow = &w[j * n_in..(j + 1) * n_in];
                *o = wrow.iter().zip(xrow.iter()).map(|(&a, &b)| a * b).sum();
            }
        }
    }
    out
}

/// MUL_MAT direto em bytes Q8_0 — sem expandir para f32.
///
/// Layout W: `n_out` linhas × `(n_in/32)` blocos × 34 bytes/bloco.
/// Bloco: 2 bytes (f16 LE scale `d`) + 32 bytes (i8 quants).
/// `n_in` deve ser múltiplo de 32 (garantido pelo formato GGUF).
pub(crate) fn matmul_q8_0(
    w: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    n_tok: usize,
) -> Vec<f32> {
    const Q: usize = 32;
    const B: usize = 34;
    debug_assert_eq!(n_in % Q, 0, "n_in deve ser múltiplo de 32");

    let n_blocks = n_in / Q;
    let row_bytes = n_blocks * B;
    let mut out = vec![0.0f32; n_tok * n_out];

    for t in 0..n_tok {
        let x_row = &x[t * n_in..(t + 1) * n_in];
        let o_row = &mut out[t * n_out..(t + 1) * n_out];

        if n_out >= PAR_MIN_N_OUT {
            o_row
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    for (k, o) in chunk.iter_mut().enumerate() {
                        let j = ci * PAR_CHUNK + k;
                        *o = q8_0_dot(&w[j * row_bytes..(j + 1) * row_bytes], x_row, n_blocks);
                    }
                });
        } else {
            for (j, o) in o_row.iter_mut().enumerate() {
                *o = q8_0_dot(&w[j * row_bytes..(j + 1) * row_bytes], x_row, n_blocks);
            }
        }
    }
    out
}

/// Escala f16 little-endian de um bloco Q8_0.
#[inline(always)]
fn escala(d: [u8; 2]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes(d)).to_f32()
}

/// Separa um bloco Q8_0 (34 bytes) em escala e quants.
#[inline(always)]
fn bloco_q8_0(b: &[u8; 34]) -> (f32, &[u8; 32]) {
    let [d0, d1, qs @ ..] = b;
    (escala([*d0, *d1]), qs)
}

/// Σ qw·qx dos 32 quants i8 em 8 somas parciais, já em f32. Com AVX2 são as mesmas
/// instruções do kernel `std::arch` que isto substituiu (`vpabsb`/`vpsignb`/`vpmaddubsw`/
/// `vpmaddwd`), pela API segura do `safe_arch`. Prova de não-saturação do `vpmaddubsw`:
/// max par = 128×127+128×127 = 32512 < 32767.
#[cfg(target_feature = "avx2")]
#[inline(always)]
fn dot_parcial(qw: &[u8; 32], qx: &[u8; 32]) -> f32x8 {
    use safe_arch::*;
    let (w, x) = (m256i::from(*qw), m256i::from(*qx));
    let pares =
        mul_u8i8_add_horizontal_saturating_m256i(abs_i8_m256i(w), sign_apply_i8_m256i(x, w));
    let soma = mul_i16_horizontal_add_m256i(pares, set_splat_i16_m256i(1));
    bytemuck::cast(convert_to_m256_from_i32_m256i(soma))
}

/// Sem AVX2: estende para i16 e usa o `dot` do `wide` (que cai para SSE ou escalar).
#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
fn dot_parcial(qw: &[u8; 32], qx: &[u8; 32]) -> f32x8 {
    use wide::{i8x16, i16x16};
    let i16s = |q: &[u8; 32]| -> [i16x16; 2] {
        let [a, b]: [i8x16; 2] = bytemuck::cast(*q);
        [i16x16::from_i8x16(a), i16x16::from_i8x16(b)]
    };
    let (w, x) = (i16s(qw), i16s(qx));
    f32x8::from_i32x8(w[0].dot(x[0]) + w[1].dot(x[1]))
}

/// Produto escalar Q8_0 × f32. Oito acumuladores independentes: soma de f32 não é
/// associativa, e com um acumulador só o LLVM não vetoriza a redução.
#[inline]
fn q8_0_dot(w_row: &[u8], x_row: &[f32], n_blocks: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    let blocos = w_row
        .as_chunks::<34>()
        .0
        .iter()
        .zip(x_row.as_chunks::<32>().0);
    for (bw, xb) in blocos.take(n_blocks) {
        let (d, qs) = bloco_q8_0(bw);
        for (q8, x8) in qs.as_chunks::<8>().0.iter().zip(xb.as_chunks::<8>().0) {
            for ((a, &q), &x) in acc.iter_mut().zip(q8).zip(x8) {
                *a += d * f32::from(q.cast_signed()) * x;
            }
        }
    }
    acc.iter().sum()
}

/// Otimização 2a — quantiza um vetor f32 para Q8_0.
/// Saída: n_in/32 blocos × 34 bytes (2 bytes f16 scale + 32 bytes i8).
fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
    let blocos = x.as_chunks::<32>().0;
    let mut out = vec![0u8; blocos.len() * 34];
    for (blk, dst) in blocos.iter().zip(out.as_chunks_mut::<34>().0) {
        let v: [f32x8; 4] = bytemuck::cast(*blk);
        let max_abs = v
            .iter()
            .fold(f32x8::ZERO, |m, x| m.max(x.abs()))
            .to_array()
            .into_iter()
            .fold(0.0f32, f32::max);
        let d = max_abs / 127.0;
        let d_inv = f32x8::splat(if d > 0.0 { 1.0 / d } else { 0.0 });
        let [d0, d1, qs @ ..] = dst;
        [*d0, *d1] = half::f16::from_f32(d).to_bits().to_le_bytes();
        for (q8, x) in qs.as_chunks_mut::<8>().0.iter_mut().zip(v) {
            // `round_int` é o `cvtps2dq` (ties-even) do kernel AVX2 que esta função
            // substituiu. |x·d_inv| <= 127 por construção: o cast para i8 não trunca.
            #[allow(clippy::cast_possible_truncation)]
            let i = (x * d_inv)
                .round_int()
                .to_array()
                .map(|i| (i as i8).cast_unsigned());
            *q8 = i;
        }
    }
    out
}

/// Otimização 2b — quantiza x token-major [n_tok × n_in] para Q8_0 (token a token).
pub(crate) fn quantize_q8_0_split(x: &[f32], n_in: usize, n_tok: usize) -> Vec<u8> {
    const Q: usize = 32;
    const B: usize = 34;
    let n_blocks = n_in / Q;
    let tok_bytes = n_blocks * B;
    let mut out = Vec::with_capacity(n_tok * tok_bytes);
    for t in 0..n_tok {
        let row = &x[t * n_in..(t + 1) * n_in];
        out.extend(quantize_q8_0(row));
    }
    out
}

/// Otimização 2c — produto escalar Q8_0 × Q8_0 → f32.
#[inline]
fn q8_0_q8_0_dot(w_row: &[u8], x_q8_row: &[u8], n_blocks: usize) -> f32 {
    let blocos = w_row
        .as_chunks::<34>()
        .0
        .iter()
        .zip(x_q8_row.as_chunks::<34>().0);
    blocos
        .take(n_blocks)
        .fold(f32x8::ZERO, |acc, (bw, bx)| {
            let (dw, qw) = bloco_q8_0(bw);
            let (dx, qx) = bloco_q8_0(bx);
            dot_parcial(qw, qx).mul_add(f32x8::splat(dw * dx), acc)
        })
        .reduce_add()
}

/// Recompacta pesos Q8_0 de row-major para block_q8_0x8.
/// Por grupo de 8 linhas, por bloco b:
///   bytes 0..16   : 8 escalas f16 (d0..d7)
///   bytes 16..272 : 8 × 32 quants (qs0[32] ++ qs1[32] ++ ... ++ qs7[32])
///   total = 272 bytes por superbloco.
/// Exige n_out múltiplo de 8; n_in múltiplo de 32.
pub(crate) fn repack_q8_0_8rows(w: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
    const Q: usize = 32;
    const B: usize = 34;
    const PB: usize = 272; // 16 + 256
    debug_assert_eq!(n_out % 8, 0);
    debug_assert_eq!(n_in % Q, 0);
    let n_blocks = n_in / Q;
    let row_bytes = n_blocks * B;
    let n_groups = n_out / 8;
    let mut out = vec![0u8; n_groups * n_blocks * PB];
    for g in 0..n_groups {
        let base_row = g * 8;
        for b in 0..n_blocks {
            let dst = (g * n_blocks + b) * PB;
            // 8 escalas f16
            for j in 0..8 {
                let src = (base_row + j) * row_bytes + b * B;
                out[dst + j * 2] = w[src];
                out[dst + j * 2 + 1] = w[src + 1];
            }
            // 8 × 32 quants
            for j in 0..8 {
                let src = (base_row + j) * row_bytes + b * B + 2;
                let d = dst + 16 + j * Q;
                out[d..d + Q].copy_from_slice(&w[src..src + Q]);
            }
        }
    }
    out
}

/// Oito linhas em block_q8_0x8 (ver `repack_q8_0_8rows`) contra uma linha de ativações
/// Q8_0: os quants de x são lidos uma vez por bloco e servem às oito linhas.
fn dot_8linhas_repacked(grupo: &[u8], x_row: &[u8]) -> [f32; 8] {
    let mut acc = [f32x8::ZERO; 8];
    let blocos = grupo.as_chunks::<272>().0;
    for (b, (p, bx)) in blocos.iter().zip(x_row.as_chunks::<34>().0).enumerate() {
        // Prefetch do bloco b+2, como o kernel `std::arch` que isto substituiu.
        #[cfg(target_feature = "sse")]
        if let Some(prox) = blocos.get(b + 2) {
            prox.as_chunks::<64>()
                .0
                .iter()
                .for_each(safe_arch::prefetch_t0);
        }
        let (dx, qx) = bloco_q8_0(bx);
        let (escalas, quants) = p.split_at(16);
        let linhas = escalas
            .as_chunks::<2>()
            .0
            .iter()
            .zip(quants.as_chunks::<32>().0);
        for (a, (d, qw)) in acc.iter_mut().zip(linhas) {
            *a = dot_parcial(qw, qx).mul_add(f32x8::splat(escala(*d) * dx), *a);
        }
    }
    acc.map(f32x8::reduce_add)
}

/// Otimização 2d — MUL_MAT Q8_0 com ativações também quantizadas (i8×i8 mais rápido que f32×i8).
/// `x_q8`: saída de `quantize_q8_0_split(x, n_in, n_tok)`.
pub(crate) fn matmul_q8_0_actq(
    w: &[u8],
    x_q8: &[u8],
    n_in: usize,
    n_out: usize,
    n_tok: usize,
) -> Vec<f32> {
    const Q: usize = 32;
    const B: usize = 34;
    let n_blocks = n_in / Q;
    let row_bytes = n_blocks * B;
    let mut out = vec![0.0f32; n_tok * n_out];
    for t in 0..n_tok {
        let x_row = &x_q8[t * row_bytes..(t + 1) * row_bytes];
        let o_row = &mut out[t * n_out..(t + 1) * n_out];
        let linha = |(j, o): (usize, &mut f32)| {
            *o = q8_0_q8_0_dot(&w[j * row_bytes..(j + 1) * row_bytes], x_row, n_blocks);
        };
        if n_out >= PAR_MIN_N_OUT_Q8 {
            o_row.par_iter_mut().enumerate().for_each(linha);
        } else {
            o_row.iter_mut().enumerate().for_each(linha);
        }
    }
    out
}

/// Variante de matmul_q8_0_actq que usa pesos pré-empacotados em block_q8_0x8.
/// packed_w: saída de repack_q8_0_8rows (cobre as primeiras n_out_packed linhas, múltiplo de 8).
/// w_tail: bytes raw das linhas restantes (n_out - n_out_packed), pode ser vazio.
pub(crate) fn matmul_q8_0_actq_packed(
    packed_w: &[u8],
    w_tail: &[u8],
    x_q8: &[u8],
    n_in: usize,
    n_out: usize,
    n_out_packed: usize,
    n_tok: usize,
) -> Vec<f32> {
    const Q: usize = 32;
    const B: usize = 34;
    const PB: usize = 272;
    let n_blocks = n_in / Q;
    let row_bytes = n_blocks * B;
    let group_bytes = n_blocks * PB;
    let mut out = vec![0.0f32; n_tok * n_out];

    // Static work division: one contiguous range per thread (same as llamafile).
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = {
        let raw = n_out_packed.max(1).div_ceil(n_threads);
        raw.max(8).div_ceil(8) * 8
    };

    for t in 0..n_tok {
        let x_row = &x_q8[t * row_bytes..(t + 1) * row_bytes];
        let o_row = &mut out[t * n_out..(t + 1) * n_out];
        let (o_packed, o_tail) = o_row.split_at_mut(n_out_packed);

        o_packed
            .par_chunks_mut(chunk_size)
            .zip(packed_w.par_chunks(chunk_size / 8 * group_bytes))
            .for_each(|(saida, grupos)| {
                for (s, g) in saida
                    .as_chunks_mut::<8>()
                    .0
                    .iter_mut()
                    .zip(grupos.chunks_exact(group_bytes))
                {
                    *s = dot_8linhas_repacked(g, x_row);
                }
            });

        // Tail rows (not multiple of 8)
        for (j, o) in o_tail.iter_mut().enumerate() {
            *o = q8_0_q8_0_dot(&w_tail[j * row_bytes..(j + 1) * row_bytes], x_row, n_blocks);
        }
    }
    out
}

/// Otimização 3 — RoPE NORM (ou NEOX, com `neox`) com tabela de frequências pré-computada.
/// `freq_table[i] = freq_base^(-2i/rope_dim)` para i em 0..rope_dim/2.
/// Elimina `powf` no loop interno (era a operação mais cara da RoPE).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_norm(
    x: &mut [f32],
    n_tok: usize,
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_table: &[f32],
    pos0: usize,
    neox: bool,
) {
    let meia = rope_dim / 2;
    for t in 0..n_tok {
        let pos = (pos0 + t) as f32;
        for h in 0..n_head {
            let base = (t * n_head + h) * head_dim;
            for (i, &freq) in freq_table.iter().enumerate().take(meia) {
                let theta = pos * freq;
                let (s, c) = theta.sin_cos();
                let (ia, ib) = if neox {
                    (base + i, base + i + meia)
                } else {
                    (base + 2 * i, base + 2 * i + 1)
                };
                let a = x[ia];
                let b = x[ib];
                x[ia] = a * c - b * s;
                x[ib] = a * s + b * c;
            }
        }
    }
}

/// SiLU: `x * sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SWIGLU: `silu(gate) * up`, elementwise (mesmo comprimento).
pub(crate) fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu(g) * u)
        .collect()
}

/// Softmax numericamente estável sobre um slice (in-place).
pub(crate) fn softmax(z: &mut [f32]) {
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in z.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in z.iter_mut() {
            *v /= sum;
        }
    }
}

/// Índice do maior valor (greedy / argmax). Empate → menor índice.
pub(crate) fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_lookup_copies_rows() {
        // vocab=3, n_embd=2
        let embd = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = embedding_lookup(&embd, &[2, 0], 2).unwrap();
        assert_eq!(out, vec![4.0, 5.0, 0.0, 1.0]);
    }

    #[test]
    fn embedding_lookup_rejects_oob_token() {
        let embd = vec![0.0, 1.0];
        assert!(embedding_lookup(&embd, &[5], 2).is_err());
    }

    #[test]
    fn rmsnorm_unit_vector() {
        // x = [3,4], dim=2, eps=0 → mean(x^2)=12.5 → scale=1/sqrt(12.5)
        let out = rmsnorm(&[3.0, 4.0], 2, 0.0);
        let s = 1.0 / 12.5f32.sqrt();
        assert!((out[0] - 3.0 * s).abs() < 1e-6);
        assert!((out[1] - 4.0 * s).abs() < 1e-6);
    }

    #[test]
    fn mul_rows_broadcasts_weight() {
        // 2 linhas dim=2, peso [10,100]
        let out = mul_rows(&[1.0, 2.0, 3.0, 4.0], &[10.0, 100.0], 2);
        assert_eq!(out, vec![10.0, 200.0, 30.0, 400.0]);
    }

    #[test]
    fn rmsnorm_and_scale_matches_rmsnorm_plus_mul_rows() {
        // Verifica equivalência matemática com dois passos separados
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 tokens, dim=2
        let weight = vec![0.5f32, 2.0];
        let eps = 1e-5;
        let dim = 2;
        let expected = mul_rows(&rmsnorm(&x, dim, eps), &weight, dim);
        let got = rmsnorm_and_scale(&x, &weight, dim, eps);
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-6, "divergência: {a} vs {b}");
        }
    }

    #[test]
    fn matmul_2x2_identity_and_general() {
        // W in=2,out=2: linha0=[1,0], linha1=[0,1] (identidade) → out=x
        let w_id = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![5.0, 7.0]; // 1 token
        assert_eq!(matmul(&w_id, &x, 2, 2, 1), vec![5.0, 7.0]);

        // W in=2,out=1: linha0=[2,3] → out[0]=2*5+3*7=31
        let w = vec![2.0, 3.0];
        assert_eq!(matmul(&w, &x, 2, 1, 1), vec![31.0]);
    }

    #[test]
    fn matmul_two_tokens() {
        // W in=2,out=2: linha0=[1,1], linha1=[1,-1]
        let w = vec![1.0, 1.0, 1.0, -1.0];
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens
        // t0: [1+2, 1-2]=[3,-1]; t1:[3+4,3-4]=[7,-1]
        assert_eq!(matmul(&w, &x, 2, 2, 2), vec![3.0, -1.0, 7.0, -1.0]);
    }

    #[test]
    fn rope_norm_pos_zero_is_identity() {
        // pos=0 → θ=0 → cos=1,sin=0 → sem mudança
        let mut x = vec![1.0, 2.0, 3.0, 4.0]; // 1 tok, 1 head, head_dim=4, rope_dim=4
        // freq_table para rope_dim=4: [base^0, base^(-2/4)] = [1.0, 1/100.0]
        let freq_table: Vec<f32> = (0..2)
            .map(|i| 10000.0f32.powf(-2.0 * i as f32 / 4.0))
            .collect();
        rope_norm(&mut x, 1, 1, 4, 4, &freq_table, 0, false);
        assert!(
            x.iter()
                .zip([1.0, 2.0, 3.0, 4.0])
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
    }

    #[test]
    fn rope_norm_pos_one_rotates_first_pair_by_one_radian() {
        // i=0 → θ = 1 * base^0 = 1 rad. par (x0,x1)=(1,0) → (cos1, sin1)
        let mut x = vec![1.0, 0.0, 0.0, 0.0];
        let freq_table: Vec<f32> = (0..2)
            .map(|i| 10000.0f32.powf(-2.0 * i as f32 / 4.0))
            .collect();
        rope_norm(&mut x, 1, 1, 4, 4, &freq_table, 1, false);
        assert!((x[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((x[1] - 1.0f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn swiglu_matches_manual() {
        // gate=[0,2], up=[1,3]; silu(0)=0, silu(2)=2*sigmoid(2)
        let out = swiglu(&[0.0, 2.0], &[1.0, 3.0]);
        assert!((out[0] - 0.0).abs() < 1e-6);
        let silu2 = 2.0 / (1.0 + (-2.0f32).exp());
        assert!((out[1] - silu2 * 3.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut z = vec![1.0, 2.0, 3.0];
        softmax(&mut z);
        assert!((z.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(z[2] > z[1] && z[1] > z[0]);
    }

    #[test]
    fn argmax_picks_first_max() {
        assert_eq!(argmax(&[0.1, 0.9, 0.9, 0.2]), 1);
    }

    #[test]
    fn matmul_q8_0_matches_manual() {
        // 2 rows × 1 block Q8_0 (n_in=32, n_out=2, n_tok=1)
        // row 0: d=1.0, qs=[1,2,0×30] → dot(x) = 1×1 + 2×2 = 5 → out=5.0
        // row 1: d=2.0, qs=[1,0×31]   → dot(x) = 1×1       = 1 → out=2.0
        fn f16_le(v: f32) -> [u8; 2] {
            half::f16::from_f32(v).to_bits().to_le_bytes()
        }
        let mut w = Vec::with_capacity(68);
        w.extend_from_slice(&f16_le(1.0));
        w.push(1u8);
        w.push(2u8);
        w.extend(std::iter::repeat_n(0u8, 30));
        w.extend_from_slice(&f16_le(2.0));
        w.push(1u8);
        w.extend(std::iter::repeat_n(0u8, 31));

        let x: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let out = matmul_q8_0(&w, &x, 32, 2, 1);

        assert!((out[0] - 5.0).abs() < 1e-4, "out[0]={}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-4, "out[1]={}", out[1]);
    }

    #[test]
    fn matmul_q8_0_two_tokens() {
        // n_in=32, n_out=1, n_tok=2; d=1.0, qs=[1,0×31]
        fn f16_le(v: f32) -> [u8; 2] {
            half::f16::from_f32(v).to_bits().to_le_bytes()
        }
        let mut w = Vec::with_capacity(34);
        w.extend_from_slice(&f16_le(1.0));
        w.push(1u8);
        w.extend(std::iter::repeat_n(0u8, 31));

        let mut x = vec![0.0f32; 64];
        x[0] = 3.0; // token 0, dim 0
        x[32] = 7.0; // token 1, dim 0

        let out = matmul_q8_0(&w, &x, 32, 1, 2);

        assert!((out[0] - 3.0).abs() < 1e-4, "t0={}", out[0]);
        assert!((out[1] - 7.0).abs() < 1e-4, "t1={}", out[1]);
    }

    #[test]
    fn matmul_q8_0_actq_matches_matmul_q8_0() {
        // Verifica que actq produz resultados próximos ao q8_0 original
        // (pequena diferença de quantização de ativações é esperada)
        fn f16_le(v: f32) -> [u8; 2] {
            half::f16::from_f32(v).to_bits().to_le_bytes()
        }
        let mut w = Vec::with_capacity(68);
        w.extend_from_slice(&f16_le(1.0));
        w.push(1u8);
        w.push(2u8);
        w.extend(std::iter::repeat_n(0u8, 30));
        w.extend_from_slice(&f16_le(2.0));
        w.push(1u8);
        w.extend(std::iter::repeat_n(0u8, 31));

        let x: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let expected = matmul_q8_0(&w, &x, 32, 2, 1);
        let x_q8 = quantize_q8_0_split(&x, 32, 1);
        let got = matmul_q8_0_actq(&w, &x_q8, 32, 2, 1);

        // Tolerância maior (1%) devido ao ruído de quantização das ativações
        assert!(
            (got[0] - expected[0]).abs() / expected[0].abs() < 0.01,
            "out[0]: got={} expected={}",
            got[0],
            expected[0]
        );
        assert!(
            (got[1] - expected[1]).abs() / expected[1].abs() < 0.01,
            "out[1]: got={} expected={}",
            got[1],
            expected[1]
        );
    }
}

/// Bancada dos kernels Q8_0 de CPU (mediana de 50 execuções, em µs).
/// `numactl --cpunodebind=0 --membind=0 cargo test --release -p llama-model bancada_cpu -- --ignored --nocapture`
#[cfg(test)]
mod bancada {
    use super::*;
    use std::time::Instant;

    /// Linhas Q8_0 válidas (escala f16 pequena + quants pseudoaleatórios).
    fn pesos_q8_0(n_linhas: usize, n_in: usize, semente: u64) -> Vec<u8> {
        let mut s = semente;
        let mut out = Vec::with_capacity(n_linhas * n_in / 32 * 34);
        for _ in 0..n_linhas * n_in / 32 {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            out.extend_from_slice(
                &half::f16::from_f32(0.001 + (s >> 40) as f32 * 1e-9)
                    .to_bits()
                    .to_le_bytes(),
            );
            for _ in 0..32 {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                out.push((s >> 56) as u8);
            }
        }
        out
    }

    fn mediana_us(mut f: impl FnMut() -> f32) -> (f64, f32) {
        let mut chk = 0.0;
        for _ in 0..5 {
            chk = f();
        }
        let mut t: Vec<f64> = (0..50)
            .map(|_| {
                let i = Instant::now();
                chk = std::hint::black_box(f());
                i.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        t.sort_by(f64::total_cmp);
        (t[25], chk)
    }

    #[test]
    #[ignore = "bancada: rodar em --release"]
    fn bancada_cpu() {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(7)
            .build_global();
        for &(n_in, n_out, n_tok) in &[(4096, 4096, 1), (4096, 14336, 1), (4096, 4096, 32)] {
            let w = pesos_q8_0(n_out, n_in, 7);
            let x: Vec<f32> = (0..n_in * n_tok)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01)
                .collect();
            let xq = quantize_q8_0_split(&x, n_in, n_tok);
            let packed = repack_q8_0_8rows(&w, n_in, n_out);
            let soma = |v: Vec<f32>| v.iter().sum::<f32>();
            let (q, _) = mediana_us(|| quantize_q8_0_split(&x, n_in, n_tok).len() as f32);
            let (a, ca) = mediana_us(|| soma(matmul_q8_0(&w, &x, n_in, n_out, n_tok)));
            let (b, cb) = mediana_us(|| soma(matmul_q8_0_actq(&w, &xq, n_in, n_out, n_tok)));
            let (c, cc) = mediana_us(|| {
                soma(matmul_q8_0_actq_packed(
                    &packed,
                    &[],
                    &xq,
                    n_in,
                    n_out,
                    n_out,
                    n_tok,
                ))
            });
            println!(
                "{n_in}x{n_out} tok={n_tok}: quantize {q:.1} | q8_0×f32 {a:.1} ({ca:.4}) | actq {b:.1} ({cb:.4}) | packed {c:.1} ({cc:.4})"
            );
        }
    }
}
