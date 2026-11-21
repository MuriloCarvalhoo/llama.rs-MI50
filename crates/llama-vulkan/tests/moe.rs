//! Kernels do MoE do qwen35moe contra a referência de CPU (`ggml_cpu::dequant_to_f32`, que
//! bate bit a bit com o gguf-py): Q3_K denso e o matvec dos experts em Q3_K e IQ3_XXS.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use gguf::GgmlType;

/// Ativação como o shader a enxerga: quantizada em int8 por bloco de 32 e de volta.
fn quant_dequant_x(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for b in 0..x.len() / 32 {
        let blk = &x[b * 32..b * 32 + 32];
        let d_x = blk.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
        let inv = if d_x > 0.0 { 1.0 / d_x } else { 0.0 };
        for i in 0..32 {
            out[b * 32 + i] = (blk[i] * inv).round().clamp(-127.0, 127.0) * d_x;
        }
    }
    out
}

/// Bytes pseudo-aleatórios com um `d` f16 razoável no lugar certo de cada bloco.
fn pesos(ty: GgmlType, n_blocos: usize) -> Vec<u8> {
    let bloco = ty.type_size() as usize;
    let mut w: Vec<u8> = (0..n_blocos * bloco)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    let d = half::f16::from_f32(0.0071).to_le_bytes();
    for b in 0..n_blocos {
        let o = b * bloco;
        match ty {
            GgmlType::Q3_K => w[o + 108..o + 110].copy_from_slice(&d),
            _ => w[o..o + 2].copy_from_slice(&d),
        }
    }
    w
}

fn ativ(n: usize, semente: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + semente * 13) % 37) as f32 - 18.0) * 0.027)
        .collect()
}

fn cpu_matvec(ty: GgmlType, w: &[u8], x: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let row = n_in / 256 * ty.type_size() as usize;
    let xq = quant_dequant_x(x);
    (0..n_out)
        .map(|r| {
            let deq = ggml_cpu::dequant_to_f32(&w[r * row..(r + 1) * row], ty).unwrap();
            deq.iter().zip(&xq).map(|(a, b)| a * b).sum()
        })
        .collect()
}

fn erro_rel(gpu: &[f32], cpu: &[f32]) -> f32 {
    let max_abs = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
    gpu.iter()
        .zip(cpu)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs() / max_abs))
}

fn com_gpu(
    f: impl FnOnce(
        &llama_vulkan::VulkanContext,
        &llama_vulkan::VulkanPhysicalDevice,
        &llama_vulkan::VulkanDevice,
    ),
) {
    let Ok(ctx) = llama_vulkan::VulkanContext::new() else {
        return;
    };
    let phys = ctx.amd_compute_devices();
    let Some(p) = phys.first() else {
        return;
    };
    let dev = llama_vulkan::VulkanDevice::create(&ctx, p).unwrap();
    f(&ctx, p, &dev);
}

#[test]
fn q3_k_matvec_bate_com_a_cpu() {
    com_gpu(|ctx, phys, dev| {
        for (n_in, cols) in [(512usize, 1usize), (2048, 1), (4096, 4), (2048, 32)] {
            let n_out = 24;
            let w = pesos(GgmlType::Q3_K, n_out * n_in / 256);
            let x = ativ(n_in * cols, 1);
            let gpu = llama_vulkan::matmul::dispatch_q3_k_matvec(
                ctx, phys, dev, &w, &x, n_in, n_out, cols,
            )
            .unwrap();
            for c in 0..cols {
                let cpu = cpu_matvec(
                    GgmlType::Q3_K,
                    &w,
                    &x[c * n_in..(c + 1) * n_in],
                    n_in,
                    n_out,
                );
                let e = erro_rel(&gpu[c * n_out..(c + 1) * n_out], &cpu);
                eprintln!("Q3_K n_in={n_in} col={c}: erro rel {e:.2e}");
                assert!(e < 1e-5, "Q3_K divergiu: {e}");
            }
        }
    });
}

/// Os dois usos do kernel: gate/up (os slots de um token leem a mesma ativação) e down
/// (cada slot tem a sua). Dois tokens, dois experts por token, de oito.
fn moe_caso(ty: GgmlType, x_por_slot: bool) {
    com_gpu(|ctx, phys, dev| {
        let (n_expert, n_usados, n_tok) = (8usize, 2usize, 2usize);
        let (n_in, n_out) = if x_por_slot { (512, 64) } else { (2048, 40) };
        let w = pesos(ty, n_expert * n_out * n_in / 256);
        let ids: Vec<u32> = vec![5, 2, 7, 5];
        let n_cols = if x_por_slot { n_usados * n_tok } else { n_tok };
        let x = ativ(n_in * n_cols, 3);
        let gpu = llama_vulkan::matmul::dispatch_moe_matvec(
            ctx, phys, dev, ty, &w, &x, &ids, n_in, n_out, n_usados, x_por_slot,
        )
        .unwrap();
        let por_expert = n_out * n_in / 256 * ty.type_size() as usize;
        for (s, &e) in ids.iter().enumerate() {
            let col = if x_por_slot { s } else { s / n_usados };
            let we = &w[e as usize * por_expert..(e as usize + 1) * por_expert];
            let cpu = cpu_matvec(ty, we, &x[col * n_in..(col + 1) * n_in], n_in, n_out);
            let err = erro_rel(&gpu[s * n_out..(s + 1) * n_out], &cpu);
            eprintln!("{ty:?} slot {s} (expert {e}, x_por_slot={x_por_slot}): erro rel {err:.2e}");
            assert!(err < 1e-5, "{ty:?} slot {s} divergiu: {err}");
        }
    });
}

#[test]
fn moe_q3_k_bate_com_a_cpu() {
    moe_caso(GgmlType::Q3_K, false);
    moe_caso(GgmlType::Q3_K, true);
}

#[test]
fn moe_iq3_xxs_bate_com_a_cpu() {
    moe_caso(GgmlType::IQ3_XXS, false);
    moe_caso(GgmlType::IQ3_XXS, true);
}

/// O GEMM do prefill (`mul_mm.comp`, `TIPO = 3`) contra a CPU, nas larguras do bloco.
#[test]
fn gemm_q3_k_bate_com_a_cpu() {
    com_gpu(|ctx, phys, dev| {
        for (n_in, n_out, cols) in [(2048usize, 200usize, 32usize), (4096, 128, 8)] {
            let w = pesos(GgmlType::Q3_K, n_out * n_in / 256);
            let x = ativ(n_in * cols, 5);
            let gpu = llama_vulkan::matmul::dispatch_mul_mm_q3k(
                ctx, phys, dev, &w, &x, n_in, n_out, cols,
            )
            .unwrap();
            for c in 0..cols {
                let cpu = cpu_matvec(
                    GgmlType::Q3_K,
                    &w,
                    &x[c * n_in..(c + 1) * n_in],
                    n_in,
                    n_out,
                );
                let e = erro_rel(&gpu[c * n_out..(c + 1) * n_out], &cpu);
                assert!(e < 1e-5, "GEMM Q3_K n_in={n_in} col={c} divergiu: {e}");
            }
            eprintln!("GEMM Q3_K n_in={n_in} cols={cols}: ok");
        }
    });
}
