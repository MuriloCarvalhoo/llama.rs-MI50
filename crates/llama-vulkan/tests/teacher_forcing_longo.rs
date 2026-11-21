//! Teacher forcing longo contra o llama.cpp: o prompt e a continuação greedy vêm do
//! llama.cpp, e o llama.rs grava o top-8 de logprobs de cada passo para comparar.
//!
//! Uso: `TF_MODELO=<gguf> TF_ENTRADA=<arq> TF_SAIDA=<arq> cargo test --release -p llama-vulkan
//! --test teacher_forcing_longo -- --ignored --nocapture`. A entrada tem duas linhas de ids
//! separados por espaço: o prompt e a continuação. A saída tem uma linha por passo, com
//! pares `id:logprob`.

use llama_model::GpuResidentDecode;
use std::fmt::Write as _;

fn top8(logits: &[f32]) -> String {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let lse = m + logits.iter().map(|&x| (x - m).exp()).sum::<f32>().ln();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let mut s = String::new();
    for &i in idx.iter().take(8) {
        let _ = write!(s, "{i}:{:.5} ", logits[i] - lse);
    }
    s
}

/// O mesmo corte em blocos do `prefill_residente` (que é pub(crate)), do zero.
fn prefill(gpu: &llama_vulkan::LayerSplitForward, ids: &[u32]) -> (Vec<f32>, usize) {
    gpu.reset();
    let mut tamanhos = vec![gpu.batch_size()];
    tamanhos.extend(gpu.tamanhos_resto());
    let mut blocos: Vec<&[u32]> = Vec::new();
    let mut pos = 0usize;
    for nb in tamanhos {
        while nb > 1 && ids.len() - pos >= nb && gpu.lote_cabe(pos, nb) {
            blocos.push(&ids[pos..pos + nb]);
            pos += nb;
        }
    }
    let mut logits = if blocos.is_empty() {
        Vec::new()
    } else {
        gpu.prefill_blocos(&blocos, 0).unwrap()
    };
    for &t in &ids[pos..] {
        logits = gpu.decode(t, pos).unwrap();
        pos += 1;
    }
    (logits, pos)
}

#[test]
#[ignore = "precisa do modelo e da referência do llama.cpp"]
fn teacher_forcing_contra_o_llama_cpp() {
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("falta {k}"));
    let entrada = std::fs::read_to_string(var("TF_ENTRADA")).unwrap();
    let mut linhas = entrada.lines().map(|l| {
        l.split_whitespace()
            .map(|t| t.parse::<u32>().unwrap())
            .collect::<Vec<_>>()
    });
    let prompt = linhas.next().unwrap();
    let forcados = linhas.next().unwrap();

    let bytes = llama_sys::mapear(std::path::Path::new(&var("TF_MODELO"))).unwrap();
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let mut cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    cfg.ctx = (prompt.len() + forcados.len() + 256).next_multiple_of(256);
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let aux = llama_model::GpuAuxWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let ctx = llama_vulkan::VulkanContext::new().unwrap();
    let gpu = llama_vulkan::LayerSplitForward::new(&ctx, &cfg, &raw, &aux).unwrap();

    // `TF_PREFILL_ATE=k1,k2,...`: em vez do decode forçado, refaz o prefill de
    // `prompt + forcados[..k]` e grava só os logits do fim — separa o caminho de decode
    // do de prefill.
    if let Ok(ks) = std::env::var("TF_PREFILL_ATE") {
        let mut saida = String::new();
        for k in ks.split(',').map(|k| k.parse::<usize>().unwrap()) {
            let seq: Vec<u32> = prompt.iter().chain(&forcados[..k]).copied().collect();
            let (logits, _) = prefill(&gpu, &seq);
            let _ = writeln!(saida, "{k} {}", top8(&logits));
        }
        std::fs::write(var("TF_SAIDA"), saida).unwrap();
        return;
    }

    // `TF_HIDDEN=<shard>`: grava em `TF_SAIDA` o hidden do fim de `prompt + forcados`
    // (f32 cru) daquele shard — com `LLAMA_RS_SPLIT` e `LLAMA_RS_STOP_LAYER`, é o hidden
    // depois de qualquer camada, para comparar com o `l_out-N` do llama.cpp.
    if let Ok(s) = std::env::var("TF_HIDDEN") {
        let seq: Vec<u32> = prompt.iter().chain(&forcados).copied().collect();
        prefill(&gpu, &seq);
        let s: usize = s.parse().unwrap();
        let h = gpu.dbg_hidden(s).unwrap();
        let b: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(var("TF_SAIDA"), b).unwrap();
        // `TF_BUFS=q,k,...`: buffers do `dbg_dn_buf`, em `TF_SAIDA.<nome>`.
        for nome in std::env::var("TF_BUFS")
            .unwrap_or_default()
            .split_terminator(',')
        {
            let v = gpu.dbg_dn_buf(s, nome).unwrap();
            let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(format!("{}.{nome}", var("TF_SAIDA")), b).unwrap();
        }
        return;
    }

    let (mut logits, mut pos) = prefill(&gpu, &prompt);

    let mut saida = String::new();
    for &t in &forcados {
        saida.push_str(&top8(&logits));
        saida.push('\n');
        logits = gpu.decode(t, pos).unwrap();
        pos += 1;
    }
    std::fs::write(var("TF_SAIDA"), saida).unwrap();
    eprintln!("{} passos gravados", forcados.len());
}
