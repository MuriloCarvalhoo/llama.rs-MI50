//! Chamadas ao SO sem API segura, cada uma atrás de uma função segura com o seu `SAFETY`.
//! O resto da workspace roda com `#![forbid(unsafe_code)]` e passa por aqui.

use std::path::Path;

pub use memmap2::Mmap;

/// Mapeia um arquivo só para leitura.
pub fn mapear(path: &Path) -> std::io::Result<Mmap> {
    let file = std::fs::File::open(path)?;
    // SAFETY: mapeamento read-only de um arquivo tratado como imutável durante a
    // execução. Modificação externa do .gguf enquanto mapeado seria UB — mesma
    // premissa que o llama.cpp adota para o seu mmap de modelo.
    unsafe { Mmap::map(&file) }
}

/// Mapeia o `.gguf` em memória em vez de copiá-lo para a RAM. Um `fs::read` do
/// 14B Q8_0 aloca 15 GB de RAM anônima antes mesmo dos pesos serem extraídos;
/// com mmap as páginas ficam no page cache e são recuperáveis sob pressão.
///
/// `populate` (`MAP_POPULATE`) faz o prefault das page tables na hora do mapa, e
/// `Advice::Sequential` dá ao kernel a janela de readahead do padrão desta carga: ler
/// tudo uma vez, em ordem, para subir à VRAM. Sem os dois, o page-in acontece como
/// tempestade de page fault **no meio** do memcpy para o staging, serializado com ele.
pub fn mapear_para_carga(path: &Path) -> std::io::Result<Mmap> {
    let file = std::fs::File::open(path)?;
    // SAFETY: o mesmo de `mapear`.
    let mm = unsafe { memmap2::MmapOptions::new().populate().map(&file) }?;
    // Conselho ao kernel, não contrato: se o SO recusar, a carga só fica mais lenta.
    let _ = mm.advise(memmap2::Advice::Sequential);
    Ok(mm)
}

/// Pina a thread atual ao conjunto de CPUs via sched_setaffinity.
/// cpu_set_t = 128 bytes (1024 bits) em Linux x86_64.
#[cfg(target_os = "linux")]
pub fn pinar_thread(cpus: &[usize]) {
    let mut mask = [0u64; 16];
    for &cpu in cpus {
        // get_mut carrega a mesma garantia que o `cpu < 1024` fazia: 16 palavras de 64 bits.
        if let Some(palavra) = mask.get_mut(cpu / 64) {
            *palavra |= 1u64 << (cpu % 64);
        }
    }
    // SAFETY: `mask` é um array local de 128 bytes (o tamanho passado), com o layout de
    // bits do cpu_set_t; pid 0 = thread atual. Falha só deixa a thread sem afinidade.
    unsafe {
        libc::sched_setaffinity(0, size_of_val(&mask), mask.as_ptr().cast());
    }
}

/// Define política MPOL_BIND para o NUMA node 0 via set_mempolicy.
#[cfg(target_os = "linux")]
pub fn ligar_memoria_ao_no0() {
    let nodemask: u64 = 1; // bit 0 = NUMA node 0
    // SAFETY: `nodemask` é um u64 local válido durante a chamada e maxnode (64) cabe nele.
    unsafe {
        libc::syscall(
            libc::SYS_set_mempolicy,
            libc::MPOL_BIND,
            &raw const nodemask,
            64,
        );
    }
}
