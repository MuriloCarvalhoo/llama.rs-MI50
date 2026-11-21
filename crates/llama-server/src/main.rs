//! Servidor OpenAI-compatível para o backend residente.
//!
//! Uma geração por vez, de propósito: os pesos são residentes e o decode ocupa as
//! duas GPUs inteiras: paralelizar requisições só disputaria a mesma banda de memória.
//! As outras esperam numa fila curta, e `/health` responde durante a geração — ver
//! `servidor.rs`.
#![forbid(unsafe_code)]

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "llama-server",
    about = "Servidor de chat (API OpenAI) do llama-rs"
)]
struct Args {
    /// Caminho para o modelo GGUF
    #[arg(short, long, required_unless_present = "models_preset")]
    model: Option<PathBuf>,

    /// Endereço de escuta
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,

    /// Host de escuta, como no llama.cpp (com `--port`, substitui o `--bind`).
    #[arg(long, conflicts_with = "bind")]
    host: Option<String>,

    /// Porta de escuta, como no llama.cpp (com `--host`, substitui o `--bind`).
    #[arg(long, conflicts_with = "bind")]
    port: Option<u16>,

    /// Teto do contexto (KV-cache). Custa VRAM: 64 KiB por token no Qwen3.8-27B (16 camadas
    /// de atenção × kv_dim 1024 × K e V × 2 B do f16), 2 GiB em 32k — mais os snapshots do
    /// MTP com `--mtp`.
    #[arg(short = 'c', long, visible_alias = "ctx-size", default_value_t = 32768)]
    ctx: usize,

    /// Segundos sem receber nada do cliente enquanto ele manda a requisição, antes do 408.
    /// Protege a thread do HTTP de conexão que abre e não fala (o prazo é por leitura, não
    /// da requisição inteira).
    #[arg(long = "timeout-leitura", default_value_t = 30)]
    timeout_leitura: u64,

    /// Segundos que uma escrita no socket pode ficar bloqueada. Cliente que para de ler o
    /// stream estoura esse prazo, e a geração dele é cancelada em vez de travar a fila.
    #[arg(long = "timeout-escrita", default_value_t = 60)]
    timeout_escrita: u64,

    /// Nome do modelo exposto na API (o padrão é o nome do arquivo).
    #[arg(long, visible_alias = "alias")]
    nome: Option<String>,

    /// Divide as camadas entre as GPUs. Necessário para o que não cabe numa placa.
    #[arg(long = "gpu-layer-split", default_value_t = false)]
    gpu_layer_split: bool,

    /// Constrói o backend com a cabeça de multi-token prediction (`nextn`) e o plano de
    /// verify, e o laço do motor passa a propor→verificar (2 propostas encadeadas por
    /// passo). Desligado por padrão: custa ~310 MB de snapshots mais 289 MB do bloco.
    #[arg(long = "mtp", default_value_t = false)]
    mtp: bool,

    /// Aceito para os comandos do llama.cpp funcionarem, mas sem efeito: os pesos vão
    /// sempre inteiros para a GPU, e o multi-GPU é o `--gpu-layer-split`.
    #[arg(long = "n-gpu-layers")]
    n_gpu_layers: Option<String>,

    /// Modo roteador: INI de presets no formato do llama.cpp, um modelo por seção. Cada
    /// modelo roda num `llama-server` filho, carregado quando pedido.
    #[arg(long = "models-preset")]
    models_preset: Option<PathBuf>,

    /// Modo roteador: modelos carregados ao mesmo tempo (0 = sem limite). Dois de 27B não
    /// cabem juntos nas 2×16 GB.
    #[arg(long = "models-max", default_value_t = 1)]
    models_max: usize,

    /// Uso interno do roteador: sai quando o stdin fecha, para o filho não ficar órfão
    /// segurando VRAM se o roteador morrer.
    #[arg(long = "sai-com-o-pai", hide = true)]
    sai_com_o_pai: bool,
}

impl Args {
    fn endereco(&self) -> String {
        if self.host.is_none() && self.port.is_none() {
            return self.bind.clone();
        }
        let host = self.host.as_deref().unwrap_or("127.0.0.1");
        format!("{host}:{}", self.port.unwrap_or(8080))
    }
}

fn main() -> std::process::ExitCode {
    // `-ngl` do llama.cpp tem duas letras depois do hífen, o que o clap não aceita como curta.
    let args = Args::parse_from(std::env::args_os().map(|a| {
        if a == "-ngl" {
            "--n-gpu-layers".into()
        } else {
            a
        }
    }));
    if args.n_gpu_layers.is_some() {
        eprintln!("[args] --n-gpu-layers ignorado: os pesos vão sempre inteiros para a GPU");
    }
    if args.sai_com_o_pai {
        std::thread::spawn(|| {
            let _ = std::io::copy(&mut std::io::stdin(), &mut std::io::sink());
            std::process::exit(0);
        });
    }
    let resultado = match &args.models_preset {
        Some(ini) => rotear(&args, ini),
        None => servir(&args),
    };
    match resultado {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("erro: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn prazos(args: &Args) -> llama_server::servidor::Prazos {
    llama_server::servidor::Prazos {
        leitura: std::time::Duration::from_secs(args.timeout_leitura),
        escrita: std::time::Duration::from_secs(args.timeout_escrita),
    }
}

fn rotear(args: &Args, ini: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    use llama_server::roteador::{Roteador, ler_presets, servir};
    let presets = ler_presets(&std::fs::read_to_string(ini)?)?;
    if presets.is_empty() {
        return Err(format!("{}: nenhum preset", ini.display()).into());
    }
    // As flags booleanas do roteador valem para todo filho.
    let extra = [
        (args.gpu_layer_split, "--gpu-layer-split"),
        (args.mtp, "--mtp"),
    ]
    .into_iter()
    .filter(|(ligada, _)| *ligada)
    .map(|(_, flag)| flag.to_owned())
    .collect();
    let roteador = Roteador::novo(presets, args.models_max, extra);
    servir(&args.endereco(), roteador, prazos(args))
}

#[cfg(not(feature = "gpu"))]
fn servir(_args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    Err("compile com --features gpu: o servidor precisa do backend Vulkan".into())
}

#[cfg(feature = "gpu")]
fn servir(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use llama_server::motor::Motor;
    use llama_server::servidor::laco;
    use llama_vulkan::{LayerSplitForward, ResidentForward, VulkanContext};

    let prazos = prazos(args);
    let bind = args.endereco();
    let modelo = args.model.as_ref().ok_or("falta -m/--model")?;

    let bytes = llama_sys::mapear(modelo)?;
    let f = gguf::GgufFile::parse(&bytes)?;
    let mut cfg = llama_model::LlamaConfig::from_gguf(&f)?;
    cfg.ctx = cfg.ctx.min(args.ctx);
    let tokenizer = llama_tokenizer::Tokenizer::from_gguf(&f)?;
    let nome = args.nome.clone().unwrap_or_else(|| {
        modelo
            .file_stem()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_else(|| "modelo".to_owned())
    });

    eprintln!("[carga] lendo pesos de {}", modelo.display());
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg)?;
    let aux = llama_model::GpuAuxWeights::from_gguf(&f, &bytes, &cfg)?;

    let vk = VulkanContext::new()?;
    let n_gpus = vk.amd_compute_devices().len();
    if n_gpus == 0 {
        return Err("nenhuma GPU AMD encontrada".into());
    }

    let mtp = args.mtp && raw.mtp.is_some();
    if args.mtp && !mtp {
        eprintln!("[mtp] o modelo não traz bloco nextn — seguindo sem MTP");
    }

    if args.gpu_layer_split && n_gpus >= 2 {
        let backend =
            LayerSplitForward::new_com(&vk, &cfg, &raw, &aux, mtp).map_err(|e| e.to_string())?;
        let layout: Vec<String> = backend
            .layout()
            .iter()
            .map(|(d, a, b)| format!("GPU{d}: camadas {a}..{b}"))
            .collect();
        eprintln!("[gpu] layer-split — {}", layout.join(" | "));
        laco(
            &bind,
            &nome,
            Motor::novo(&tokenizer, &backend, cfg.ctx, cfg.eos_id),
            prazos,
        )
    } else {
        let backend =
            ResidentForward::new_com(&vk, &cfg, &raw, &aux, mtp).map_err(|e| e.to_string())?;
        eprintln!("[gpu] residente numa placa");
        laco(
            &bind,
            &nome,
            Motor::novo(&tokenizer, &backend, cfg.ctx, cfg.eos_id),
            prazos,
        )
    }
}
