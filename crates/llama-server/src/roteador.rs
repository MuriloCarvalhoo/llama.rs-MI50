//! Modo roteador, como o do `llama-server` do llama.cpp: vários modelos numa API só.
//!
//! O roteador não toca em GPU. Cada modelo carregado é um processo filho deste mesmo binário
//! em modo de um modelo, numa porta local, e o roteador repassa a requisição a ele pelo
//! `model` do corpo — o stream SSE passa byte a byte. Descarregar é matar o filho: a VRAM
//! volta com certeza quando o processo sai, o que nenhum `Drop` dentro do mesmo processo
//! garante.

use std::collections::BTreeMap;
use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

use crate::api;
use crate::http::{self, Requisicao};
use crate::servidor::{Prazos, agora, caminho_base};

/// Uma seção do INI: o nome exposto na API e os argumentos do filho que o serve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Preset {
    pub nome: String,
    pub args: Vec<String>,
    pub carregar_ao_iniciar: bool,
}

/// Chave do INI → flag do filho, e se é booleana. As abreviações são as do llama.cpp.
fn flag_do_filho(chave: &str) -> Option<(&'static str, bool)> {
    match chave {
        "model" | "m" => Some(("--model", false)),
        "ctx-size" | "c" | "ctx" => Some(("--ctx-size", false)),
        "n-gpu-layers" | "ngl" => Some(("--n-gpu-layers", false)),
        "gpu-layer-split" => Some(("--gpu-layer-split", true)),
        "mtp" => Some(("--mtp", true)),
        "load-on-startup" => Some(("load-on-startup", true)),
        _ => None,
    }
}

/// (flag do filho, é booleana) → valor.
type Opcoes = BTreeMap<(&'static str, bool), String>;

fn verdadeiro(valor: &str) -> bool {
    matches!(
        valor.to_ascii_lowercase().as_str(),
        "true" | "on" | "1" | "yes" | "enabled"
    )
}

/// Lê o INI de presets do llama.cpp: `[*]` vale para todas as seções e cada seção o
/// sobrescreve. Chave sem equivalente aqui (`flash-attn`, `mmproj`...) vira aviso e é
/// ignorada; seção sem `model` é erro.
pub fn ler_presets(texto: &str) -> Result<Vec<Preset>, String> {
    let mut comum = Opcoes::new();
    let mut secoes: Vec<(String, Opcoes)> = Vec::new();
    // `None` antes da primeira seção (onde fica o `version = 1`), `Some(None)` no `[*]`.
    let mut atual: Option<Option<usize>> = None;
    for linha in texto.lines().map(str::trim) {
        if linha.is_empty() || linha.starts_with(';') || linha.starts_with('#') {
            continue;
        }
        if let Some(nome) = linha.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            let nome = nome.trim();
            atual = Some(if nome == "*" {
                None
            } else {
                secoes.push((nome.to_owned(), Opcoes::new()));
                Some(secoes.len() - 1)
            });
            continue;
        }
        let Some((chave, valor)) = linha.split_once('=') else {
            return Err(format!("linha sem `=` no preset: {linha}"));
        };
        let (chave, valor) = (chave.trim(), valor.trim().to_owned());
        let Some(alvo) = atual else {
            continue; // `version = 1` e afins, fora de seção
        };
        let Some(flag) = flag_do_filho(chave) else {
            eprintln!("[roteador] chave `{chave}` não tem equivalente aqui — ignorada");
            continue;
        };
        match alvo {
            None => comum.insert(flag, valor),
            Some(i) => secoes.get_mut(i).and_then(|(_, m)| m.insert(flag, valor)),
        };
    }

    secoes
        .into_iter()
        .map(|(nome, proprias)| {
            let mut opcoes = comum.clone();
            opcoes.extend(proprias);
            if !opcoes.contains_key(&("--model", false)) {
                return Err(format!("preset `{nome}` sem `model`"));
            }
            let mut args = Vec::new();
            let mut carregar_ao_iniciar = false;
            for ((flag, booleana), valor) in opcoes {
                if flag == "load-on-startup" {
                    carregar_ao_iniciar = verdadeiro(&valor);
                } else if !booleana {
                    args.extend([flag.to_owned(), valor]);
                } else if verdadeiro(&valor) {
                    args.push(flag.to_owned());
                }
            }
            Ok(Preset {
                nome,
                args,
                carregar_ao_iniciar,
            })
        })
        .collect()
}

/// Um modelo carregado. `processo` é o filho; nos testes, qualquer coisa.
#[derive(Debug)]
pub struct Instancia<P> {
    pub nome: String,
    pub porta: u16,
    /// Requisições repassadas a ele que ainda não terminaram: com alguma, não é descarregado.
    pub em_uso: usize,
    /// Relógio lógico do último uso, para descarregar o menos recente (LRU).
    pub uso: u64,
    pub processo: P,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Passo {
    /// Já carregado, no índice dado.
    Usar(usize),
    /// Há vaga: carregar.
    Carregar,
    /// Sem vaga: descarregar o do índice dado (o ocioso usado há mais tempo) e decidir de novo.
    Descarregar(usize),
    /// Sem vaga e todos ocupados: esperar alguma requisição terminar.
    Esperar,
}

/// O que fazer para servir `nome` com no máximo `max` modelos carregados (0 = sem limite).
pub fn proximo_passo<P>(carregados: &[Instancia<P>], nome: &str, max: usize) -> Passo {
    if let Some(i) = carregados.iter().position(|c| c.nome == nome) {
        return Passo::Usar(i);
    }
    if max == 0 || carregados.len() < max {
        return Passo::Carregar;
    }
    carregados
        .iter()
        .enumerate()
        .filter(|(_, c)| c.em_uso == 0)
        .min_by_key(|(_, c)| c.uso)
        .map_or(Passo::Esperar, |(i, _)| Passo::Descarregar(i))
}

#[derive(Default)]
struct Estado {
    carregados: Vec<Instancia<Child>>,
    relogio: u64,
}

pub struct Roteador {
    presets: Vec<Preset>,
    max: usize,
    /// Flags do próprio roteador que valem para todo filho (`--gpu-layer-split`, `--mtp`).
    extra: Vec<String>,
    estado: Mutex<Estado>,
    livre: Condvar,
}

impl Roteador {
    pub fn novo(presets: Vec<Preset>, max: usize, extra: Vec<String>) -> Self {
        Self {
            presets,
            max,
            extra,
            estado: Mutex::default(),
            livre: Condvar::new(),
        }
    }

    /// Garante o modelo carregado e marca uma requisição em uso nele; devolve a porta.
    /// A trava fica com quem carrega: as cargas saem uma por vez, e o filho anterior já
    /// saiu (`wait`) antes de o próximo subir — dois modelos de 27B não cabem juntos.
    fn reservar(&self, preset: &Preset) -> Result<u16, String> {
        let mut g = self.estado.lock().unwrap_or_else(PoisonError::into_inner);
        // Filho que morreu sozinho (crash, OOM) sai da lista e é recarregado.
        g.carregados
            .retain_mut(|c| matches!(c.processo.try_wait(), Ok(None)));
        loop {
            match proximo_passo(&g.carregados, &preset.nome, self.max) {
                Passo::Usar(i) => {
                    g.relogio += 1;
                    let relogio = g.relogio;
                    let c = g.carregados.get_mut(i).ok_or("índice fora da lista")?;
                    c.em_uso += 1;
                    c.uso = relogio;
                    return Ok(c.porta);
                }
                Passo::Descarregar(i) => {
                    let mut c = g.carregados.remove(i);
                    eprintln!("[roteador] descarregando `{}`", c.nome);
                    let _ = c.processo.kill();
                    let _ = c.processo.wait();
                }
                Passo::Esperar => {
                    g = self.livre.wait(g).unwrap_or_else(PoisonError::into_inner);
                }
                Passo::Carregar => {
                    let (processo, porta) = subir(preset, &self.extra)?;
                    g.carregados.push(Instancia {
                        nome: preset.nome.clone(),
                        porta,
                        em_uso: 0,
                        uso: 0,
                        processo,
                    });
                }
            }
        }
    }

    fn liberar(&self, porta: u16) {
        let mut g = self.estado.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(c) = g.carregados.iter_mut().find(|c| c.porta == porta) {
            c.em_uso = c.em_uso.saturating_sub(1);
        }
        self.livre.notify_all();
    }

    fn atender(&self, fluxo: &TcpStream, prazos: Prazos) -> Result<(), Box<dyn std::error::Error>> {
        fluxo.set_read_timeout(Some(prazos.leitura))?;
        fluxo.set_write_timeout(Some(prazos.escrita))?;
        let mut escritor = fluxo;
        let json = "application/json";
        let req = match http::ler_requisicao(&mut BufReader::new(fluxo)) {
            Ok(r) => r,
            Err(e) => {
                http::responder(
                    &mut escritor,
                    e.status(),
                    json,
                    &api::erro_json(&e.to_string()),
                )?;
                return Ok(());
            }
        };
        match (req.metodo.as_str(), caminho_base(&req.caminho)) {
            ("GET", "/health") => http::responder(&mut escritor, 200, json, br#"{"status":"ok"}"#)?,
            ("GET", "/v1/models" | "/models") => {
                let nomes = self.presets.iter().map(|p| p.nome.as_str());
                let corpo = serde_json::to_vec(&api::lista_de_modelos(nomes, agora()))?;
                http::responder(&mut escritor, 200, json, &corpo)?;
            }
            ("OPTIONS", _) => http::responder(&mut escritor, 200, json, b"")?,
            ("POST", _) => {
                let pedido = serde_json::from_slice::<serde_json::Value>(&req.corpo)
                    .ok()
                    .and_then(|v| v.get("model")?.as_str().map(str::to_owned))
                    .unwrap_or_default();
                let Some(preset) = self
                    .presets
                    .iter()
                    .find(|p| p.nome.eq_ignore_ascii_case(&pedido))
                else {
                    let nomes: Vec<&str> = self.presets.iter().map(|p| p.nome.as_str()).collect();
                    let msg = format!("modelo `{pedido}` desconhecido; há: {}", nomes.join(", "));
                    http::responder(&mut escritor, 404, json, &api::erro_json(&msg))?;
                    return Ok(());
                };
                let porta = match self.reservar(preset) {
                    Ok(p) => p,
                    Err(e) => {
                        http::responder(&mut escritor, 500, json, &api::erro_interno_json(&e))?;
                        return Ok(());
                    }
                };
                let r = repassar(porta, &req, &mut escritor);
                self.liberar(porta);
                r?;
            }
            _ => http::responder(
                &mut escritor,
                404,
                json,
                &api::erro_json(&format!("sem rota para {} {}", req.metodo, req.caminho)),
            )?,
        }
        Ok(())
    }
}

/// Sobe o filho de um preset numa porta livre e espera o `/health` dele.
fn subir(preset: &Preset, extra: &[String]) -> Result<(Child, u16), String> {
    // shortcut: a porta pode ser tomada entre soltá-la aqui e o filho abri-la; numa máquina
    // local isso não acontece na prática — se acontecer, o filho sai e a carga falha com erro.
    let porta = TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map_err(|e| e.to_string())?
        .port();
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut args = preset.args.clone();
    args.extend(extra.iter().cloned());
    args.extend([
        "--alias".to_owned(),
        preset.nome.clone(),
        "--bind".to_owned(),
        format!("127.0.0.1:{porta}"),
        "--sai-com-o-pai".to_owned(),
    ]);
    eprintln!("[roteador] carregando `{}` na porta {porta}", preset.nome);
    // Sem `numactl --interleave=all` o nó 0 estoura ao subir os pesos e a máquina trava.
    let mut com_numa = Command::new("numactl");
    com_numa.arg("--interleave=all").arg(&exe).args(&args);
    let mut processo = match com_numa.stdin(Stdio::piped()).spawn() {
        Ok(p) => p,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("AVISO: numactl não encontrado — sem ele o nó 0 estoura e a máquina trava");
            Command::new(&exe)
                .args(&args)
                .stdin(Stdio::piped())
                .spawn()
                .map_err(|e| e.to_string())?
        }
        Err(e) => return Err(e.to_string()),
    };
    loop {
        if let Ok(Some(status)) = processo.try_wait() {
            return Err(format!(
                "o servidor de `{}` saiu durante a carga ({status})",
                preset.nome
            ));
        }
        if saudavel(porta) {
            return Ok((processo, porta));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn saudavel(porta: u16) -> bool {
    let Ok(mut s) = TcpStream::connect(("127.0.0.1", porta)) else {
        return false;
    };
    let mut resposta = String::new();
    s.write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        .is_ok()
        && s.read_to_string(&mut resposta).is_ok()
        && resposta.starts_with("HTTP/1.1 200")
}

/// Manda a requisição ao filho e copia a resposta ao cliente conforme chega (SSE incluso):
/// o filho fecha a conexão no fim da resposta.
fn repassar<W: Write>(porta: u16, req: &Requisicao, cliente: &mut W) -> std::io::Result<()> {
    let mut filho = TcpStream::connect(("127.0.0.1", porta))?;
    write!(
        filho,
        "{} {} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        req.metodo,
        req.caminho,
        req.corpo.len()
    )?;
    filho.write_all(&req.corpo)?;
    std::io::copy(&mut filho, cliente)?;
    Ok(())
}

/// Escuta em `bind` e atende cada conexão numa thread: um stream longo não segura o
/// `/v1/models` nem os pedidos para o mesmo modelo (o filho tem a fila dele).
pub fn servir(
    bind: &str,
    roteador: Roteador,
    prazos: Prazos,
) -> Result<(), Box<dyn std::error::Error>> {
    let escuta = TcpListener::bind(bind)?;
    let roteador = Arc::new(roteador);
    let nomes: Vec<&str> = roteador.presets.iter().map(|p| p.nome.as_str()).collect();
    eprintln!(
        "[roteador] {bind} — {} (no máximo {} carregado(s))",
        nomes.join(", "),
        roteador.max
    );
    let r = Arc::clone(&roteador);
    std::thread::spawn(move || {
        for p in r.presets.iter().filter(|p| p.carregar_ao_iniciar) {
            match r.reservar(p) {
                Ok(porta) => r.liberar(porta),
                Err(e) => eprintln!("[roteador] load-on-startup de `{}`: {e}", p.nome),
            }
        }
    });
    for conexao in escuta.incoming() {
        match conexao {
            Ok(fluxo) => {
                let r = Arc::clone(&roteador);
                std::thread::spawn(move || {
                    if let Err(e) = r.atender(&fluxo, prazos) {
                        eprintln!("[roteador] conexão encerrada: {e}");
                    }
                });
            }
            Err(e) => eprintln!("[roteador] accept falhou: {e}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;

    const INI: &str = "
; comentário
version = 1

[*]
flash-attn = on
gpu-layer-split = true
c = 8192

[qwen-a]
model = /m/a.gguf
load-on-startup = true
n-gpu-layers = 99

[qwen-b]
model = /m/b.gguf
ctx-size = 32768
gpu-layer-split = false
";

    #[test]
    fn ini_mescla_o_comum_e_a_secao_sobrescreve() {
        let p = ler_presets(INI).unwrap();

        assert_eq!(p.len(), 2);
        assert_eq!(p[0].nome, "qwen-a");
        assert!(p[0].carregar_ao_iniciar);
        assert_eq!(
            p[0].args,
            [
                "--ctx-size",
                "8192",
                "--gpu-layer-split",
                "--model",
                "/m/a.gguf",
                "--n-gpu-layers",
                "99"
            ]
        );
        assert_eq!(p[1].nome, "qwen-b");
        assert!(!p[1].carregar_ao_iniciar);
        assert_eq!(p[1].args, ["--ctx-size", "32768", "--model", "/m/b.gguf"]);
    }

    #[test]
    fn ini_sem_model_ou_linha_quebrada_e_erro() {
        assert!(ler_presets("[x]\nctx-size = 10\n").is_err());
        assert!(ler_presets("[x]\nmodel /a.gguf\n").is_err());
    }

    fn inst(nome: &str, em_uso: usize, uso: u64) -> Instancia<()> {
        Instancia {
            nome: nome.to_owned(),
            porta: 0,
            em_uso,
            uso,
            processo: (),
        }
    }

    #[test]
    fn modelo_carregado_e_usado_sem_recarga() {
        let c = [inst("a", 0, 1), inst("b", 0, 2)];
        assert_eq!(proximo_passo(&c, "b", 2), Passo::Usar(1));
    }

    #[test]
    fn com_vaga_carrega_e_sem_limite_sempre_ha_vaga() {
        assert_eq!(proximo_passo(&[inst("a", 0, 1)], "b", 2), Passo::Carregar);
        assert_eq!(proximo_passo(&[inst("a", 1, 1)], "b", 0), Passo::Carregar);
    }

    #[test]
    fn sem_vaga_descarrega_o_ocioso_menos_recente() {
        let c = [inst("a", 0, 5), inst("b", 0, 3), inst("c", 1, 1)];
        assert_eq!(proximo_passo(&c, "d", 3), Passo::Descarregar(1));
    }

    #[test]
    fn sem_vaga_e_todos_ocupados_espera() {
        assert_eq!(proximo_passo(&[inst("a", 1, 1)], "b", 1), Passo::Esperar);
    }
}
