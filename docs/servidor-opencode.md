# Servidor de chat e uso no opencode

O `llama-server` expõe o backend residente na API de chat da OpenAI, que é o que o
opencode fala. Este documento registra como subir, o que está implementado e — mais
importante — **onde estão os limites**, para não serem redescobertos na marra.

## Subir

```bash
cargo build --release -p llama-server --features gpu

numactl --interleave=all ./target/release/llama-server \
    -m models/Qwen3.8-27B-Q4_K_M.gguf \
    --gpu-layer-split --ctx 32768 --nome qwen3.8-27b
```

`numactl --interleave=all` não é opcional: sem ele o nó 0 estoura e a máquina trava (ver
`scripts/run.sh`). O servidor escuta em `127.0.0.1:8080` por padrão.

> **Rodando a partir do Claude Code o processo morre com exit 137.** O shell do agente
> tem `oom_score_adj=200` e o servidor herda esse valor, virando o alvo do OOM killer
> quando os 19 GB de pesos sobem para a VRAM. Não é falta de RAM nem bug: subir num
> terminal comum resolve.

Rotas: `POST /v1/chat/completions` (com e sem `stream`), `GET /v1/models`, `GET /health`.

As gerações são atendidas uma por vez; até 4 esperam na fila, e além disso a resposta é 503
com `Retry-After`. `/health` e `/v1/models` respondem mesmo durante uma geração. Prazos do
socket: `--timeout-leitura` (30 s sem receber nada → 408) e `--timeout-escrita` (60 s com a
escrita bloqueada → a geração é cancelada). Parâmetro inválido (`temperature` fora de
[0, 2], `n > 1`, `response_format` que não seja `text`...) é 400; `model` diferente do
`--nome` é atendido com o modelo servido e deixa uma linha `[api]` no log.

## Config do opencode

`~/.config/opencode/opencode.json`:

```json
"llamars": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "llama-rs (local)",
  "options": { "baseURL": "http://127.0.0.1:8080/v1" },
  "models": {
    "qwen3.8-27b": {
      "name": "Qwen3.8-27B (llama-rs)",
      "tool_call": true, "reasoning": true,
      "limit": { "context": 32768, "output": 8192 }
    }
  }
}
```

O `context` do config tem de caber no `--ctx` do servidor: prompt maior que o contexto é
recusado com erro, não truncado (truncar silenciosamente esconderia perda de instrução).

## Modo roteador (vários modelos, como o do llama.cpp)

Com `--models-preset`, o `llama-server` vira roteador: lê um INI no formato do
`models.ini` do llama.cpp, lista todas as seções em `GET /v1/models` e encaminha cada
pedido ao modelo do campo `model`. Cada modelo carregado é um `llama-server` filho (sob
`numactl --interleave=all`) numa porta local; o stream SSE passa direto. Pedido para um
modelo descarregado o carrega, e com `--models-max` (padrão 1) cheio o menos usado
recentemente é morto antes — o processo sai e a VRAM volta antes de o próximo subir. Modelo
com requisição em andamento não é descarregado: o pedido do outro modelo espera.

```bash
./target/release/llama-server --models-preset models.ini --host 127.0.0.1 --port 8081
```

```ini
[*]
gpu-layer-split = true

[qwen3.8-27b]
model           = /run/media/murilo/BACKUP/MODELOS/Qwen3.8-27B-Q4_K_M.gguf
ctx-size        = 32768
load-on-startup = true
```

Chaves aceitas: `model` (`m`), `ctx-size` (`c`), `n-gpu-layers` (`ngl`, sem efeito aqui),
`gpu-layer-split`, `mtp` e `load-on-startup`. As outras (`flash-attn`, `mmproj`,
`spec-type`...) geram um aviso e são ignoradas, então o `models.ini` do llama.cpp é lido
sem edição. `--gpu-layer-split` e `--mtp` passados ao roteador valem para todos os filhos.
Se o roteador morrer, os filhos saem sozinhos (percebem o stdin fechado), sem ficarem
órfãos segurando VRAM.

No opencode, o provider aponta para o roteador e lista os nomes das seções:

```json
"llamars-router": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "llama-rs router (local)",
  "options": { "baseURL": "http://127.0.0.1:8081/v1" },
  "models": {
    "qwen3.8-35b-a3b": {
      "name": "Qwen3.8-35B-A3B (llama-rs)",
      "tool_call": true, "reasoning": true,
      "limit": { "context": 131072, "output": 8192 }
    },
    "qwen3.8-27b": {
      "name": "Qwen3.8-27B (llama-rs)",
      "tool_call": true, "reasoning": true,
      "limit": { "context": 32768, "output": 8192 }
    }
  }
}
```

As flags do llama.cpp também valem fora do roteador: `--host`/`--port` (no lugar de
`--bind`), `-c`/`--ctx-size`, `--alias` (= `--nome`) e `-ngl` (aceito e ignorado).

## O formato deste modelo

Duas coisas do Qwen3.8 que fogem do que se espera de um modelo OpenAI-like:

**Tool call é XML, não JSON.** O template ensina o formato abaixo, e é o que o parser em
`llama-server/src/saida.rs` lê:

```text
<tool_call>
<function=read>
<parameter=path>
src/main.rs
</parameter>
</function>
</tool_call>
```

O valor de cada parâmetro é texto puro. O tipo vem do schema que o cliente mandou em
`tools`; sem schema, o valor fica string — inventar tipo quebraria uma ferramenta que
espera `"10"`.

**A resposta começa dentro do raciocínio.** O prompt termina em `<|im_start|>assistant\n
<think>\n`, então tudo até `</think>` é `reasoning_content`, não conteúdo. Um servidor que
ignorasse isso entregaria o raciocínio como resposta.

O chat template não é interpretado em runtime: está reimplementado em `llama-chat` e
provado contra a saída do Jinja real, caso a caso, em `refs/chat_qwen38.json`
(regenerável por `scripts/gen-chat-refs.py`).

## Reuso do KV-cache entre turnos

O servidor mantém uma sessão só e reaproveita o prefixo comum entre o que está no cache e
o prompt novo (`llama_model::Sessao`). O caso do agente — histórico inteiro de volta mais
um turno — processa apenas o que cresceu.

**Divergência no meio custava o cache inteiro.** Não era preguiça: 48 das 65 camadas do
qwen35 são delta-net, com estado recorrente. O KV de atenção volta atrás sozinho (basta
recuar o comprimento e deixar os tokens novos reescreverem os slots), mas o estado
recorrente é o produto de todos os tokens processados, em ordem. Pelo mesmo motivo não dá
para reprocessar um token que já entrou.

**Snapshot de fronteira de turno.** No prefill de cada requisição, logo depois do último token
especial do prompt, a sessão manda o backend copiar o estado recorrente, a janela da convolução e o comprimento do KV — ~155 MB
de VRAM, um snapshot só. Divergência **depois** dessa posição recua até ela
(`Reuso::RecuarPara`) em vez de reiniciar; divergência antes ainda custa tudo.

A fronteira fica no prompt (depois do último token especial, que o BPE nunca funde com o
texto seguinte), não no fim da resposta, porque é a **resposta** que o turno
seguinte re-renderiza: o cliente que não devolve o `reasoning_content` no histórico faz o
template reconstruir o turno do assistant diferente do que foi gerado, e essa divergência
começa depois da fronteira. Antes do snapshot isso reprocessava o prompt inteiro; agora
reprocessa só a resposta e o turno novo. Com o raciocínio de volta o texto re-renderizado
bate byte a byte e nem isso é preciso.

Medido no modelo real (`1769c35`): TTFT com 30k de contexto de 387 s para 31,5 s.

## Custo de contexto em VRAM

O KV-cache só existe nas camadas de atenção (16 das 65 no Qwen3.8-27B; as delta-net têm
estado de tamanho fixo). São **64 KiB por token** (16 camadas × kv_dim 1024 × K e V × 2 B
do f16):

| ctx | KV-cache |
|---:|---:|
| 4 096 | 0,26 GB |
| 32 768 | 2,2 GB |
| 65 536 | 4,4 GB |

*(Valores com o cache em f16 empacotado, 2026-08-22 — metade dos de f32.)* Com os
16,3 GB de pesos, `--ctx 32768` ocupa ~18,5 GB das duas MI50. Antes de o cache ser
indexado por camada de atenção, o mesmo contexto pediria 17 GB só de cache (em f32) e
não caberia.

## Medido no modelo real

2× MI50, layer-split (camadas 0..30 | 30..64), `--ctx 32768`, greedy, esforço `low`:

| requisição | prompt | saída | tempo |
|---|---:|---:|---:|
| pergunta curta | 46 tok | 25 tok | 1,9 s |
| com bloco de tools, turno 1 | 315 tok | 49 tok | 6,5 s |
| **turno 2 do mesmo diálogo** | **399 tok** | 60 tok | **3,4 s** |

O turno 2 tem prompt **maior** e roda em metade do tempo: o prefixo comum já estava no
cache e só os ~84 tokens novos foram processados. É o item de reuso pagando a conta.

O tool call do turno 1 saiu como `finish_reason: tool_calls` com
`{"path":"src/main.rs"}` — o XML do modelo convertido para o formato da API.

### TTFT

A linha `[gen]` do log traz o **tempo até o primeiro byte de stream**, contado da chegada
do pedido (render do template e tokenização incluídos, porque o cliente espera por eles):

```text
[gen] prompt 9110 tok (0 do cache, 9110 no prefill) …s (… tok/s) | decode … | ttft …s
```

É o número que decide a experiência com prompt grande: a taxa de decode não compensa
minutos de espera antes de a resposta começar. *Medido via CLI em 2026-08-21 (prompt de
9 312 tokens, batch 24 + GEMM, greedy): prefill a 12,75 ms/token → **TTFT ≈ 2 min** do
zero (sem cache de prefixo).* Hoje o bloco é de 256 tokens: prefill frio a 170/153/126/93
tok/s com 2k/8k/16k/30k (2026-11-20), ~1,85× abaixo do llama.cpp HIP (318 contra 171,9 com
7,5k).

## Uma requisição por vez

Uma thread de HTTP lê as requisições e um worker gera, uma por vez, com até 4 na fila. Os
pesos são residentes e um único decode já satura a banda das duas placas; atender duas
requisições em paralelo dividiria a mesma banda e ainda embaralharia a sessão do KV-cache.

## O que não está implementado

- Imagens (o runtime não tem visão), `logprobs`, `n > 1`, penalidades de repetição.
