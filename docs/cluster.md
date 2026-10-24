# Ollama on two machines (ollama-cluster)

`ollama cluster` lets one Ollama use the GPU memory of a second machine. The second machine runs only a small worker: the ggml RPC server from the same llama.cpp build that Ollama ships. Ollama keeps one address, one model list and the same API. A model that fits on the first machine stays there; a larger one is split across both machines automatically.

It was built and tested on two NVIDIA DGX Spark (ASUS Ascent GX10) connected with a 200 Gb/s direct cable (ConnectX-7, RDMA).

The same commands also keep GGUF models from Hugging Face as plain repository copies in several locations ordered from the fastest disk, and register them in Ollama without a second copy.

## How it works

- The worker is `ollama cluster worker`: it runs `ggml-rpc-server` from the Ollama build, listens only on the direct-cable address, and reports its state (version, llama.cpp build, devices, free memory, connected clients, transport) on a status port.
- Ollama sees each worker device as an extra GPU with library `RPC` (`RPC0[10.100.48.2:50052]`). The scheduler adds `--rpc` only to runners placed on a worker.
- **Ollama never depends on the second machine.** Before every load the worker is checked; a worker that is down, runs another build or already holds a model is skipped and the model loads on this machine. The check is repeated right before llama-server starts.
- Policy `auto` (default): a worker is used only when the model does not fit here, counting memory that unloading other models would free. Policy `spread` (or `OLLAMA_SCHED_SPREAD=1`) splits every model. Policy `worker-first` sends models from a given size to the worker first. Memory can be kept free on the first machine, and single models can have their own rule; see [Where models go](#where-models-go). A request can force a split or the first machine with the option `cluster_split: true|false`.
- The RPC server serves one client at a time, so one worker holds one split model. Other models load on the first machine meanwhile.

## Requirements

- Two machines with the same NVIDIA GPU and driver major version. Tested: DGX Spark, driver 580, CUDA 13.
- A direct cable between them with an IPv4 address on both ends in one subnet (e.g. 10.100.48.1/24 and 10.100.48.2/24). Ports with RDMA are detected automatically. A worker on the ordinary LAN is refused unless `--allow-lan` is given: the RPC server has no authentication.
- SSH from the first machine to the second (`add` copies your key with `ssh-copy-id` when only a password works).
- Docker on both machines when Ollama runs in Docker (the worker runs from the same image), otherwise a systemd user service.

## Build the image

The official image has no RPC module, so the cluster needs its own build. On a DGX Spark with the CUDA 13 toolkit, cmake, Go and `libibverbs-dev`:

```sh
VERSION=0.40.4-cluster scripts/build_cluster_image.sh   # -> ollama-cluster:0.40.4-cluster
```

The script builds llama-server with the RPC backend and `ggml-rpc-server` (RDMA through libibverbs), the CUDA backend for compute capability 12.1, and the Ollama binary, then packs them with `Dockerfile.cluster`.

Install the command on the host (the commands run on the host, also when Ollama is in Docker):

```sh
id=$(docker create ollama-cluster:0.40.4-cluster)
docker cp "$id":/usr/bin/ollama ~/.local/bin/ollama && docker rm "$id"
```

## Attach the second machine

```sh
ollama cluster init                                   # checks only, changes nothing
ollama cluster add user@10.100.48.2 --image ollama-cluster:0.40.4-cluster
```

`add` checks both machines before changing anything, copies the image to the worker (once per version), starts the worker container (`--network host --device /dev/infiniband --ulimit memlock=-1`), and recreates the Ollama container with what the cluster needs:

- the cluster image,
- host networking on the same port (RDMA does not work through the Docker bridge),
- `/dev/infiniband` and unlimited locked memory,
- `OLLAMA_CONTEXT_LENGTH=32768` when none is set (`--context 0` leaves it unset). Without a limit large models take their full trained context: on the test machines `llama3.1:8b` with five parallel slots took 96 GB.

The container settings are saved to `~/.ollama-cluster/backups/` first, and Ollama is restarted only after confirmation (or with `--yes`). Finally a small model is loaded across both machines as a test. Running `add` again finishes an interrupted run or reports that everything is set.

The worker listens only on the cable address. To also drop other hosts on that link, `add` offers a firewall rule on the worker (needs sudo there once):

```sh
sudo iptables -I INPUT ! -i lo -p tcp -m multiport --dports 50052,50053 ! -s 10.100.48.1 -j DROP
```

## Day to day

| Command | What it does |
|---|---|
| `ollama cluster status` | machines, free memory, worker state, where loaded models sit, locations |
| `ollama cluster ls [filter] [--files]` | every model with parameters, quantization, trained context, capabilities, whether it fits this machine or needs the worker (estimate), and where its files are: the Ollama store or a repository copy |
| `ollama cluster ps` | loaded models with memory per machine, context in use, parallel slots, active requests and when they unload; free memory and state of each machine |
| `ollama cluster policy` | shows or sets where models go: `policy set auto\|spread\|worker-first [--from GB] [--keep-free-here GB]`, `policy tensor on\|off`, `policy model <name> here\|split\|worker\|tensor\|auto` |
| `ollama cluster doctor [--fix]` | checks the container, locations, model links, worker, transport, firewall and the fallbacks of the last 24 h; `--fix` relinks models to the best copy |
| `ollama cluster mv <model>... <location\|store>` | moves the files of rarely used models to another location and leaves links, so they free the fast disk and keep working; `store` brings them back ([Move models to another disk](#move-models-to-another-disk)) |
| `ollama cluster bench <model>` | load time and the two llama-bench tests, pp512 (prompt processing) and tg128 (generation), on this machine alone, split, on the worker first and with tensor parallelism, cold and warm (warm: mean ± standard deviation of 3 runs); a model that cannot fit here alone is not tried here |
| `ollama cluster upgrade [--image ...]` | after an update: recreates Ollama when its image tag points to a newer build and brings the workers to the same image; other settings (also the context limit) stay as they are |
| `ollama cluster remove` | removes the worker, takes it out of the configuration and recreates the Ollama container exactly as it was before `add` |

`ollama ls` and `ollama ps` stay as they are, so they keep working against an official server; `ollama cluster ls` and `ollama cluster ps` add the cluster view. Both take `--json`.

`ollama cl` is short for `ollama cluster`, e.g. `ollama cl ps`.

Every command explains what it does and changes, with examples: `ollama cluster <command> help`, `ollama cluster help <command>` or `--help`.

The configuration lives in `cluster.json` next to the models (`OLLAMA_MODELS`, or `OLLAMA_CLUSTER_CONFIG`).

## Where models go

`ollama cluster policy` shows the policy; Ollama reads it at every model load, so a change applies to the next load without a restart.

| Setting | Effect |
|---|---|
| `policy set auto` (default) | a model that fits here stays here, a larger one is split |
| `policy set spread` | every model is split across all machines |
| `policy set worker-first --from 30` | models of 30 GB and more go to the worker: the whole model when it fits there (nothing of it here), otherwise split with the worker filled first; smaller models stay here |
| `policy set <any> --keep-free-here 40` | 40 GB of this machine stay free for other software: a model that does not fit in the rest is split, and the split leaves those 40 GB free |
| `policy tensor on` | every model without a rule of its own runs with tensor parallelism across both machines (see below); `off` goes back to the policy |
| `policy model <name> here\|split\|worker\|tensor` | a rule for one model that wins over the policy and over `policy tensor`; `auto` removes it |

How it reaches llama-server: a model placed only on the worker starts with `--device RPC0[...]`; memory kept free here is passed as a per-device `--fit-target`. llama-server lists worker devices first and fits layers from the last device, so this machine is filled up to its target and the worker takes the rest. The worker holds one model at a time; when it is busy, a model that would go there loads here as under `auto`. `ollama cluster ps` and the `/cluster` page show `worker` for a model with nothing on this machine.

### Tensor parallelism

Split by layers, the two machines compute one after the other, so a split model is slower than the same model on one machine. With tensor parallelism (llama.cpp `-sm tensor` over RPC, from llama.cpp b11521) both machines compute every layer, each with half of the weights, and exchange partial results directly; generation gets faster than on one machine, also for models that fit here. Turn it on for all models with `ollama cluster policy tensor on`, or for one model with `ollama cluster policy model <name> tensor`; a request can force or forbid it with `"cluster_tensor": true` or `false` in its options.

How it runs: llama.cpp pairs exactly two RPC servers, so the GPU of this machine is served by a second RPC server that Ollama starts next to llama-server and stops with it. It listens on the address of this machine on the direct link (`local_addr` of the worker): through 127.0.0.1 the connection stays on TCP and generation was 4x slower than on one machine (llama3.1:8b 13 tok/s). llama-server gets `--rpc <worker>,<this machine> --device RPC0,RPC1 --split-mode tensor --tensor-split 1,1 -ngl 999 --fit off`. The worker comes first: the first server listens for the second one on its RPC port + 1000 (51052 by default).

When it is used: one GPU here, one worker GPU, the worker free, and half of the model plus a tenth fits on each machine (llama-server cannot fit the context to the memory in this mode, so Ollama checks it before). Otherwise the policy decides, e.g. a 156 GB model with this machine half full is split by layers. A model running this way takes the worker even when it fits here, so another model that needs the worker meanwhile loads here or fails as with any busy worker. `ollama cluster ps` shows `tensor: here ... + <worker> ...`; llama-server reports the split buffers as one device (`Meta()`) with the size each GPU holds, and Ollama counts that size on each machine.

## Watch it in a browser

`http://<ollama host>:11434/cluster` (also `/status`) is a page that refreshes every 2 seconds and shows:

- every machine with free and total memory, how much the loaded models take and the worker state (idle, holding a split model, transport, or why it is not used),
- the loaded models: memory, where they sit (here or split, with the share on each machine), context in use and trained, parallel slots, active requests, when they unload,
- requests running now and the last 100 requests: time, model, path, client, status, duration, bytes sent,
- requests per model since the start and the number of server errors,
- the last server log entries at INFO and above (model loads, splits, fallbacks, errors).

The same data as JSON: `GET /api/cluster/status` and `GET /api/cluster/activity`. Everything is kept in memory only (the last 200 requests and 300 log entries) and starts empty after a restart. The page and the endpoints have no authentication, like the rest of the Ollama API, so they show model names and client addresses to anyone who can reach the port.

Requests that change something are accepted only from the machine Ollama runs on: `PUT /api/cluster/config`, `POST /api/cluster/link`, `POST /api/cluster/relink` and `POST /api/cluster/move` (the `ollama cluster` commands use them). The peer address of the connection must be a loopback address, the Host header an IP address, `localhost` or the machine name, and a request from a browser (with an `Origin` header) is refused. When Ollama runs in Docker on a bridge network (before `add` switches it to host networking), the host reaches it from the bridge gateway: allow that address with `OLLAMA_CLUSTER_ADMIN_ADDRS=172.17.0.1` in the container. Every saved configuration is checked: worker addresses must be IP addresses, the SSH target `user@host`, locations absolute paths outside system directories. Fields that come from clients (path, model, client address) are cut to 256 characters in the request log.

## Move models to another disk

Models pulled from the Ollama library live in the Ollama store (`OLLAMA_MODELS`). A model that is rarely used can move to another location and keep working:

```sh
ollama cluster locations add /mnt/second-disk/models
ollama cl mv qwen3.5:9b gemma4:12b /mnt/second-disk/models
ollama cl mv qwen3.5:9b store                              # back to the Ollama store
```

- Only the large files move (weights, projector, draft); template, parameters and license stay in the store. They go to `<location>/ollama/blobs/sha256-<sum>`, and the blob in the store becomes a link to them, like the links to repository copies. `ollama cluster ls --files` shows where each file is.
- Ollama does the move itself (`POST /api/cluster/move`), so it works also when Ollama runs in Docker and owns the store. Each file is written as `.part`, checked against its sha256 while it is copied and flushed to disk; only then the blob is switched to the link and the old file is removed. Within one disk (and one mount in Docker) the file is hard linked instead of copied. An interrupted move changes nothing.
- The target must be a load location; an archive location is refused because Ollama never loads from it. The model then loads from that disk, so it must be mounted when the model runs; when it is not, the load fails with the name of the missing file, as for repository copies.
- A file shared by several models (e.g. after `ollama cp`) moves for all of them; `mv` names them before it asks. Files of Hugging Face repository copies stay where they are (`copy` and `drop` manage them). A loaded model keeps running from the file it has open.
- `ollama rm` removes only the link: the moved file stays in the location. `ollama cluster doctor` lists such unused files with the command that deletes them.

## Model copies from Hugging Face

```sh
ollama cluster locations add /mnt/fast-nvme
ollama cluster locations add /mnt/second-disk
ollama cluster locations add /mnt/nas/models --archive   # slow: kept, never loaded from
ollama cluster pull unsloth/Qwen3-4B-GGUF:Q4_K_M         # -> hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M
```

- A copy lives in `<location>/hf/<owner>/<repo>/` with the repository paths, so llama.cpp, vLLM or llama-swap can use the same files. `.ollama-cluster.json` in the copy records the files, their sha256 and the Ollama models made from them.
- Hugging Face names every large file by its sha256 and Ollama names blobs the same way, so `blobs/sha256-<sum>` becomes a link to the file in the copy and nothing is stored twice.
- A file is used only when its content has the expected sha256: a link is made or switched only to a file with the blob's content, a downloaded or copied file is checked while it is written (a mismatch removes it), and an existing copy is reused only when its files match the repository listing. A file with the same name and size from another revision, or a truncated file, is never linked. The sums are kept per location in `.ollama-cluster-digests.json` (by path, size and modification time), so an unchanged file is read once.
- `pull` reuses a copy found in any location, downloads only what is missing (resumable `.part` files) to the first location with room for the model plus 20 GB, and picks the F16 projector for vision models.
- Before every load the links are checked: a copy that disappeared (unmounted disk, powered-off node) is replaced by the next location that has it, and a copy in a better location is preferred once it is back.
- `models` lists copies and which one Ollama loads; `copy <repo> <location>` copies a repository (Ollama switches to it when the location is better); `drop <repo> <location>` deletes one copy only when every file of it exists in another copy with the same sha256 (another copy may hold other quantizations); a file Ollama loads needs a copy outside archive locations. If a link cannot be switched, the copy is put back and nothing is deleted.
- `import <hf-cache> [repo...]` turns a Hugging Face cache (`models--owner--repo/snapshots/...`) into copies by moving the files within the same disk and leaving links behind, so tools using the cache keep working. A 140 GB model moves in under a second. Run it as the owner of the cache (for a root-owned cache: as root, e.g. in the cluster image with the disk mounted).

## Measured on two DGX Spark (ASUS Ascent GX10) connected 200 Gb/s

llama.cpp b11351 in Ollama 0.40.3-cluster, every model from the Ollama store as installed, context 32768 per slot, 256 generated tokens, generation speed of one warm run (model already loaded; measured with the earlier version of `bench`, before it reported pp512/tg128). "Load" is the cold start. "2 nodes" forces the split with `cluster_split: true`, so it also shows the price of the split for models that fit on one machine; Ollama splits on its own only what does not fit. Measured 2026-10-14 with `ollama cluster bench <model> --json`, one model at a time, with the company traffic paused; embedding models are left out. RDMA (RoCEv2) between the machines.

| Model | Size | Params, quant | 1 node: load | 1 node: tok/s | 2 nodes: load | 2 nodes: tok/s | 2 nodes: memory here + worker |
|---|---:|---|---:|---:|---:|---:|---|
| `llama3.1:8b` | 4.9 GB | 8.0B, Q4_K_M | 6 s | 43.1 | 9 s | 33.2 | 12.8 + 15.5 GB |
| `ornith:9b` | 5.6 GB | 9.0B, Q4_K_M | 5 s | 38.1 | 8 s | 31.6 | 3.8 + 3.4 GB |
| `qwen3.5:9b` | 6.6 GB | 9.7B, Q4_K_M | 6 s | 36.9 | 9 s | 29.0 | 3.6 + 3.8 GB |
| `gemma4:e2b` | 7.2 GB | 5.1B, Q4_K_M | 10 s | 106.6 | 9 s | 44.1 | 6.7 + 2.2 GB |
| `gemma4:e4b` | 9.6 GB | 8.0B, Q4_K_M | 9 s | 59.5 | 10 s | 27.8 | 5.0 + 3.4 GB |
| `gpt-oss:20b` | 13.8 GB | 20.9B, MXFP4 | 8 s | 58.5 | 19 s | 41.3 | 7.1 + 7.9 GB |
| `f0rc3ps/nu11secur1tyAI4:latest` (num_gpu 0 in its Modelfile: CPU only, never split) | 15.2 GB | 7.6B, F16 | 14 s | 5.5 | - | - | - |
| `qwen3.5:27b` | 17.4 GB | 27.8B, Q4_K_M | 8 s | 12.3 | 24 s | 11.6 | 9.0 + 10.3 GB |
| `qwen3.6:27b` | 17.4 GB | 27.8B, Q4_K_M | 9 s | 12.3 | 23 s | 11.3 | 9.0 + 10.3 GB |
| `qwen3.8:27b` | 17.7 GB | 27.3B, Q4_K_M | 9 s | 25.2 | 25 s | 21.3 | 9.0 + 10.8 GB |
| `gemma4:26b` | 18.0 GB | 25.8B, Q4_K_M | 10 s | 66.9 | 26 s | 46.3 | 10.2 + 12.2 GB |
| `muse-glimmer:30b` | 18.2 GB | 27.9B, Q4_K_M | 9 s | 12.3 | 25 s | 11.4 | 9.4 + 10.5 GB |
| `north-mini-code-1.0:latest` | 18.6 GB | 30.5B, Q4_K_M | 9 s | 82.6 | 26 s | 51.3 | 11.0 + 12.1 GB |
| `glm-4.7-flash:latest` | 19.0 GB | 29.9B, Q4_K_M | 8 s | 64.8 | 26 s | 45.2 | 13.8 + 15.7 GB |
| `gemma4:31b` | 19.9 GB | 31.3B, Q4_K_M | 11 s | 10.7 | 29 s | 9.7 | 16.1 + 18.2 GB |
| `laguna-xs-2.1:latest` | 20.3 GB | 33.4B, Q4_K_M | 8 s | 89.4 | 27 s | 57.6 | 11.2 + 12.5 GB |
| `ornith:35b` | 21.2 GB | 34.7B, Q4_K_M | 10 s | 74.2 | 30 s | 51.7 | 10.6 + 12.0 GB |
| `qwen3.5:35b` | 23.9 GB | 36.0B, Q4_K_M | 10 s | 72.1 | 32 s | 49.7 | 10.5 + 13.6 GB |
| `qwen3.6:35b` | 23.9 GB | 36.0B, Q4_K_M | 10 s | 73.5 | 32 s | 50.7 | 10.5 + 13.6 GB |
| `gpt-oss:120b` | 65.4 GB | 116.8B, MXFP4 | 19 s | 41.7 | 1 min 22 s | 33.1 | 30.4 + 36.3 GB |
| `qwen3.5:122b` | 81.4 GB | 125.1B, Q4_K_M | 23 s | 29.4 | 1 min 40 s | 25.2 | 35.8 + 45.2 GB |
| `hf.co/unsloth/Qwen3.5-397B-A17B-GGUF:UD-IQ3_XXS` | 141.3 GB | 396.3B, IQ3_XXS | does not fit | does not fit | 3 min 8 s | 15.3 | 64.6 + 77.8 GB |

A model that fits on one machine is faster there, which is why the default policy splits only what does not fit. Tool calls work on the split 397B model.

### The whole model on the worker

Same settings, `ollama cluster bench <model> --only worker`: the model gets the placement rule `worker` for the run, so it sits entirely on the second machine and is driven from the first one ("worker only"); the 397B model does not fit there and is split with the worker filled first. "Change" compares with the same model entirely on the first machine.

| Model | Size | Placement | 1 node: tok/s | worker: load | worker: tok/s | change | memory here + worker |
|---|---:|---|---:|---:|---:|---:|---|
| `llama3.1:8b` | 4.9 GB | worker only | 43.1 | 8 s | 35.2 | -18% | 0.0 + 26.7 GB |
| `ornith:9b` | 5.6 GB | worker only | 38.1 | 9 s | 31.9 | -16% | 0.0 + 6.4 GB |
| `qwen3.5:9b` | 6.6 GB | worker only | 36.9 | 9 s | 31.8 | -14% | 0.0 + 6.6 GB |
| `gemma4:e2b` | 7.2 GB | worker only | 106.6 | 5 s | 67.9 | -36% | 0.0 + 2.4 GB |
| `gemma4:e4b` | 9.6 GB | worker only | 59.5 | 8 s | 44.3 | -26% | 0.0 + 4.2 GB |
| `gpt-oss:20b` | 13.8 GB | worker only | 58.5 | 21 s | 45.6 | -22% | 0.0 + 13.4 GB |
| `qwen3.5:27b` | 17.4 GB | worker only | 12.3 | 44 s | 11.7 | -5% | 0.0 + 18.4 GB |
| `qwen3.6:27b` | 17.4 GB | worker only | 12.3 | 45 s | 11.7 | -5% | 0.0 + 18.4 GB |
| `qwen3.8:27b` | 17.7 GB | worker only | 25.2 | 34 s | 21.4 | -15% | 0.0 + 17.5 GB |
| `gemma4:26b` | 18.0 GB | worker only | 66.9 | 35 s | 50.5 | -25% | 0.0 + 20.6 GB |
| `muse-glimmer:30b` | 18.2 GB | worker only | 12.3 | 34 s | 11.7 | -5% | 0.0 + 17.8 GB |
| `north-mini-code-1.0:latest` | 18.6 GB | worker only | 82.6 | 37 s | 58.5 | -29% | 0.0 + 21.4 GB |
| `glm-4.7-flash:latest` | 19.0 GB | worker only | 64.8 | 53 s | 48.9 | -25% | 0.0 + 27.9 GB |
| `gemma4:31b` | 19.9 GB | worker only | 10.7 | 56 s | 10.1 | -6% | 0.0 + 32.3 GB |
| `laguna-xs-2.1:latest` | 20.3 GB | worker only | 89.4 | 55 s | 60.7 | -32% | 0.0 + 22.1 GB |
| `ornith:35b` | 21.2 GB | worker only | 74.2 | 59 s | 55.5 | -25% | 0.0 + 21.8 GB |
| `qwen3.5:35b` | 23.9 GB | worker only | 72.1 | 1 min 4 s | 54.7 | -24% | 0.0 + 23.4 GB |
| `qwen3.6:35b` | 23.9 GB | worker only | 73.5 | 1 min 2 s | 57.0 | -22% | 0.0 + 23.4 GB |
| `gpt-oss:120b` | 65.4 GB | worker only | 41.7 | 2 min 54 s | 36.5 | -12% | 0.0 + 65.1 GB |
| `qwen3.5:122b` | 81.4 GB | worker only | 29.4 | 3 min 31 s | 25.3 | -14% | 0.0 + 80.1 GB |
| `hf.co/unsloth/Qwen3.5-397B-A17B-GGUF:UD-IQ3_XXS` | 141.3 GB | worker first | does not fit | 4 min 41 s | 11.5 | - | 19.0 + 122.5 GB |

A model on the worker alone is always slower than the same model here: slow dense models lose 5-6%, big ones 12-15%, fast small MoE models 22-36%, because every token goes back over the link. It pays off only when this machine has no room, or when this machine is busy with another model at the same moment.

### After the merge with upstream main (llama.cpp b11669)

Ollama 0.40.4-cluster-24bcdd6f (upstream `main` of 2026-10-17 plus this fork), llama.cpp b11669, the same two machines, RDMA (RoCEv2, MTU 1024). `ollama cluster bench <model> --json` with its defaults: context 32768, pp512 and tg128, warm runs are the mean ± standard deviation of 3, tokens/s. Measured 2026-10-18, one model at a time; unlike the table above the company traffic was **not** paused, so another model (`qwen3.8:27b`, 17.5 GB) was answering on the first machine during the runs. The last column is generation on 0.40.3 from the tables above (256 tokens, earlier `bench`), for comparison.

| Model | Placement | Load | pp512 warm | tg128 warm | Memory here + worker | tok/s on 0.40.3 |
|---|---|---:|---:|---:|---|---:|
| `llama3.1:8b` | this machine | 5 s | 2925.0 ± 8.8 | 43.2 ± 0.5 | 31.0 + 0 GB | 43.1 |
| | split | 9 s | 2728.0 ± 43.8 | 33.8 ± 0.4 | 15.6 + 17.3 GB | 33.2 |
| | worker only | 10 s | 2740.9 ± 44.1 | 34.8 ± 0.7 | 0 + 31.0 GB | 35.2 |
| `qwen3.6:35b` | this machine | 10 s | 2465.4 ± 12.7 | 72.4 ± 1.4 | 23.5 + 0 GB | 73.5 |
| | split | 31 s | 2120.5 ± 51.5 | 51.8 ± 2.7 | 11.1 + 13.1 GB | 50.7 |
| | worker only | 61 s | 1960.7 ± 59.1 | 56.7 ± 3.1 | 0 + 23.4 GB | 57.0 |
| `gpt-oss:120b` | this machine | 18 s | 2193.0 ± 16.8 | 42.2 ± 0.6 | 64.7 + 0 GB | 41.7 |
| | split | 1 min 20 s | 2053.9 ± 58.0 | 34.1 ± 0.5 | 32.2 + 34.8 GB | 33.1 |
| | worker only | 2 min 7 s | 1953.4 ± 100.8 | 35.4 ± 1.3 | 0 + 64.7 GB | 36.5 (load 2 min 54 s) |
| `qwen3.8-flash-next:125b-a6b-q4` (112.2 GB, qwen4exp, 6B active) | split | 2 min 18 s | 751.5 ± 18.9 | 23.0 ± 0.6 | 40.3 + 46.5 GB | 22.2-23.7 (context 256k) |
| `deepseek-v4-flash:0731-mxfp4` (156.4 GB, MXFP4) | split | 3 min 29 s | 508.0 ± 11.4 | 14.2 ± 0.2 | 73.6 + 83.0 GB | - |

Generation and load are the same as on 0.40.3 within the noise, so the merge did not slow the fork down; loading the whole `gpt-oss:120b` on the worker got faster. The Qwen3.5-397B files were removed from the test machines before this run, so it was not measured again.

## Known limits

- The RPC server has no authentication: keep the worker on a direct cable and add the firewall rule.
- A model placed on the worker only (policy `worker-first` or the rule `worker`) does not fall back to this machine when the worker cannot take it at start; the request fails and the next one is placed again. A split model falls back to this machine as before.
- When the worker stops while a split model is loaded, the request in flight fails once; the next request loads the model on this machine.
- Loading over RPC is slower than from a local disk (the weights travel over the link at about 0.5 GB/s).
- One worker holds one split model at a time.
- Models are split by layers unless tensor parallelism is on for them ([Tensor parallelism](#tensor-parallelism)). `--split-mode row` does not work with RPC devices.
- With tensor parallelism the worker listens for the RPC server of this machine on its RPC port + 1000 on all addresses while the model loads; the firewall rule of `ollama cluster add` covers only the RPC and status ports. Loading takes a little longer than split by layers (DeepSeek-V4-Flash 156 GB: 3 min 44 s against 3 min 30 s).
- The image is built for DGX Spark (arm64, CUDA 13, compute capability 12.1).

### Tensor parallel in newer llama.cpp

[ggml-org/llama.cpp#26610](https://github.com/ggml-org/llama.cpp/pull/26610) (merged 2026-10-06, in b11521 and later, so in Ollama 0.40.4) adds a server-to-server all-reduce for tensor parallel over RPC: both machines run an RPC server and the client drives them as `-dev RPC0/RPC1 -sm tensor`. Measured with plain llama-bench (llama.cpp b11580, `-fa 1 -p 512 -n 128`, tokens/s, mean ± standard deviation) on the same two DGX Spark (ASUS Ascent GX10) connected 200 Gb/s with RDMA, 2026-10-14:

| Model | Test | 1 node | layer split (as this fork) | tensor parallel | tensor vs 1 node |
|---|---|---:|---:|---:|---:|
| `llama3.1:8b` Q4_K_M | pp512 | 3057.47 ± 32.23 | 2851.33 ± 41.66 | 2751.94 ± 62.42 | 0.90x |
| | tg128 | 44.99 ± 0.12 | 37.99 ± 2.29 | 64.42 ± 0.50 | 1.43x |
| `qwen3.8:27b` Q4_K_M | pp512 | 816.58 ± 10.28 | 792.66 ± 14.90 | 916.34 ± 0.18 | 1.12x |
| | tg128 | 12.36 ± 0.00 | 11.49 ± 0.00 | 20.52 ± 0.03 | 1.66x |
| `Qwen3.5-397B-A17B` UD-IQ3_XXS | pp512 | does not fit | 310.33 ± 8.20 | 326.61 ± 25.71 | 1.05x vs layer |
| | tg128 | does not fit | 16.09 ± 0.05 | 18.02 ± 0.06 | 1.12x vs layer |

Generation of dense models gets 1.4-1.7x faster than on one machine; the large MoE model gains 12% over the layer split, since it reads few weights per token. Prompt processing changes little. Several Ollama blobs do not load in plain llama.cpp b11580 (`gptoss`, `glm4moelite`, and the qwen35moe and gemma4 files of the Ollama library), so they were not measured there; the llama-server of Ollama loads them.

Since 2026-10-18 `ollama cluster` uses it ([Tensor parallelism](#tensor-parallelism)). Measured through Ollama 0.40.4-cluster (llama.cpp b11669) with `ollama cluster bench <model> --only here,split,tensor`, context 32768, warm runs: mean ± standard deviation of 3, tokens/s; the same two DGX Spark (ASUS Ascent GX10) connected 200 Gb/s with RDMA, 2026-10-18. The company traffic ran on the first machine meanwhile, through a separate Ollama, so the numbers carry some noise. `qwen3.8:27b` runs with the MTP draft of the Ollama library model, as Ollama starts it.

| Model | Placement | Load | pp512 warm | tg128 warm | Memory here + worker | tg vs this machine |
|---|---|---:|---:|---:|---|---:|
| `qwen3.8:27b` | this machine | 8 s | 689.9 ± 2.1 | 24.4 ± 0.1 | 17.6 + 0 GB | |
| | split by layers | 23 s | 613.6 ± 31.3 | 21.4 ± 0.3 | 9.4 + 10.4 GB | 0.88x |
| | tensor parallel | 27 s | 611.1 ± 6.3 | 34.7 ± 0.5 | 8.9 + 9.4 GB | 1.42x |
| `gpt-oss:120b` | this machine | 18 s | 2216.4 ± 3.5 | 42.7 ± 0.3 | 64.5 + 0 GB | |
| | split by layers | 1 min 18 s | 2009.9 ± 67.1 | 33.8 ± 0.6 | 31.5 + 33.9 GB | 0.79x |
| | tensor parallel | 1 min 26 s | 2021.6 ± 42.1 | 58.1 ± 1.6 | 32.4 + 32.4 GB | 1.36x |
| DeepSeek-V4-Flash-0731 MXFP4, 156 GB | this machine | does not fit | | | | |
| | split by layers | 3 min 30 s | 500.8 ± 5.3 | 14.5 ± 0.0 | 74.0 + 83.2 GB | |
| | tensor parallel | 3 min 44 s | 576.8 ± 4.7 | 21.4 ± 0.1 | 79.4 + 79.4 GB | 1.47x vs layers |

Tensor parallel generates 1.4x faster than one machine and 1.5-1.7x faster than the layer split; prompt processing stays at the level of the layer split, 10% below one machine for the models that fit here; loading takes a few seconds more than the layer split. A manual run of llama-server with the same arguments gave llama3.1:8b 57-60 tok/s against 43.3 on one machine. The memory column of the run used a build that counted half of the split buffers; the table shows the corrected values.
