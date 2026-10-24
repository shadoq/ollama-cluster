package clustercmd

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"
)

// Command groups shown by `ollama cluster --help`.
var commandGroups = []*cobra.Group{
	{ID: "machines", Title: "Second machine:"},
	{ID: "inspect", Title: "Look and check:"},
	{ID: "copies", Title: "Model files and copies from Hugging Face:"},
}

type helpText struct {
	group   string
	long    string
	example string
}

// helpTexts holds the long help of every command, keyed by its path below
// `ollama cluster`. Keeping them in one place keeps them consistent.
var helpTexts = map[string]helpText{
	"init": {
		group: "machines",
		long: `Check this machine before attaching a second one. Changes nothing.

Reports the Ollama version and where it runs (Docker container or service),
whether the build supports a second machine, the GPU and driver, the direct
cable interfaces (with RDMA) and the model locations. Every problem comes with
one sentence and the command that fixes it.`,
		example: `  ollama cluster init`,
	},
	"add": {
		group: "machines",
		long: `Attach a second machine to this Ollama.

What it does, in order:
  1. checks both machines (GPU, driver, memory, disk, Docker, the direct
     cable and RDMA) and stops before changing anything if something is wrong,
  2. copies the cluster image to the second machine (once per build),
  3. starts the worker there: the RPC server listening only on the cable
     address, plus a small status server,
  4. recreates the Ollama container with what the cluster needs (cluster
     image, host networking on the same port, /dev/infiniband, unlimited
     locked memory, OLLAMA_CONTEXT_LENGTH when none is set, the model
     locations); the old settings are saved to ~/.ollama-cluster/backups/
     first and Ollama restarts only after you confirm (or with --yes),
  5. loads a small model split across both machines as a test.

Running it again finishes an interrupted run or confirms that everything is
set. Applications keep using the same address and port. Undo with
'ollama cluster remove'.

The worker runs in Docker when Docker is available on the second machine and
Ollama here runs in Docker; otherwise as a systemd user service (--mode).`,
		example: `  ollama cluster add jarek@10.100.48.2
  ollama cluster add jarek@10.100.48.2 --image ollama-cluster:0.40.3-cluster
  ollama cluster add jarek@10.100.48.2 --context 0 --yes   # keep the context unlimited, no question`,
	},
	"upgrade": {
		group: "machines",
		long: `Bring Ollama and the workers to one build after an update.

In Docker, the Ollama container is recreated when its image tag points to a
newer build (for example after rebuilding ollama-cluster:<version>) or when
--image names another image, and every worker gets the same image. Steps
that are already done are skipped.

Settings that are already right are kept: network, ports, environment
(upgrade never adds OLLAMA_CONTEXT_LENGTH), volumes, GPUs and the restart
policy. The old settings are saved to ~/.ollama-cluster/backups/ before the
restart. Ollama stops answering for a few seconds; check 'ollama cluster ps'
first if a model may be in use.`,
		example: `  ollama cluster upgrade
  ollama cluster upgrade --image ollama-cluster:0.40.4-cluster --yes`,
	},
	"remove": {
		group: "machines",
		long: `Detach the second machine and restore Ollama as it was before 'add'.

Stops and removes the worker (container or service) and its firewall rule,
removes it from the configuration and recreates the Ollama container from the
first backup in ~/.ollama-cluster/backups/: the original image, network,
ports, environment and volumes. Models and repository copies stay on disk.`,
		example: `  ollama cluster remove
  ollama cluster remove --yes`,
	},
	"worker": {
		group: "machines",
		long: `Run the worker on the second machine. 'ollama cluster add' starts it; you
normally never run it by hand.

The worker starts ggml-rpc-server from the same build, listening only on the
direct cable address (--addr), restarts it when it exits, and answers on the
status port with its version, llama.cpp build, devices, free memory,
connected clients and transport. The RPC server has no authentication.`,
		example: `  ollama cluster worker --addr 10.100.48.2`,
	},
	"status": {
		group: "inspect",
		long: `Show the cluster at a glance: Ollama version and llama.cpp build, the
placement policy, every machine with its total and free memory, the worker
state (idle, holding a split model, or why it is not used), the loaded
models with memory per machine and the model locations in order with free
space. Changes nothing.

For models see 'ollama cluster ls' and 'ollama cluster ps'. In a browser,
http://<ollama host>:11434/cluster shows the same live, with the requests
running now, the last requests and the server log.`,
		example: `  ollama cluster status`,
	},
	"ls": {
		group: "inspect",
		long: `List the models with more than 'ollama ls' shows.

Columns:
  NAME          model name; * marks a model that is loaded now
  SIZE          size of the model files
  PARAMS        parameter count
  QUANT         quantization
  CONTEXT       context the model was trained with (Ollama uses less when
                OLLAMA_CONTEXT_LENGTH or num_ctx says so)
  CAPABILITIES  tools, vision, thinking, embedding, insert
  RUNS          here = fits this machine; split = needs the second machine;
                too large = does not fit both. An estimate from the file size
                plus 10% for context and buffers; a long context takes more.
  FILES         Ollama store = blobs in OLLAMA_MODELS; a path = a repository
                copy in that location, linked into Ollama (hf owner/repo);
                BROKEN = a linked file is unreachable (ollama cluster doctor --fix)

The optional filter keeps models whose name contains it. Changes nothing.`,
		example: `  ollama cluster ls
  ollama cluster ls qwen
  ollama cluster ls 397b --files      # every data file and its path
  ollama cluster ls --json`,
	},
	"ps": {
		group: "inspect",
		long: `Show the loaded models with more than 'ollama ps' shows.

Columns:
  NAME     model name
  MEMORY   memory the model takes in total
  WHERE    here, or split with the memory on this machine and on each worker
  CONTEXT  context in use, and the trained context when larger
  SLOTS    requests the model serves at the same time (OLLAMA_NUM_PARALLEL)
  ACTIVE   requests using the model right now
  UNLOADS  when an idle model leaves memory (keep_alive); "in use" while
           requests run, "never" for keep_alive -1

Below: free and total memory on this machine and on each worker, and the
worker state (idle, holding a split model, transport, or why it is not used).
Changes nothing.`,
		example: `  ollama cluster ps
  ollama cluster ps --json`,
	},
	"doctor": {
		group: "inspect",
		long: `Check everything the cluster depends on and explain each problem in one
sentence with the command that fixes it.

Checks: Ollama answers and runs a cluster build; the container (image with
RPC, host networking, /dev/infiniband, locked memory, context limit); every
location is available and visible to Ollama; model links point at reachable
files; the worker answers, runs the same build, its transport (RDMA or TCP)
and restarts; other GPU processes and model servers on the worker; the
firewall rule; how often a model loaded only here in the last 24 hours
because the worker was not usable.

--fix repairs what is safe to repair: it relinks every model to the best
reachable copy. Nothing else is changed.`,
		example: `  ollama cluster doctor
  ollama cluster doctor --fix`,
	},
	"bench": {
		group: "inspect",
		long: `Measure load time and speed of a model with the two tests of llama-bench:
  pp512   prompt processing: a prompt of 512 tokens (--prompt-tokens), sent
          raw and starting differently each time, so no cached prefix helps
  tg128   generation: 128 new tokens (--tokens)
Results are tokens per second; warm tests run 3 times (--repeat) and show
mean ± standard deviation, like llama-bench. The numbers come from the
Ollama API, so they include Ollama's own overhead.

Each test runs in four placements, from a cold start and warm:
  here    this machine alone
  split   across this machine and the worker, as the default policy splits
  worker  the worker first, as the policy worker-first places it: the whole
          model on the worker when it fits there ("worker only"), otherwise
          split with the worker filled first; for these runs the model gets
          the rule 'worker', and its previous rule comes back at the end
  tensor  tensor parallelism across both machines, half of the model on each

The model is unloaded between runs, so applications using it wait; you are
asked first when it is loaded (or use --yes). A model that cannot fit here
alone is not run here. Use it to decide where a model should go: split by
layers, a model that fits one machine is faster there; tensor parallel
usually generates faster than one machine.`,
		example: `  ollama cluster bench qwen3.5:122b
  ollama cluster bench qwen3.5:122b --only split,worker --repeat 5
  ollama cluster bench llama3.1:8b --prompt-tokens 2048 --tokens 256
  ollama cluster bench gpt-oss:120b --only worker --json
  ollama cluster bench hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M --context 8192`,
	},
	"policy": {
		group: "machines",
		long: `Show or set where models go. Ollama reads the policy at every model load, so
changes apply to the next load without a restart (a loaded model stays where
it is until it unloads).

Policies:
  auto          a model that fits on this machine stays here, a larger one
                is split across this machine and the worker (default)
  spread        every model is split across all machines
  worker-first  models from --from GB up go to the worker: the whole model
                when it fits there, otherwise split with the worker filled
                first; smaller models stay here

--keep-free-here GB keeps memory free on this machine for other software
(any policy): a model that does not fit in the rest is split, and the split
leaves that memory free.

Tensor parallelism ('ollama cluster policy tensor on') runs models with
llama.cpp -sm tensor: both machines compute every layer with half of the
model each, so generation is faster than on one machine, also for models that
fit here. The model takes the worker; while the worker holds another model, or
half of the model does not fit on one machine, the policy decides.

Rules for single models win over the policy and over tensor parallelism:
  here    this machine only
  split   across this machine and the worker, by layers
  worker  as worker-first, for this model
  tensor  tensor parallel across both machines, for this model
  auto    remove the rule

The worker holds one model at a time. When it is busy, a model that would go
there loads here as under auto. A request with "cluster_split": true or false
in its options still forces a split or this machine, and "cluster_tensor":
true or false forces or forbids tensor parallelism.`,
		example: `  ollama cluster policy
  ollama cluster policy set worker-first --from 30
  ollama cluster policy set auto --keep-free-here 40
  ollama cluster policy model gpt-oss:120b worker
  ollama cluster policy model qwen3.8:27b tensor
  ollama cluster policy tensor on
  ollama cluster policy model qwen3.8:27b auto`,
	},
	"policy tensor": {
		long: `Turn tensor parallelism for all models on or off. When on, every model
without a rule of its own runs across both machines with llama.cpp -sm tensor,
half of the model on each, while the worker is free and each half fits; the
GPU of this machine is served by a second RPC server that Ollama starts on the
direct link next to llama-server. When off, models go by the policy and
the rule 'tensor' of single models.`,
		example: `  ollama cluster policy tensor on
  ollama cluster policy tensor off`,
	},
	"policy set": {
		long: `Set the policy for all models: auto, spread or worker-first.
--from (worker-first) and --keep-free-here are in GB; a flag left out keeps
its current value, 0 turns it off.`,
		example: `  ollama cluster policy set worker-first --from 30
  ollama cluster policy set worker-first --from 0
  ollama cluster policy set auto --keep-free-here 40
  ollama cluster policy set auto --keep-free-here 0`,
	},
	"policy model": {
		long: `Fix where one model goes: here, split, worker or tensor; auto removes the rule.
Use the name as 'ollama ls' shows it ("name" and "name:latest" are the same).`,
		example: `  ollama cluster policy model gpt-oss:120b worker
  ollama cluster policy model llama3.1:8b here
  ollama cluster policy model qwen3.8:27b tensor
  ollama cluster policy model gpt-oss:120b auto`,
	},
	"locations": {
		group: "copies",
		long: `List and order the directories that hold repository copies of models,
from the fastest disk to the slowest.

A copy lives in <location>/hf/<owner>/<repo>/ with the same files as on
Hugging Face, so llama.cpp, vLLM or llama-swap can use it too. Ollama links
its blobs to the copy in the first location that has it; when that disk is
not available it switches to the next one, and back when it returns.
An archive location (e.g. a NAS) keeps copies but is never loaded from.

Without a subcommand: prints the locations in order with free space.`,
		example: `  ollama cluster locations
  ollama cluster locations add /mnt/fast-nvme
  ollama cluster locations add /mnt/second-disk --position 2
  ollama cluster locations add /mnt/nas/models --archive
  ollama cluster locations remove /mnt/second-disk`,
	},
	"locations add": {
		long: `Add a directory as a location (at the end, or at --position). The directory
and its hf/ subdirectory are created when missing.

When Ollama runs in Docker and does not see the directory, the container is
recreated with it mounted at the same path (after you confirm, or with
--yes); the old settings are saved to ~/.ollama-cluster/backups/ first.
Adding an existing location again moves it to the new position.`,
		example: `  ollama cluster locations add /mnt/dgx-data/models
  ollama cluster locations add /mnt/nvme/models --position 1
  ollama cluster locations add /mnt/nas/models --archive`,
	},
	"locations remove": {
		long: `Remove a directory from the locations. The files stay on disk and the
container keeps the mount. Models linked to copies there keep working while
the files exist; 'ollama cluster doctor --fix' relinks them to another
location when one has a copy.`,
		example: `  ollama cluster locations remove /mnt/second-disk`,
	},
	"pull": {
		group: "copies",
		long: `Get a GGUF model from Hugging Face as a plain repository copy and add it to
Ollama without a second copy.

  - <owner/repo:TAG> picks the files of one quantization, e.g. Q4_K_M or
    UD-IQ3_XXS; split files (-00001-of-00005) are taken together. For a
    vision model a projector is added (F16 first, then BF16, then F32).
  - An existing copy in any location is reused when its files have the
    sha256 the repository lists (another revision of the same size is not);
    only missing files are downloaded. A copy only in an archive location
    is first copied to a faster one.
  - New files go to the first location with room for the model plus 20 GB
    (or --to). Downloads resume from .part files after an interruption.
    Each file is checked against its sha256 while it is written; a file
    that does not match is removed.
  - Ollama blobs become links to the files in the copy (Hugging Face and
    Ollama both name files by sha256, so nothing is stored twice). A link
    is made only to a file with the blob's content; computed sums are kept
    per location, so an unchanged file is read once.

The token from HF_TOKEN or ~/.cache/huggingface/token is used for gated
models; when Hugging Face rejects it, public repositories are fetched
without it.`,
		example: `  ollama cluster pull unsloth/Qwen3-4B-GGUF:Q4_K_M
  ollama cluster pull unsloth/Qwen3.5-397B-A17B-GGUF:UD-IQ3_XXS --to /mnt/dgx-data/models
  ollama cluster pull unsloth/Qwen3-4B-GGUF:Q8_0 --name qwen3-4b-q8`,
	},
	"models": {
		group: "copies",
		long: `List the repository copies in all locations: size, whether each copy is
complete, which copy Ollama loads from, archive copies, and the Ollama
models made from each repository. Changes nothing.

For all Ollama models (also those from the Ollama library) see
'ollama cluster ls'.`,
		example: `  ollama cluster models`,
	},
	"copy": {
		group: "copies",
		long: `Copy a repository to another location, for example from a slow disk to a
fast one before heavy use. Each file is written as <name>.part, flushed to
disk and only then renamed, so an interrupted copy is never taken for a
complete one; each file is checked against its sha256 while it is copied.
Running it again copies only what is missing. When the target
location comes earlier in the order, Ollama switches its links to the new
copy.`,
		example: `  ollama cluster copy unsloth/Qwen3-4B-GGUF /mnt/nvme/models`,
	},
	"drop": {
		group: "copies",
		long: `Delete the copy of a repository from one location. Every file of the copy
must exist in another copy with the same sha256: another copy of the same
repository can hold other quantizations or another revision. A file Ollama
loads needs a copy outside archive locations; for the other files an archive
copy is enough. The first check of a copy reads its files whole (a few
minutes for a large model); the sums are kept per location, so later checks
are quick. Without --force, files with no other copy stop the deletion.

The copy is first moved aside, Ollama is relinked to the other copies, and
only then are the files removed. If a link cannot be switched or still
points into the copy, the copy is put back and nothing is deleted.`,
		example: `  ollama cluster drop unsloth/Qwen3-4B-GGUF /mnt/second-disk
  ollama cluster drop unsloth/Qwen3-4B-GGUF /mnt/second-disk --yes`,
	},
	"mv": {
		group: "copies",
		long: `Move the files of models to another location, or back to the Ollama store,
and leave links in Ollama, so a model that is rarely used takes no space on
the fast disk. The model keeps its name and works as before; it loads from
the location, so that disk must be mounted for it to run.

  - <location> is a load location (ollama cluster locations); an archive
    location is refused, Ollama never loads from it. 'store' moves the
    files back into OLLAMA_MODELS.
  - Only the large files move (weights, projector, draft); template,
    parameters and license stay in the store. The files go to
    <location>/ollama/blobs/ under their sha256 names.
  - Ollama does the move itself (it owns the store, also in Docker). Each
    file is copied as <name>.part, checked against its sha256 while it is
    written and flushed to disk, then the blob becomes a link to it, and
    only then is the old file removed. Within one disk the file is hard
    linked instead of copied. An interrupted move changes nothing.
  - A file shared with other models moves for all of them; the plan names
    them. Files of Hugging Face repository copies stay where they are
    (ollama cluster copy / drop).
  - A loaded model keeps running: it holds the old file open.

Models deleted with 'ollama rm' leave their moved files in the location.`,
		example: `  ollama cluster mv qwen3.5:9b /mnt/second-disk/models
  ollama cl mv qwen3.5:9b gemma4:12b llama3.1:8b /mnt/second-disk/models --yes
  ollama cl mv qwen3.5:9b store`,
	},
	"import": {
		group: "copies",
		long: `Turn a Hugging Face cache (the hub/ directory with models--owner--repo/
snapshots/...) into repository copies without copying data.

Each file is moved within the same disk into <location>/hf/<owner>/<repo>/
and a link is left at its old place, so vLLM, llama.cpp or llama-swap using
the cache keep working (in Docker they need the location mounted too).
Files already imported are skipped, so it can be run again. A 140 GB model
moves in under a second. To use an imported GGUF repository in Ollama run
'ollama cluster pull <owner/repo:TAG>': it finds the copy and downloads
nothing.

Run it as the owner of the cache; for a cache owned by root run it as root,
for example in the cluster image with the disk mounted.`,
		example: `  ollama cluster import /mnt/dgx-data/hf/hub
  ollama cluster import /mnt/dgx-data/gguf unsloth/Qwen3-4B-GGUF --yes
  docker run --rm -v /mnt/dgx-data:/mnt/dgx-data --entrypoint /usr/bin/ollama \
    ollama-cluster:0.40.3-cluster cluster import /mnt/dgx-data/hf --to /mnt/dgx-data/models --yes`,
	},
}

// applyHelp sets groups and long help, adds `help` to commands with
// subcommands and makes `<command> help` print the help of that command.
func applyHelp(root *cobra.Command) {
	root.AddGroup(commandGroups...)
	walk(root, func(c *cobra.Command) {
		key := strings.TrimSpace(strings.TrimPrefix(c.CommandPath(), root.CommandPath()))
		if t, ok := helpTexts[key]; ok {
			if c.Parent() == root {
				c.GroupID = t.group
			}
			c.Long = t.long
			c.Example = t.example
		}
		if c.HasSubCommands() {
			c.AddCommand(helpCmd(c))
		}
		helpArg(c)
	})
}

func walk(c *cobra.Command, fn func(*cobra.Command)) {
	subs := c.Commands()
	fn(c)
	for _, s := range subs {
		walk(s, fn)
	}
}

// helpCmd is `help [command]` below a command group, so that
// `ollama cluster help pull` and `ollama cluster locations help` work.
func helpCmd(parent *cobra.Command) *cobra.Command {
	return &cobra.Command{
		Use:   "help [command]",
		Short: "Help about any command",
		RunE: func(cmd *cobra.Command, args []string) error {
			target, rest, err := parent.Find(args)
			if err != nil || len(rest) > 0 || target == nil {
				return fmt.Errorf("unknown command %q for %q", strings.Join(args, " "), parent.CommandPath())
			}
			return target.Help()
		},
	}
}

// helpArg makes `<command> help` print the help of a runnable command
// instead of taking "help" as its argument.
func helpArg(c *cobra.Command) {
	if c.RunE == nil {
		return
	}
	args, run := c.Args, c.RunE
	c.Args = func(cmd *cobra.Command, a []string) error {
		if isHelpArg(a) {
			return nil
		}
		if args == nil {
			return nil
		}
		return args(cmd, a)
	}
	c.RunE = func(cmd *cobra.Command, a []string) error {
		if isHelpArg(a) {
			return cmd.Help()
		}
		return run(cmd, a)
	}
}

func isHelpArg(a []string) bool {
	return len(a) == 1 && a[0] == "help"
}
