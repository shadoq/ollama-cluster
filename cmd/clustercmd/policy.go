package clustercmd

import (
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cluster"
)

// --- policy -----------------------------------------------------------------

func policyCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "policy",
		Short: "Show or set where models go: this machine, split, or the worker first",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			m, err := clusterMain(cmd)
			if err != nil {
				return err
			}
			printPolicy(&m.status.Config)
			return nil
		},
	}

	var from, keepFree float64
	set := &cobra.Command{
		Use:   "set auto|spread|worker-first",
		Short: "Set the policy for all models",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			m, err := clusterMain(cmd)
			if err != nil {
				return err
			}
			cfg := m.config()
			cfg.Policy = args[0]
			if cmd.Flags().Changed("from") {
				cfg.WorkerFirstFrom = gbToBytes(from)
			}
			if cmd.Flags().Changed("keep-free-here") {
				cfg.KeepFreeHere = gbToBytes(keepFree)
			}
			if err := cfg.Validate(); err != nil {
				return err
			}
			if err := m.api.saveConfig(cmd.Context(), cfg); err != nil {
				return err
			}
			printPolicy(cfg)
			return nil
		},
	}
	set.Flags().Float64Var(&from, "from", 0, "worker-first: model size in GB from which models go to the worker (0: every model)")
	set.Flags().Float64Var(&keepFree, "keep-free-here", 0, "memory in GB that models never take on this machine (0: none)")

	tensor := &cobra.Command{
		Use:   "tensor on|off",
		Short: "Run models with tensor parallelism across both machines by default",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			var on bool
			switch args[0] {
			case "on":
				on = true
			case "off":
			default:
				return fmt.Errorf("tensor: %q is not on or off", args[0])
			}
			m, err := clusterMain(cmd)
			if err != nil {
				return err
			}
			cfg := m.config()
			cfg.Tensor = on
			if err := m.api.saveConfig(cmd.Context(), cfg); err != nil {
				return err
			}
			printPolicy(cfg)
			return nil
		},
	}

	model := &cobra.Command{
		Use:   "model <name> here|split|worker|tensor|auto",
		Short: "Fix where one model goes (auto removes the rule)",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			m, err := clusterMain(cmd)
			if err != nil {
				return err
			}
			name, place := args[0], args[1]
			if !cluster.ValidPlacement(place) {
				return fmt.Errorf("placement must be %s, %s, %s, %s or %s", cluster.PlaceHere, cluster.PlaceSplit, cluster.PlaceWorker, cluster.PlaceTensor, cluster.PlaceAuto)
			}
			cfg := m.config()
			models := make(map[string]string, len(cfg.Models)+1)
			for k, v := range cfg.Models {
				if k != name && k != name+":latest" && k+":latest" != name {
					models[k] = v
				}
			}
			if place != cluster.PlaceAuto {
				models[name] = place
			}
			cfg.Models = models
			if err := m.api.saveConfig(cmd.Context(), cfg); err != nil {
				return err
			}
			if !m.installed(cmd, name) {
				fmt.Printf("Note: %s is not installed; the rule applies once it is.\n", name)
			}
			printPolicy(cfg)
			return nil
		},
	}

	cmd.AddCommand(set, tensor, model)
	return cmd
}

// installed reports whether Ollama lists the model; true when it cannot tell.
func (m *mainOllama) installed(cmd *cobra.Command, name string) bool {
	var tags api.ListResponse
	if err := m.api.do(cmd.Context(), http.MethodGet, "/api/tags", nil, &tags); err != nil {
		return true
	}
	for _, t := range tags.Models {
		if t.Name == name || t.Name == name+":latest" || strings.TrimSuffix(t.Name, ":latest") == name {
			return true
		}
	}
	return false
}

func gbToBytes(gb float64) uint64 {
	if gb <= 0 {
		return 0
	}
	return uint64(gb * 1e9)
}

// policySummary is the one-line description used by status and policy.
func policySummary(c *cluster.Config) string {
	var s string
	switch {
	case c.Policy == cluster.PolicyWorkerFirst && c.WorkerFirstFrom > 0:
		s = fmt.Sprintf("worker-first: models from %s go to the worker first, smaller ones stay here", humanBytes(c.WorkerFirstFrom))
	case c.Policy == cluster.PolicyWorkerFirst:
		s = "worker-first: every model goes to the worker first"
	case c.Spread():
		s = "spread: every model is split across all machines"
	default:
		s = "auto: models that fit here stay here, larger ones are split"
	}
	if c.KeepFreeHere > 0 {
		s += fmt.Sprintf("; %s kept free here", humanBytes(c.KeepFreeHere))
	}
	if c.Tensor {
		s += "; tensor parallel by default"
	}
	if n := len(c.Models); n > 0 {
		s += fmt.Sprintf("; %d model rule(s)", n)
	}
	return s
}

func printPolicy(c *cluster.Config) {
	fmt.Printf("Policy    %s\n", policySummary(c))
	if c.KeepFreeHere > 0 {
		fmt.Printf("\nModels never take the last %s of memory on this machine. A model that\ndoes not fit in the rest is split, and the split leaves that memory free.\n", humanBytes(c.KeepFreeHere))
	}
	if c.Tensor {
		fmt.Println("\nModels without a rule of their own run with tensor parallelism: both machines")
		fmt.Println("compute every layer, half of the model on each, which generates faster than")
		fmt.Println("one machine. The model takes a slot of the worker even when it fits here.")
		fmt.Println("While all slots of the worker hold a model, or half of the model does not fit")
		fmt.Println("on one of the machines, the policy above decides.")
	}
	if c.Policy == cluster.PolicyWorkerFirst {
		fmt.Println("\nA model that fits on the worker loads there only; a larger one is split")
		fmt.Println("with the worker filled first. When all slots of the worker hold a model,")
		fmt.Println("the model loads here as under auto.")
	}
	if len(c.Models) == 0 {
		return
	}
	fmt.Println("\nModel rules")
	names := make([]string, 0, len(c.Models))
	for n := range c.Models {
		names = append(names, n)
	}
	sort.Strings(names)
	w := 0
	for _, n := range names {
		w = max(w, len(n))
	}
	for _, n := range names {
		fmt.Printf("  %-*s  %s\n", w, n, placementText(c.Models[n]))
	}
}

func placementText(p string) string {
	switch p {
	case cluster.PlaceHere:
		return "here: this machine only"
	case cluster.PlaceSplit:
		return "split: across this machine and the worker"
	case cluster.PlaceWorker:
		return "worker: on the worker, split with the worker filled first when too large"
	case cluster.PlaceTensor:
		return "tensor: tensor parallel across both machines, else as the policy says"
	}
	return p
}
