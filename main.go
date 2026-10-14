package main

import (
	"context"
	_ "embed"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/cmd"
)

//go:embed LLAMA_CPP_VERSION
var llamaCppVersion string

func main() {
	cluster.LlamaCppVersion = strings.TrimSpace(llamaCppVersion)
	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}
