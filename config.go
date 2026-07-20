package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// ── Config types ─────────────────────────────

type Config struct {
	Server  ServerConfig  `yaml:"server"`
	Backend BackendConfig `yaml:"backend"`
	Models  []ModelConfig `yaml:"models"`
}

type ServerConfig struct {
	Host                 string `yaml:"host"`
	Port                 int    `yaml:"port"`
	BackendPortStart     int    `yaml:"backend_port_start"`
	BackendPortEnd       int    `yaml:"backend_port_end"`
	MaxModels            int    `yaml:"max_models"`
	IdleTimeoutMinutes   int    `yaml:"idle_timeout_minutes"`
	HealthTimeoutSeconds int    `yaml:"health_timeout_seconds"`
	SweepIntervalSeconds int    `yaml:"sweep_interval_seconds"`
}

type BackendConfig struct {
	Binary     string            `yaml:"binary"`
	Env        map[string]string `yaml:"env"`
	CommonArgs []string          `yaml:"common_args"`
	Workdir    string            `yaml:"workdir"`
	NvidiaSMI  string            `yaml:"nvidia_smi"`
}

type ModelConfig struct {
	ID             string            `yaml:"id"`
	Model          string            `yaml:"model"`
	Path           string            `yaml:"path"`
	Mmproj         string            `yaml:"mmproj"`
	Alias          string            `yaml:"alias"`
	ContextSize    int               `yaml:"context_size"`
	Parallel       int               `yaml:"parallel"`
	Devices        []string          `yaml:"devices"`
	TensorSplit    string            `yaml:"tensor_split"`
	CtxCheckpoints int               `yaml:"ctx_checkpoints"`
	ExtraArgs      []string          `yaml:"extra_args"`

	// ── Per-model runtime override (for non-llama-server backends) ──
	// When Binary is set, it overrides backend.binary for this model.
	// When Args is set, it replaces BuildArgs() output entirely (no
	// -m/--mmproj/-c/etc flags are generated). The {port} placeholder
	// in each arg is replaced with the dynamically assigned port.
	// HealthPath overrides the /health endpoint (default: /health).
	// Env is merged on top of backend.env for per-model overrides.
	Binary     string            `yaml:"binary"`
	Args       []string          `yaml:"args"`
	HealthPath string            `yaml:"health_path"`
	Env        map[string]string `yaml:"env"`
}

// ── Path & env expansion ─────────────────────

// expand resolves ~ and $VAR in a string. Handles colon-separated
// path lists (each component is expanded independently).
func expand(s string) string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = os.Getenv("HOME")
	}

	parts := strings.Split(s, ":")
	for i, part := range parts {
		switch {
		case part == "~":
			parts[i] = home
		case strings.HasPrefix(part, "~/"):
			parts[i] = filepath.Join(home, part[2:])
		}
		parts[i] = os.Expand(parts[i], os.Getenv)
	}

	if len(parts) == 1 {
		return parts[0]
	}
	return strings.Join(parts, ":")
}

// ── Loading ──────────────────────────────────

func LoadConfig(path string) (*Config, error) {
	path = expand(path)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}

	// Apply defaults
	if cfg.Server.Port == 0 {
		cfg.Server.Port = 8080
	}
	if cfg.Server.Host == "" {
		cfg.Server.Host = "127.0.0.1"
	}
	if cfg.Server.BackendPortStart == 0 {
		cfg.Server.BackendPortStart = 8201
	}
	if cfg.Server.BackendPortEnd == 0 {
		cfg.Server.BackendPortEnd = 8299
	}
	if cfg.Server.MaxModels == 0 {
		cfg.Server.MaxModels = 2
	}
	if cfg.Server.IdleTimeoutMinutes == 0 {
		cfg.Server.IdleTimeoutMinutes = 60
	}
	if cfg.Server.HealthTimeoutSeconds == 0 {
		cfg.Server.HealthTimeoutSeconds = 180
	}
	if cfg.Server.SweepIntervalSeconds == 0 {
		cfg.Server.SweepIntervalSeconds = 15
	}
	if cfg.Backend.NvidiaSMI == "" {
		cfg.Backend.NvidiaSMI = "nvidia-smi"
	}

	return &cfg, nil
}

// Validate checks the config for logical errors and returns the first issue found.
func (c *Config) Validate() error {
	if c.Server.BackendPortStart >= c.Server.BackendPortEnd {
		return fmt.Errorf("backend_port_start (%d) must be less than backend_port_end (%d)",
			c.Server.BackendPortStart, c.Server.BackendPortEnd)
	}
	if c.Server.Port >= c.Server.BackendPortStart && c.Server.Port <= c.Server.BackendPortEnd {
		return fmt.Errorf("server port %d overlaps with backend port range %d-%d",
			c.Server.Port, c.Server.BackendPortStart, c.Server.BackendPortEnd)
	}
	if c.Server.MaxModels < 1 {
		return fmt.Errorf("max_models must be at least 1, got %d", c.Server.MaxModels)
	}
	seenID := make(map[string]bool, len(c.Models))
	seenModel := make(map[string]bool, len(c.Models))
	seenAlias := make(map[string]bool, len(c.Models))
	for i := range c.Models {
		m := &c.Models[i]
		if m.ID == "" {
			return fmt.Errorf("model at index %d has empty id", i)
		}
		if seenID[m.ID] {
			return fmt.Errorf("duplicate model id: %s", m.ID)
		}
		seenID[m.ID] = true

		// For llama-server models (no custom binary/args), Path is required.
		// Custom-binary models may not have a local path.
		if m.Binary == "" && len(m.Args) == 0 && m.Path == "" {
			return fmt.Errorf("model %s has empty path (required for llama-server models)", m.ID)
		}

		if m.Model != "" {
			if seenModel[m.Model] {
				return fmt.Errorf("duplicate model display name: %s", m.Model)
			}
			seenModel[m.Model] = true
		}
		if m.Alias != "" {
			if seenAlias[m.Alias] {
				return fmt.Errorf("duplicate model alias: %s", m.Alias)
			}
			seenAlias[m.Alias] = true
		}
	}
	return nil
}

// findModel looks up a model by ID, alias, or display name.
func (c *Config) FindModel(id string) *ModelConfig {
	for i := range c.Models {
		m := &c.Models[i]
		if m.ID == id || m.Alias == id || m.Model == id {
			return m
		}
	}
	return nil
}

// ── Argument building ────────────────────────

// BuildArgs constructs the CLI args for launching this model's backend.
// If m.Args (the "args" YAML field) is set, it is used verbatim (after
// expansion and {port} substitution), bypassing all llama-server-specific
// flag construction. Otherwise, llama-server-style args are built from the
// individual model fields.
func (m *ModelConfig) BuildArgs(common []string, port int) []string {
	// Raw args mode: use the args list directly, substituting {port}.
	if len(m.Args) > 0 {
		args := make([]string, 0, len(m.Args))
		for _, a := range m.Args {
			a = expand(a)
			a = strings.ReplaceAll(a, "{port}", strconv.Itoa(port))
			args = append(args, a)
		}
		return args
	}

	// llama-server mode: build args from individual fields.
	args := []string{"-m", expand(m.Path)}

	if m.Mmproj != "" {
		args = append(args, "--mmproj", expand(m.Mmproj))
	}
	if m.Model != "" {
		args = append(args, "--alias", m.Model)
	}
	if m.ContextSize != 0 {
		args = append(args, "-c", strconv.Itoa(m.ContextSize))
	}
	if m.Parallel != 0 {
		args = append(args, "--parallel", strconv.Itoa(m.Parallel))
	}
	if len(m.Devices) > 0 {
		args = append(args, "--device", strings.Join(m.Devices, ","))
	}
	if m.TensorSplit != "" {
		args = append(args, "--tensor-split", m.TensorSplit)
	}
	if m.CtxCheckpoints != 0 {
		args = append(args, "--ctx-checkpoints", strconv.Itoa(m.CtxCheckpoints))
	}

	// Dynamic host/port (always loopback for backend)
	args = append(args, "--host", "127.0.0.1", "--port", strconv.Itoa(port))

	// Common args from config
	for _, a := range common {
		args = append(args, a)
	}

	// Model-specific extra args
	args = append(args, m.ExtraArgs...)

	return args
}

// ResolveBinary returns the binary path to use for this model. If the
// model has a per-model Binary override, it is used (with expansion and
// PATH lookup). Otherwise the backend-level binary is resolved.
func (m *ModelConfig) ResolveBinary(b *BackendConfig) (string, error) {
	if m.Binary != "" {
		bin := expand(m.Binary)
		if filepath.IsAbs(bin) {
			return bin, nil
		}
		return exec.LookPath(bin)
	}
	return b.ResolveBinary()
}

// HealthEndpoint returns the health check path for this model, defaulting
// to "/health" if not overridden.
func (m *ModelConfig) HealthEndpoint() string {
	if m.HealthPath != "" {
		return m.HealthPath
	}
	return "/health"
}

// BuildModelEnv constructs the full environment for the backend process,
// starting from the current environment, overlaying backend-level env
// vars, and finally overlaying per-model env vars. Per-model env takes
// precedence over backend env, which takes precedence over the OS env.
func (m *ModelConfig) BuildModelEnv(b *BackendConfig) []string {
	envMap := make(map[string]string)
	for _, e := range os.Environ() {
		if idx := strings.IndexByte(e, '='); idx >= 0 {
			envMap[e[:idx]] = e[idx+1:]
		}
	}
	// Backend env (lower priority)
	for k, v := range b.Env {
		envMap[k] = expand(v)
	}
	// Per-model env (higher priority)
	for k, v := range m.Env {
		envMap[k] = expand(v)
	}

	env := make([]string, 0, len(envMap))
	for k, v := range envMap {
		env = append(env, k+"="+v)
	}
	return env
}

// resolveBinary returns the expanded binary path, checking PATH if needed.
func (b *BackendConfig) ResolveBinary() (string, error) {
	bin := expand(b.Binary)
	if filepath.IsAbs(bin) {
		return bin, nil
	}
	return exec.LookPath(bin)
}
