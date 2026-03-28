package main

import (
	"flag"
	"log"
	"net/http"
	"os"

	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/config"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/proxy"
)

func main() {
	configPath := flag.String("config", "", "path to config.json (default: config.json or FEMTOLLM_CONFIG)")
	flag.Parse()

	path := *configPath
	if path == "" {
		path = os.Getenv("FEMTOLLM_CONFIG")
	}
	if path == "" {
		path = "config.json"
	}

	cfg, err := config.Load(path)
	if err != nil {
		log.Fatalf("Failed to load config from %s: %v", path, err)
	}

	log.Printf("femtollm starting on %s with %d backend(s)", cfg.Listen, len(cfg.Backends))
	for _, b := range cfg.Backends {
		log.Printf("  backend %s: pattern=%s url=%s", b.Name, b.Pattern, b.URL)
	}

	srv := proxy.NewServer(cfg)
	if err := http.ListenAndServe(cfg.Listen, srv.Handler()); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
