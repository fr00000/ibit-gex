package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"gex-terminal/internal/background"
	"gex-terminal/internal/config"
	"gex-terminal/internal/db"
	"gex-terminal/internal/server"
)

func main() {
	host := flag.String("host", "", "listen address (default from env or 127.0.0.1)")
	port := flag.String("port", "", "listen port (default from env or 5001)")
	flag.Parse()

	// Setup structured logging
	logHandler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	})
	slog.SetDefault(slog.New(logHandler))

	// Load configuration
	config.Load()
	if *host != "" {
		config.ServerHost = *host
	}
	if *port != "" {
		config.ServerPort = *port
	}

	slog.Info("starting gex-terminal",
		"db", config.DBPath,
		"host", config.ServerHost,
		"port", config.ServerPort,
		"anthropic_key", config.AnthropicAPIKey != "",
		"coinglass_key", config.CoinglassAPIKey != "",
	)

	// Initialize database
	if err := db.Init(); err != nil {
		slog.Error("database init failed", "err", err)
		os.Exit(1)
	}
	defer db.Close()

	// Create server
	templateDir := filepath.Join(".", "templates")
	srv, err := server.New(templateDir)
	if err != nil {
		slog.Error("server init failed", "err", err)
		os.Exit(1)
	}

	// Start background refresh goroutine
	bgCtx, bgCancel := context.WithCancel(context.Background())
	defer bgCancel()
	go background.Run(bgCtx)

	// HTTP server with graceful shutdown
	addr := fmt.Sprintf("%s:%s", config.ServerHost, config.ServerPort)
	httpServer := &http.Server{
		Addr:         addr,
		Handler:      srv.Handler(),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Graceful shutdown on SIGINT/SIGTERM
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		slog.Info("server listening", "addr", addr)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	slog.Info("shutting down...")

	// Cancel background refresh first
	bgCancel()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		slog.Error("shutdown error", "err", err)
	}
	slog.Info("server stopped")
}
