package server

import (
	"encoding/json"
	"html/template"
	"log/slog"
	"net/http"
	"path/filepath"
	"runtime"
	"time"
)

// Server holds the HTTP server and its dependencies.
type Server struct {
	mux       *http.ServeMux
	templates *template.Template
}

// New creates a new server with all routes registered.
func New(templateDir string) (*Server, error) {
	s := &Server{
		mux: http.NewServeMux(),
	}

	// Parse templates
	tmplPath := filepath.Join(templateDir, "*.html")
	var err error
	s.templates, err = template.ParseGlob(tmplPath)
	if err != nil {
		return nil, err
	}

	s.registerRoutes()
	return s, nil
}

func (s *Server) registerRoutes() {
	// Page routes
	s.mux.HandleFunc("/", s.handleIndex)
	s.mux.HandleFunc("/macro", s.handleMacro)
	s.mux.HandleFunc("/m", s.handleMobile)

	// Data API routes
	s.mux.HandleFunc("/api/snapshot/dates", s.handleSnapshotDates)
	s.mux.HandleFunc("/api/data", s.handleData)
	s.mux.HandleFunc("/api/outlook", s.handleOutlook)
	s.mux.HandleFunc("/api/range-cone", s.handleRangeCone)
	s.mux.HandleFunc("/api/structure", s.handleStructure)
	s.mux.HandleFunc("/api/structure/heatmap", s.handleStructureHeatmap)
	s.mux.HandleFunc("/api/candles", s.handleCandles)
	s.mux.HandleFunc("/api/btc-candles", s.handleCandles) // alias
	s.mux.HandleFunc("/api/flows", s.handleFlows)
	s.mux.HandleFunc("/api/expiry-data", s.handleExpiryData)

	// Analysis routes
	s.mux.HandleFunc("/api/analysis", s.handleAnalysis)
	s.mux.HandleFunc("/api/analysis-data", s.handleAnalysisData)
	s.mux.HandleFunc("/api/analyze", s.handleAnalyze)
	s.mux.HandleFunc("/api/analysis/save", s.handleAnalysisSave)
	s.mux.HandleFunc("/api/accuracy", s.handleAccuracy)
	s.mux.HandleFunc("/api/macro-regime", s.handleMacroRegime)
}

// Handler returns the http.Handler with middleware applied.
func (s *Server) Handler() http.Handler {
	return withLogging(withCORS(s.mux))
}

// Page handlers

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	s.templates.ExecuteTemplate(w, "index.html", nil)
}

func (s *Server) handleMacro(w http.ResponseWriter, r *http.Request) {
	s.templates.ExecuteTemplate(w, "macro.html", nil)
}

func (s *Server) handleMobile(w http.ResponseWriter, r *http.Request) {
	s.templates.ExecuteTemplate(w, "mobile.html", nil)
}

// JSON helpers

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeRawJSON(w http.ResponseWriter, status int, raw json.RawMessage) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(raw)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// Middleware

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, X-Admin-Key")
		if r.Method == "OPTIONS" {
			w.WriteHeader(204)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		slog.Debug("request",
			"method", r.Method,
			"path", r.URL.Path,
			"query", r.URL.RawQuery,
			"duration", time.Since(start),
			"goroutines", runtime.NumGoroutine(),
		)
	})
}
