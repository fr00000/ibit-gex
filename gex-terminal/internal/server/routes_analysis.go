package server

import (
	"encoding/json"
	"io"
	"net/http"

	"gex-terminal/internal/analysis"
	"gex-terminal/internal/config"
	"gex-terminal/internal/db"
)

func (s *Server) handleAnalysis(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}
	if _, ok := config.TickerConfig[ticker]; !ok {
		writeError(w, 400, "unknown ticker")
		return
	}

	dateParam := r.URL.Query().Get("date")
	_, raw, err := db.GetAnalysisCache(ticker, dateParam)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	if raw == nil {
		if dateParam != "" {
			writeError(w, 404, "no analysis for that date")
			return
		}
		writeJSON(w, 200, map[string]string{"status": "pending"})
		return
	}
	writeRawJSON(w, 200, raw)
}

func (s *Server) handleAnalysisData(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}
	if _, ok := config.TickerConfig[ticker]; !ok {
		writeError(w, 400, "unknown ticker")
		return
	}

	data, err := analysis.BuildAnalysisData(ticker)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	if data == nil {
		writeJSON(w, 200, map[string]string{"status": "no data"})
		return
	}
	writeJSON(w, 200, data)
}

func (s *Server) handleAnalyze(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		writeError(w, 405, "method not allowed")
		return
	}

	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}
	if _, ok := config.TickerConfig[ticker]; !ok {
		writeError(w, 400, "unknown ticker")
		return
	}

	// Check admin key
	if config.AdminKey != "" {
		if r.Header.Get("X-Admin-Key") != config.AdminKey {
			writeError(w, 403, "invalid admin key")
			return
		}
	}

	result, err := analysis.RunAnalysis(ticker, true)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, result)
}

func (s *Server) handleAnalysisSave(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		writeError(w, 405, "method not allowed")
		return
	}

	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}

	if config.AdminKey != "" {
		if r.Header.Get("X-Admin-Key") != config.AdminKey {
			writeError(w, 403, "invalid admin key")
			return
		}
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, 400, "cannot read body")
		return
	}

	var data map[string]interface{}
	if err := json.Unmarshal(body, &data); err != nil {
		writeError(w, 400, "invalid JSON")
		return
	}

	if err := db.SetAnalysisCache(ticker, data); err != nil {
		writeError(w, 500, err.Error())
		return
	}

	writeJSON(w, 200, map[string]string{"status": "saved"})
}
