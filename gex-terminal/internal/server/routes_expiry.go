package server

import (
	"encoding/json"
	"net/http"

	"gex-terminal/internal/config"
	"gex-terminal/internal/db"
)

func (s *Server) handleExpiryData(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}
	if _, ok := config.TickerConfig[ticker]; !ok {
		writeError(w, 400, "unknown ticker")
		return
	}

	dateParam := r.URL.Query().Get("date")
	expiryParam := r.URL.Query().Get("expiry")
	fromDate := r.URL.Query().Get("from_date")
	toDate := r.URL.Query().Get("to_date")

	// Mode 1: List available expiries
	if expiryParam == "" && fromDate == "" {
		expiries, err := db.GetExpiryCacheList(ticker, dateParam)
		if err != nil {
			writeError(w, 500, err.Error())
			return
		}
		if expiries == nil {
			expiries = []string{}
		}
		writeJSON(w, 200, map[string]interface{}{
			"date":     dateParam,
			"expiries": expiries,
		})
		return
	}

	// Resolve the cache date
	cacheDate := dateParam
	if cacheDate == "" {
		expiries, _ := db.GetExpiryCacheList(ticker, "")
		if len(expiries) == 0 {
			writeJSON(w, 200, map[string]interface{}{
				"date":     "",
				"expiries": []string{},
			})
			return
		}
		// Need to get the actual date from the cache
		// For now, return empty
	}

	// Mode 2: Single expiry
	if expiryParam != "" && cacheDate != "" {
		raw, err := db.GetExpiryCacheData(ticker, cacheDate, expiryParam)
		if err != nil {
			writeError(w, 500, err.Error())
			return
		}
		if raw == nil {
			writeError(w, 404, "expiry not found")
			return
		}
		writeRawJSON(w, 200, raw)
		return
	}

	// Mode 3: Date range
	if fromDate != "" && toDate != "" && cacheDate != "" {
		items, err := db.GetExpiryCacheRange(ticker, cacheDate, fromDate, toDate)
		if err != nil {
			writeError(w, 500, err.Error())
			return
		}

		// Merge all expiry data
		allGEX := []json.RawMessage{}
		includedExpiries := []string{}
		for _, item := range items {
			includedExpiries = append(includedExpiries, item.ExpiryDate)
			allGEX = append(allGEX, item.DataJSON)
		}

		writeJSON(w, 200, map[string]interface{}{
			"date":              cacheDate,
			"included_expiries": includedExpiries,
			"expiry_data":       allGEX,
		})
		return
	}

	writeError(w, 400, "invalid parameters")
}
