package server

import (
	"net/http"
	"strconv"

	"gex-terminal/internal/config"
	"gex-terminal/internal/db"
	"gex-terminal/internal/macro"
)

func (s *Server) handleCandles(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}
	tc, ok := config.TickerConfig[ticker]
	if !ok {
		writeError(w, 400, "unknown ticker")
		return
	}

	tf := r.URL.Query().Get("tf")
	if tf == "" {
		tf = "15m"
	}
	switch tf {
	case "15m", "1h", "4h", "1d":
	default:
		writeError(w, 400, "invalid timeframe; use 15m, 1h, 4h, or 1d")
		return
	}

	candles, err := db.GetCandles(tc.BinanceSymbol, tf)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	if candles == nil {
		// Backfill not done yet
		writeError(w, 503, "candle backfill in progress")
		return
	}
	writeJSON(w, 200, candles)
}

func (s *Server) handleFlows(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}

	flows, err := db.GetFlows(ticker, 30)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	if flows == nil {
		writeJSON(w, 200, []interface{}{})
		return
	}
	writeJSON(w, 200, flows)
}

func (s *Server) handleAccuracy(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}

	recent, err := db.GetRecentPredictions(ticker, 50)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}

	// Build convergence buckets
	buckets := map[string]struct {
		Total         int
		CWHeld        int
		PWHeld        int
		RangeHeld     int
		EMHeld        int
		RegimeCorrect int
	}{}

	for _, p := range recent {
		bucket := dteBucket(p.DTE)
		b := buckets[bucket]
		b.Total++
		if p.CallWallHeld != nil && *p.CallWallHeld { b.CWHeld++ }
		if p.PutWallHeld != nil && *p.PutWallHeld { b.PWHeld++ }
		if p.RangeHeld != nil && *p.RangeHeld { b.RangeHeld++ }
		if p.EMHeld != nil && *p.EMHeld { b.EMHeld++ }
		if p.RegimeCorrect != nil && *p.RegimeCorrect { b.RegimeCorrect++ }
		buckets[bucket] = b
	}

	convergence := map[string]interface{}{}
	for k, b := range buckets {
		if b.Total == 0 {
			continue
		}
		convergence[k] = map[string]interface{}{
			"total":              b.Total,
			"call_wall_held_pct": pct(b.CWHeld, b.Total),
			"put_wall_held_pct":  pct(b.PWHeld, b.Total),
			"range_held_pct":     pct(b.RangeHeld, b.Total),
			"em_held_pct":        pct(b.EMHeld, b.Total),
			"regime_correct_pct": pct(b.RegimeCorrect, b.Total),
		}
	}

	// Build recent results for display
	recentOut := make([]map[string]interface{}, 0, len(recent))
	for _, p := range recent {
		entry := map[string]interface{}{
			"analysis_date": p.AnalysisDate,
			"expiry_date":   p.ExpiryDate,
			"dte":           p.DTE,
			"window":        p.DTEWindow,
			"spot":          p.SpotBTC,
			"call_wall":     p.CallWallBTC,
			"put_wall":      p.PutWallBTC,
			"regime":        p.Regime,
		}
		if p.BTCCloseOnExpiry != nil { entry["btc_close"] = *p.BTCCloseOnExpiry }
		if p.RangeHeld != nil { entry["range_held"] = *p.RangeHeld }
		if p.RegimeCorrect != nil { entry["regime_correct"] = *p.RegimeCorrect }
		if p.VenueWallsAgree != nil { entry["venue_agree"] = *p.VenueWallsAgree }
		if p.VenueAgreeHeld != nil { entry["venue_agree_held"] = *p.VenueAgreeHeld }
		if p.AIBottomLine != "" { entry["ai_bottom_line"] = p.AIBottomLine }
		recentOut = append(recentOut, entry)
	}

	writeJSON(w, 200, map[string]interface{}{
		"convergence":    convergence,
		"expiry_history": []interface{}{}, // TODO: populate in Phase 5
		"recent":         recentOut,
	})
}

func (s *Server) handleMacroRegime(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}
	if _, ok := config.TickerConfig[ticker]; !ok {
		writeError(w, 400, "unknown ticker")
		return
	}

	result := macro.ComputeMacroRegime(ticker, 30)
	writeJSON(w, 200, result)
}

func dteBucket(dte int) string {
	switch {
	case dte <= 1:
		return "0-1"
	case dte <= 3:
		return "2-3"
	case dte <= 7:
		return "4-7"
	case dte <= 14:
		return "8-14"
	case dte <= 30:
		return "15-30"
	default:
		return "31-45"
	}
}

func pct(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(n) / float64(total) * 100
}

func atoi(s string, def int) int {
	if v, err := strconv.Atoi(s); err == nil {
		return v
	}
	return def
}
