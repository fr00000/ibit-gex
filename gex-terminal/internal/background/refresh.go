package background

import (
	"context"
	"encoding/json"
	"log/slog"
	"math"
	"time"

	"gex-terminal/internal/analysis"
	"gex-terminal/internal/bs"
	"gex-terminal/internal/config"
	"gex-terminal/internal/db"
	"gex-terminal/internal/fetch"
	"gex-terminal/internal/gex"
	"gex-terminal/internal/model"
)

func nowET() time.Time {
	loc, _ := time.LoadLocation("America/New_York")
	return time.Now().In(loc)
}

// Run starts the background refresh loop. It blocks until ctx is cancelled.
func Run(ctx context.Context) {
	slog.Info("[bg-refresh] starting background refresh")

	// Phase 0: Fetch Farside ETF flows
	if _, err := fetch.FetchFarsideFlows(); err != nil {
		slog.Error("[bg-refresh] Farside flow fetch error", "err", err)
	}

	// Phase 0b: Fetch Coinglass data
	if err := fetch.FetchCoinglassData(); err != nil {
		slog.Error("[bg-refresh] Coinglass data fetch error", "err", err)
	}

	// Phase 1: Backfill candles for all tickers
	for tk, cfg := range config.TickerConfig {
		slog.Info("[bg-refresh] starting candle backfill", "symbol", cfg.BinanceSymbol, "ticker", tk)
		for _, tf := range []string{"15m", "1h", "4h", "1d"} {
			if err := fetch.BackfillCandles(cfg.BinanceSymbol, tf, 90); err != nil {
				slog.Error("[bg-refresh] backfill error", "symbol", cfg.BinanceSymbol, "tf", tf, "err", err)
			}
		}
		slog.Info("[bg-refresh] candle backfill complete", "symbol", cfg.BinanceSymbol)
	}

	// Phase 2: Main loop — 5 min cadence
	lastCandleUpdate := time.Time{}
	postCloseDone := map[string]string{} // ticker -> date string

	ticker := time.NewTicker(10 * time.Second) // initial check quickly, then sleep at bottom
	defer ticker.Stop()

	// Run first iteration immediately
	runLoop(ctx, &lastCandleUpdate, postCloseDone)

	for {
		select {
		case <-ctx.Done():
			slog.Info("[bg-refresh] stopping background refresh")
			return
		case <-ticker.C:
			runLoop(ctx, &lastCandleUpdate, postCloseDone)
			// Reset ticker to 5 minutes after first fast iteration
			ticker.Reset(5 * time.Minute)
		}
	}
}

func runLoop(ctx context.Context, lastCandleUpdate *time.Time, postCloseDone map[string]string) {
	// Cleanup old expiry_cache rows
	if err := db.CleanExpiryCacheOld(90, 7); err != nil {
		slog.Debug("[bg-refresh] expiry cleanup error", "err", err)
	}

	// Update candles every 5 minutes
	if time.Since(*lastCandleUpdate) >= 5*time.Minute {
		for _, cfg := range config.TickerConfig {
			for _, tf := range []string{"15m", "1h", "4h", "1d"} {
				if err := fetch.UpdateCandles(cfg.BinanceSymbol, tf); err != nil {
					slog.Error("[bg-refresh] candle update error", "symbol", cfg.BinanceSymbol, "tf", tf, "err", err)
				}
			}
		}
		*lastCandleUpdate = time.Now()
	}

	// Refresh Farside ETF flows
	if _, err := fetch.FetchFarsideFlows(); err != nil {
		slog.Error("[bg-refresh] Farside flow error", "err", err)
	}

	// GEX refresh for all tickers
	for tk, cfg := range config.TickerConfig {
		if ctx.Err() != nil {
			return
		}
		refreshTicker(ctx, tk, cfg, postCloseDone)
	}

	// Score expired predictions
	for tk := range config.TickerConfig {
		scored, err := ScoreExpiredPredictions(tk)
		if err != nil {
			slog.Error("[bg-refresh] prediction scoring error", "ticker", tk, "err", err)
		} else if scored > 0 {
			slog.Info("[bg-refresh] scored predictions", "ticker", tk, "count", scored)
		}
	}
}

func refreshTicker(ctx context.Context, tk string, cfg *config.TickerCfg, postCloseDone map[string]string) {
	today := nowET().Format("2006-01-02")
	allFresh := true

	// Check which DTE windows are stale
	type staleWin struct {
		label, min, max int
	}
	var staleWindows []staleWin

	for _, w := range config.DTEWindows {
		cacheDate, raw, err := db.GetLatestCache(tk, w.MaxDTE, "")
		if err != nil {
			slog.Error("[bg-refresh] cache check error", "ticker", tk, "dte", w.MaxDTE, "err", err)
			continue
		}
		if cacheDate != today {
			staleWindows = append(staleWindows, staleWin{w.Label, w.MinDTE, w.MaxDTE})
			continue
		}
		// Also check that min_dte matches
		if raw != nil {
			var blob struct {
				DTEWindow struct {
					Min int `json:"min"`
				} `json:"dte_window"`
			}
			if json.Unmarshal(raw, &blob) == nil && blob.DTEWindow.Min != w.MinDTE {
				staleWindows = append(staleWindows, staleWin{w.Label, w.MinDTE, w.MaxDTE})
			}
		}
	}

	if len(staleWindows) > 0 {
		allFresh = false
		// Fetch stale windows sequentially
		for _, sw := range staleWindows {
			if ctx.Err() != nil {
				return
			}
			if _, err := gex.FetchWithCache(tk, sw.max, sw.min, false); err != nil {
				slog.Error("[bg-refresh] fetch error", "ticker", tk, "dte_min", sw.min, "dte_max", sw.max, "err", err)
			}
		}
	}

	// If all fresh, check for >2% move to re-run analysis
	if allFresh {
		checkLargeMove(tk, cfg)
	}

	// Deribit intraday refresh
	if cfg.DeribitCurrency != "" {
		deribitRefresh(ctx, tk, cfg, allFresh)
	}

	// Post-close analysis (4:20 PM ET on weekdays)
	t := nowET()
	isTradingDay := t.Weekday() != time.Saturday && t.Weekday() != time.Sunday
	postCloseWindow := (t.Hour() == 16 && t.Minute() >= 20) || t.Hour() == 17
	alreadyDoneToday := postCloseDone[tk] == today

	if isTradingDay && postCloseWindow && !alreadyDoneToday {
		slog.Info("[bg-refresh] post-close refresh", "ticker", tk)

		// Force-refresh all DTE windows with fresh IBIT data
		for _, w := range config.DTEWindows {
			if _, err := gex.FetchWithCache(tk, w.MaxDTE, w.MinDTE, true); err != nil {
				slog.Error("[bg-refresh] post-close DTE error", "ticker", tk, "dte_min", w.MinDTE, "dte_max", w.MaxDTE, "err", err)
			}
		}

		// Run AI analysis
		if _, err := analysis.RunAnalysis(tk, true); err != nil {
			slog.Error("[bg-refresh] post-close analysis error", "ticker", tk, "err", err)
		} else {
			slog.Info("[bg-refresh] post-close analysis complete", "ticker", tk)
			savePredictionsFromCache(tk, today)
		}

		postCloseDone[tk] = today

	} else if !isTradingDay && !alreadyDoneToday {
		// Weekend: run analysis once per day if none cached today
		_, raw, _ := db.GetAnalysisCache(tk, today)
		if raw == nil {
			slog.Info("[bg-refresh] weekend analysis (Deribit-primary)", "ticker", tk)
			if _, err := analysis.RunAnalysis(tk, true); err != nil {
				slog.Error("[bg-refresh] weekend analysis error", "ticker", tk, "err", err)
			} else {
				slog.Info("[bg-refresh] weekend analysis complete", "ticker", tk)
				savePredictionsFromCache(tk, today)
			}
			postCloseDone[tk] = today
		}
	}
}

func checkLargeMove(tk string, cfg *config.TickerCfg) {
	_, raw, _ := db.GetAnalysisCache(tk, "")
	if raw == nil {
		return
	}

	var cached map[string]interface{}
	if json.Unmarshal(raw, &cached) != nil {
		return
	}

	oldPrice, ok := cached["_btc_price"].(float64)
	if !ok || oldPrice == 0 {
		return
	}

	candles := db.GetRecentCandles(cfg.BinanceSymbol, "1h", 1)
	if len(candles) == 0 {
		return
	}
	currentPrice := candles[0].Close

	pctMove := math.Abs(currentPrice-oldPrice) / oldPrice * 100
	if pctMove > 2 {
		slog.Info("[bg-refresh] large move detected", "ticker", tk, "pct", pctMove,
			"old", oldPrice, "new", currentPrice)
		if _, err := analysis.RunAnalysis(tk, true); err != nil {
			slog.Error("[bg-refresh] large-move analysis error", "ticker", tk, "err", err)
		} else {
			slog.Info("[bg-refresh] large-move analysis complete", "ticker", tk)
		}
	}
}

func deribitRefresh(ctx context.Context, tk string, cfg *config.TickerCfg, allFresh bool) {
	ageMin, ok := fetch.DeribitFreshness(cfg.DeribitCurrency)
	if !ok || ageMin >= 16 {
		// Reset Deribit cache so next fetch gets fresh data
		fetch.ResetDeribitCache(cfg.DeribitCurrency)

		if allFresh {
			slog.Info("[bg-refresh] Deribit stale, full refresh", "ticker", tk)
			for _, w := range config.DTEWindows {
				if ctx.Err() != nil {
					return
				}
				if _, err := gex.FetchWithCache(tk, w.MaxDTE, w.MinDTE, true); err != nil {
					slog.Error("[bg-refresh] Deribit refresh error", "ticker", tk, "dte_min", w.MinDTE, "dte_max", w.MaxDTE, "err", err)
				}
			}
			slog.Info("[bg-refresh] Deribit full refresh complete", "ticker", tk)
		} else {
			slog.Info("[bg-refresh] Deribit stale, weekend overlay", "ticker", tk)
			refreshDeribitOnly(tk, cfg)
			slog.Info("[bg-refresh] Deribit weekend overlay complete", "ticker", tk)
		}
	}
}

func refreshDeribitOnly(tk string, cfg *config.TickerCfg) {
	rfr, err := fetch.FetchRiskFreeRate()
	if err != nil {
		rfr = 0.05 // fallback
	}

	for _, w := range config.DTEWindows {
		func() {
			defer func() {
				if r := recover(); r != nil {
					slog.Error("[bg-refresh] Deribit overlay panic", "ticker", tk, "dte_min", w.MinDTE, "dte_max", w.MaxDTE, "err", r)
				}
			}()

			_, raw, err := db.GetLatestCache(tk, w.MaxDTE, "")
			if err != nil || raw == nil {
				return
			}

			var blob model.DataBlob
			if json.Unmarshal(raw, &blob) != nil {
				return
			}
			if blob.BTCSpot == 0 {
				return
			}

			deribitOpts, err := fetch.FetchDeribitOptions(w.MinDTE, w.MaxDTE, cfg.DeribitCurrency)
			if err != nil || len(deribitOpts) == 0 {
				return
			}

			// Process Deribit options: build strike map
			type strikeAgg struct {
				callGEX, putGEX float64
				callOI, putOI   float64
			}
			deribitStrikes := map[int]*strikeAgg{}
			var deribitOIBTC float64
			var deribitNetGEX float64

			btcSpot := blob.BTCSpot
			btcLo, btcHi := btcSpot*0.82, btcSpot*1.22

			for _, opt := range deribitOpts {
				T := opt.DTE / 365.0
				if T < 0.5/365 {
					T = 0.5 / 365
				}
				iv := opt.IV / 100.0
				sign := 1.0
				if opt.OptionType == "put" {
					sign = -1.0
				}

				gamma := bs.Gamma(opt.UnderlyingPrice, opt.Strike, T, rfr, iv)
				gexVal := sign * gamma * opt.OI * opt.UnderlyingPrice * opt.UnderlyingPrice * 0.01
				deribitOIBTC += opt.OI

				rounded := int(opt.Strike)
				agg, exists := deribitStrikes[rounded]
				if !exists {
					agg = &strikeAgg{}
					deribitStrikes[rounded] = agg
				}
				if opt.OptionType == "call" {
					agg.callGEX += gexVal
					agg.callOI += opt.OI
				} else {
					agg.putGEX += gexVal
					agg.putOI += opt.OI
				}
			}

			// Build deribit lookup and compute net GEX
			deribitByBTC := map[int]float64{}
			for strike, agg := range deribitStrikes {
				net := agg.callGEX + agg.putGEX
				deribitNetGEX += net
				if float64(strike) >= btcLo && float64(strike) <= btcHi {
					deribitByBTC[strike] = net
				}
			}

			// Patch gex_chart
			seen := map[int]bool{}
			for i := range blob.GEXChart {
				btcP := int(blob.GEXChart[i].BTC)
				seen[btcP] = true
				newDGex := deribitByBTC[btcP]
				blob.GEXChart[i].DeribitGEX = newDGex
				blob.GEXChart[i].NetGEX = blob.GEXChart[i].IBITGEX + newDGex
			}
			for btcP, netGEX := range deribitByBTC {
				if !seen[btcP] {
					blob.GEXChart = append(blob.GEXChart, model.StrikeData{
						BTC:        float64(btcP),
						NetGEX:     netGEX,
						IBITGEX:    0,
						DeribitGEX: netGEX,
						ActiveGEX:  netGEX,
					})
				}
			}

			// Recompute Deribit-only and combined levels
			deribitRows := make([]model.StrikeData, 0, len(deribitStrikes))
			for strike, agg := range deribitStrikes {
				deribitRows = append(deribitRows, model.StrikeData{
					Strike:    float64(strike),
					BTC:       float64(strike),
					NetGEX:    agg.callGEX + agg.putGEX,
					CallOI:    int64(agg.callOI),
					PutOI:     int64(agg.putOI),
					TotalOI:   int64(agg.callOI + agg.putOI),
					ActiveGEX: agg.callGEX + agg.putGEX,
				})
			}

			if len(deribitRows) > 0 {
				deribitLevels := gex.ComputeLevels(deribitRows, btcSpot, nil)
				if deribitLevels != nil {
					blob.DeribitLevelsBTC = deribitLevels
				}
			}

			// Combined levels from patched chart
			var combinedRows []model.StrikeData
			for _, entry := range blob.GEXChart {
				if entry.NetGEX != 0 {
					combinedRows = append(combinedRows, model.StrikeData{
						Strike:    entry.BTC,
						BTC:       entry.BTC,
						NetGEX:    entry.NetGEX,
						CallOI:    entry.CallOI,
						PutOI:     entry.PutOI,
						TotalOI:   entry.TotalOI,
						ActiveGEX: entry.ActiveGEX,
					})
				}
			}
			if len(combinedRows) > 0 {
				combinedLevels := gex.ComputeLevels(combinedRows, btcSpot, nil)
				if combinedLevels != nil {
					blob.CombinedLevelsBTC = combinedLevels
				}
			}

			// Update Deribit fields
			blob.DeribitAvail = true
			blob.DeribitOIBTC = deribitOIBTC
			blob.DeribitNetGEX = deribitNetGEX
			blob.Timestamp = nowET().Format("2006-01-02 15:04")

			// Update freshness
			if blob.DataFreshness != nil {
				ageMin, _ := fetch.DeribitFreshness(cfg.DeribitCurrency)
				if blob.DataFreshness.Deribit == nil {
					blob.DataFreshness.Deribit = &model.SourceFreshness{}
				}
				blob.DataFreshness.Deribit.AgeMinutes = ageMin
				blob.DataFreshness.Deribit.LastUpdate = nowET().Format("2006-01-02 15:04")
			}

			// Remove old Deribit IV entries, keep IBIT
			var freshIV []model.IVTermEntry
			for _, entry := range blob.IVTermStructure {
				if entry.Source != "deribit" {
					freshIV = append(freshIV, entry)
				}
			}
			deribitIVEntries := computeDeribitIV(deribitOpts, btcSpot)
			freshIV = append(freshIV, deribitIVEntries...)
			blob.IVTermStructure = freshIV

			if err := db.SetCachedData(tk, w.MaxDTE, blob); err != nil {
				slog.Error("[bg-refresh] Deribit overlay save error", "ticker", tk, "err", err)
			} else {
				slog.Info("[bg-refresh] Deribit overlay done", "ticker", tk, "dte_min", w.MinDTE, "dte_max", w.MaxDTE)
			}
		}()
	}
}

// computeDeribitIV calculates ATM IV per expiry from Deribit options.
func computeDeribitIV(opts []model.DeribitOption, btcSpot float64) []model.IVTermEntry {
	type ivEntry struct {
		strike, iv float64
		oi         float64
		dte        float64
	}

	byExpiry := map[string][]ivEntry{}
	for _, opt := range opts {
		expStr := time.Unix(opt.ExpiryDate, 0).UTC().Format("2006-01-02")
		byExpiry[expStr] = append(byExpiry[expStr], ivEntry{
			strike: opt.Strike,
			iv:     opt.IV / 100.0,
			oi:     opt.OI,
			dte:    opt.DTE,
		})
	}

	var result []model.IVTermEntry
	for expStr, entries := range byExpiry {
		if len(entries) == 0 {
			continue
		}

		// Find ATM entries (within 2% of spot)
		var atm []ivEntry
		for _, e := range entries {
			if math.Abs(e.strike-btcSpot)/btcSpot < 0.02 {
				atm = append(atm, e)
			}
		}
		if len(atm) == 0 {
			// Fallback: closest 2 strikes
			if len(entries) >= 2 {
				best := entries[0]
				secondBest := entries[1]
				for _, e := range entries[2:] {
					d := math.Abs(e.strike - btcSpot)
					if d < math.Abs(best.strike-btcSpot) {
						secondBest = best
						best = e
					} else if d < math.Abs(secondBest.strike-btcSpot) {
						secondBest = e
					}
				}
				atm = []ivEntry{best, secondBest}
			} else {
				atm = entries[:1]
			}
		}

		var totalOI float64
		for _, e := range atm {
			totalOI += e.oi
		}
		if totalOI <= 0 {
			continue
		}

		var atmIV float64
		for _, e := range atm {
			atmIV += e.iv * e.oi
		}
		atmIV /= totalOI

		result = append(result, model.IVTermEntry{
			Expiry: expStr,
			DTE:    int(math.Round(entries[0].dte)),
			ATMIV:  atmIV,
			Source: "deribit",
		})
	}
	return result
}

func savePredictionsFromCache(tk, today string) {
	// Check if already saved today
	preds, err := db.GetRecentPredictions(tk, 1)
	if err == nil && len(preds) > 0 && preds[0].AnalysisDate == today {
		return // already saved
	}

	// Gather cached data for all DTE windows
	results := map[int]*model.DataBlob{}
	for _, w := range config.DTEWindows {
		_, raw, err := db.GetLatestCache(tk, w.MaxDTE, "")
		if err != nil || raw == nil {
			continue
		}
		var blob model.DataBlob
		if json.Unmarshal(raw, &blob) == nil {
			results[w.Label] = &blob
		}
	}

	// Get analysis for AI bottom line extraction
	_, analysisRaw, _ := db.GetAnalysisCache(tk, today)
	var analysisMap map[string]interface{}
	if analysisRaw != nil {
		json.Unmarshal(analysisRaw, &analysisMap)
	}

	SavePredictions(tk, results, analysisMap)
}
