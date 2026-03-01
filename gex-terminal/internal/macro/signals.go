package macro

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"gex-terminal/internal/db"
)

// signalResult is returned by each signal function.
type signalResult struct {
	Score   int
	Detail  string
	History []map[string]interface{}
}

// computeFundingSignal computes Signal 6: Aggregate Funding Rate (-13 to +13).
func computeFundingSignal() signalResult {
	rows, err := db.GetCoinglassHistory("BTC", "avg_funding_rate", 30)
	if err != nil || len(rows) < 7 {
		return signalResult{0, "Insufficient funding data", nil}
	}

	// Build history oldest-first for charts
	history := make([]map[string]interface{}, len(rows))
	for i, r := range rows {
		history[len(rows)-1-i] = map[string]interface{}{"date": r.Date, "rate": r.Value}
	}

	// 7d moving average for each history entry
	for i := range history {
		start := i - 6
		if start < 0 {
			start = 0
		}
		var sum float64
		count := 0
		for j := start; j <= i; j++ {
			sum += history[j]["rate"].(float64)
			count++
		}
		history[i]["avg_7d"] = sum / float64(count)
	}

	// Compute averages from newest-first rows
	var sum7d, sum30d float64
	for i := 0; i < 7; i++ {
		sum7d += rows[i].Value
	}
	for i := 0; i < len(rows); i++ {
		sum30d += rows[i].Value
	}
	avg7d := sum7d / 7.0
	avg30d := sum30d / float64(len(rows))

	score := 0
	detail := fmt.Sprintf("7d avg: %.4f%%, 30d avg: %.4f%%", avg7d*100, avg30d*100)

	if avg7d < -0.0001 { // < -0.01%
		if avg7d > avg30d {
			score = 13
			detail += " | Deeply negative, reverting up"
		} else {
			score = 8
			detail += " | Deeply negative, still falling"
		}
	} else if avg7d > 0.0003 { // > 0.03%
		if avg7d < avg30d {
			score = -13
			detail += " | Deeply positive, reverting down"
		} else {
			score = -8
			detail += " | Deeply positive, still rising"
		}
	}

	return signalResult{score, detail, history}
}

// computeOISignal computes Signal 7: Aggregate Futures OI (-13 to +13).
func computeOISignal() signalResult {
	rows, err := db.GetCoinglassHistory("BTC", "total_oi_usd", 90)
	if err != nil || len(rows) < 7 {
		return signalResult{0, "Insufficient OI data", nil}
	}

	// History oldest-first
	history := make([]map[string]interface{}, len(rows))
	for i, r := range rows {
		history[len(rows)-1-i] = map[string]interface{}{"date": r.Date, "oi_usd": r.Value}
	}

	oiCurrent := rows[0].Value
	oi90dPeak := rows[0].Value
	for _, r := range rows {
		if r.Value > oi90dPeak {
			oi90dPeak = r.Value
		}
	}
	changeFromPeak := 0.0
	if oi90dPeak > 0 {
		changeFromPeak = (oiCurrent - oi90dPeak) / oi90dPeak * 100
	}

	for _, h := range history {
		oiVal := h["oi_usd"].(float64)
		if oi90dPeak > 0 {
			h["peak_pct"] = (oiVal - oi90dPeak) / oi90dPeak * 100
		}
	}

	// 7d slope
	oi7dSlope := "unknown"
	if len(rows) >= 7 {
		oi7dAgo := rows[6].Value
		if oiCurrent > oi7dAgo*1.02 {
			oi7dSlope = "rising"
		} else if oiCurrent < oi7dAgo*0.98 {
			oi7dSlope = "falling"
		} else {
			oi7dSlope = "flat"
		}
	}

	// Cross-reference funding
	fundingRows, _ := db.GetCoinglassHistory("BTC", "avg_funding_rate", 7)
	funding7dAvg := 0.0
	if len(fundingRows) > 0 {
		var sum float64
		for _, r := range fundingRows {
			sum += r.Value
		}
		funding7dAvg = sum / float64(len(fundingRows))
	}

	score := 0
	detail := fmt.Sprintf("OI $%.1fB, %+.1f%% from 90d peak, 7d %s", oiCurrent/1e9, changeFromPeak, oi7dSlope)

	if changeFromPeak < -20 {
		if oi7dSlope == "flat" || oi7dSlope == "rising" {
			score = 13
			detail += " | Flushed + stabilizing"
		} else {
			score = 6
			detail += " | Flush in progress"
		}
	} else if changeFromPeak > -5 { // Within 5% of peak
		if funding7dAvg > 0 {
			score = -13
			detail += fmt.Sprintf(" | Near peak + positive funding (%.4f%%)", funding7dAvg*100)
		} else {
			score = -6
			detail += fmt.Sprintf(" | Near peak, shorts crowded (%.4f%%)", funding7dAvg*100)
		}
	}

	return signalResult{score, detail, history}
}

// computeLiquidationSignal computes Signal 8: Liquidation Intensity (-13 to +13).
func computeLiquidationSignal() signalResult {
	// Historical daily totals
	histRows, _ := db.GetCoinglassHistory("BTC", "total_liquidation_usd", 90)
	history := make([]map[string]interface{}, 0, len(histRows))
	for i := len(histRows) - 1; i >= 0; i-- {
		history = append(history, map[string]interface{}{
			"date": histRows[i].Date, "liq_usd": histRows[i].Value,
		})
	}

	// Today's long/short snapshot
	_, snapExtra, err := db.GetCoinglassLatest("BTC", "liquidation_snapshot")
	if err != nil || snapExtra == nil {
		return signalResult{0, "No liquidation data", history}
	}

	long24h, _ := snapExtra["long_24h"].(float64)
	short24h, _ := snapExtra["short_24h"].(float64)
	total24h := long24h + short24h

	if total24h == 0 {
		return signalResult{0, "No liquidation data", history}
	}

	// 30d average
	if len(histRows) < 7 {
		return signalResult{0, "Insufficient liquidation history", history}
	}
	count := min(30, len(histRows))
	var sum30d float64
	for i := 0; i < count; i++ {
		sum30d += histRows[i].Value
	}
	avg30d := sum30d / float64(count)

	intensity := 0.0
	if avg30d > 0 {
		intensity = total24h / avg30d
	}

	longRatio := 0.5
	if total24h > 0 {
		longRatio = long24h / total24h
	}
	dominantSide := "balanced"
	if longRatio > 0.65 {
		dominantSide = "long"
	} else if longRatio < 0.35 {
		dominantSide = "short"
	}

	// Cross-reference BTC 7d price slope
	candles := db.GetRecentCandles("BTCUSDT", "1d", 7)
	priceSlope := "unknown"
	if len(candles) >= 2 {
		latest := candles[0].Close
		oldest := candles[len(candles)-1].Close
		if oldest > 0 {
			pctChg := (latest - oldest) / oldest * 100
			if pctChg > 2 {
				priceSlope = "rising"
			} else if pctChg < -2 {
				priceSlope = "falling"
			} else {
				priceSlope = "flat"
			}
		}
	}

	score := 0
	detail := fmt.Sprintf("24h: $%.0fM (L:$%.0fM / S:$%.0fM), %.1fx avg, %s, price 7d %s",
		total24h/1e6, long24h/1e6, short24h/1e6, intensity, dominantSide, priceSlope)

	if intensity >= 2.0 && dominantSide != "balanced" {
		magnitude := 6
		if intensity >= 3.0 {
			magnitude = 13
		}

		if dominantSide == "long" {
			if priceSlope == "flat" || priceSlope == "rising" {
				score = magnitude
				detail += " | Long capitulation + price stabilizing"
			} else {
				score = max(6, magnitude-4)
				detail += " | Long flush in progress"
			}
		} else if dominantSide == "short" {
			if priceSlope == "flat" || priceSlope == "falling" {
				score = -magnitude
				detail += " | Short squeeze exhaustion"
			} else {
				score = -max(6, magnitude-4)
				detail += " | Short squeeze in progress"
			}
		}
	}

	return signalResult{score, detail, history}
}

// computePCRDirection computes Signal 9: Put/Call Ratio Direction (-8 to +8).
func computePCRDirection(ticker string, btcPerShare float64) signalResult {
	days := 14
	pcrDTE := 3

	var cacheRows []db.CacheRow
	for _, tryDTE := range []int{3, 7, 14} {
		cacheRows = db.GetCacheRowsForDTE(ticker, tryDTE, days)
		if len(cacheRows) >= 5 {
			pcrDTE = tryDTE
			break
		}
	}

	if len(cacheRows) < 5 {
		return signalResult{0, "Insufficient PCR data", nil}
	}

	// Extract PCR from cache JSON, newest first from DB
	type pcrEntry struct {
		Date string
		PCR  float64
	}
	var pcrHistory []pcrEntry
	for _, row := range cacheRows {
		var d struct {
			Levels struct {
				PCR *float64 `json:"pcr"`
			} `json:"levels"`
		}
		if err := json.Unmarshal(row.DataJSON, &d); err != nil {
			continue
		}
		if d.Levels.PCR != nil {
			pcrHistory = append(pcrHistory, pcrEntry{row.Date, *d.Levels.PCR})
		}
	}

	// Reverse to oldest first
	for i, j := 0, len(pcrHistory)-1; i < j; i, j = i+1, j-1 {
		pcrHistory[i], pcrHistory[j] = pcrHistory[j], pcrHistory[i]
	}

	if len(pcrHistory) < 5 {
		return signalResult{0, "Insufficient PCR data", nil}
	}

	// Build history for charts (oldest first)
	historyMap := make([]map[string]interface{}, len(pcrHistory))
	for i, p := range pcrHistory {
		historyMap[i] = map[string]interface{}{"date": p.Date, "pcr": p.PCR}
	}

	recent3 := pcrHistory[len(pcrHistory)-3:]
	prior := pcrHistory[:len(pcrHistory)-3]
	var sumRecent, sumPrior float64
	for _, p := range recent3 {
		sumRecent += p.PCR
	}
	for _, p := range prior {
		sumPrior += p.PCR
	}
	avgRecent := sumRecent / float64(len(recent3))
	avgPrior := sumPrior / float64(len(prior))
	currentPCR := pcrHistory[len(pcrHistory)-1].PCR

	dteLabels := map[int]string{3: "0-3d", 7: "4-7d", 14: "8-14d"}
	dteLabel := dteLabels[pcrDTE]
	if dteLabel == "" {
		dteLabel = fmt.Sprintf("%dd", pcrDTE)
	}

	detail := fmt.Sprintf("PCR %.2f, 3d avg %.2f vs prior %.2f [%s]", currentPCR, avgRecent, avgPrior, dteLabel)

	pctChange := 0.0
	if avgPrior != 0 {
		pctChange = (avgRecent - avgPrior) / avgPrior
	}

	score := 0
	if pctChange > 0.15 {
		score = 8
		detail += " | PCR surging (fear, contrarian bullish)"
	} else if pctChange > 0.08 {
		score = 4
		detail += " | PCR rising (put buying increasing)"
	} else if pctChange < -0.15 {
		score = -8
		detail += " | PCR plunging (greed, contrarian bearish)"
	} else if pctChange < -0.08 {
		score = -4
		detail += " | PCR falling (call buying increasing)"
	}

	return signalResult{score, detail, historyMap}
}

// computeWallBreach computes Signal 10: Wall Breach Detection (-13 to +13).
func computeWallBreach(ticker string, btcPerShare float64) signalResult {
	var spot, cw, pw float64
	var regime string
	var tryDTE int

	for _, dte := range []int{3, 7, 14, 30} {
		rows := db.GetCacheRowsForDTE(ticker, dte, 1)
		if len(rows) == 0 {
			continue
		}
		var d struct {
			BTCSpot     float64                `json:"btc_spot"`
			CombinedBTC map[string]interface{} `json:"combined_levels_btc"`
			Levels      struct {
				CallWall float64 `json:"call_wall"`
				PutWall  float64 `json:"put_wall"`
				Regime   string  `json:"regime"`
			} `json:"levels"`
		}
		if err := json.Unmarshal(rows[0].DataJSON, &d); err != nil {
			continue
		}
		spot = d.BTCSpot
		if d.CombinedBTC != nil {
			if v, ok := d.CombinedBTC["call_wall"].(float64); ok {
				cw = v
			}
			if v, ok := d.CombinedBTC["put_wall"].(float64); ok {
				pw = v
			}
		}
		if cw == 0 && btcPerShare > 0 {
			cw = d.Levels.CallWall / btcPerShare
		}
		if pw == 0 && btcPerShare > 0 {
			pw = d.Levels.PutWall / btcPerShare
		}
		regime = d.Levels.Regime
		tryDTE = dte
		if spot > 0 && cw > 0 && pw > 0 {
			break
		}
	}

	if spot == 0 || cw == 0 || pw == 0 {
		return signalResult{0, "No wall data for breach check", nil}
	}

	dteLabels := map[int]string{3: "0-3d", 7: "4-7d", 14: "8-14d", 30: "15-30d"}
	dteLabel := dteLabels[tryDTE]
	if dteLabel == "" {
		dteLabel = fmt.Sprintf("%dd", tryDTE)
	}
	detail := fmt.Sprintf("Spot $%.0f, CW $%.0f, PW $%.0f [%s]", spot, cw, pw, dteLabel)

	cwDistPct := 0.0
	if cw > 0 {
		cwDistPct = (spot - cw) / cw
	}
	pwDistPct := 0.0
	if pw > 0 {
		pwDistPct = (pw - spot) / pw
	}

	score := 0
	if cwDistPct > 0.005 {
		if regime == "negative_gamma" {
			score = 13
			detail += fmt.Sprintf(" | ABOVE call wall (%.1f%%) + neg \u03b3 \u2192 breakout", cwDistPct*100)
		} else {
			score = 6
			detail += fmt.Sprintf(" | ABOVE call wall (%.1f%%) + pos \u03b3 \u2192 likely pin", cwDistPct*100)
		}
	} else if cwDistPct > -0.005 {
		detail += fmt.Sprintf(" | Testing call wall (%.1f%% away)", math.Abs(cwDistPct)*100)
	} else if pwDistPct > 0.005 {
		if regime == "negative_gamma" {
			score = -13
			detail += fmt.Sprintf(" | BELOW put wall (%.1f%%) + neg \u03b3 \u2192 breakdown", pwDistPct*100)
		} else {
			score = -6
			detail += fmt.Sprintf(" | BELOW put wall (%.1f%%) + pos \u03b3 \u2192 dealer support", pwDistPct*100)
		}
	} else if pwDistPct > -0.005 {
		detail += fmt.Sprintf(" | Testing put wall (%.1f%% away)", math.Abs(pwDistPct)*100)
	} else {
		detail += " | Between walls, no breach"
	}

	return signalResult{score, detail, nil}
}

// computeIVTermSignal computes Signal 11: IV Term Structure Shape (-8 to +8).
func computeIVTermSignal(ticker string, btcPerShare float64) signalResult {
	days := 14
	var ivTS []map[string]interface{}
	usedDTE := 3

	for _, tryDTE := range []int{3, 7, 14, 30, 45} {
		rows := db.GetCacheRowsForDTE(ticker, tryDTE, 1)
		if len(rows) == 0 {
			continue
		}
		var d struct {
			IVTermStructure []map[string]interface{} `json:"iv_term_structure"`
		}
		if err := json.Unmarshal(rows[0].DataJSON, &d); err != nil {
			continue
		}
		if len(d.IVTermStructure) >= 2 {
			ivTS = d.IVTermStructure
			usedDTE = tryDTE
			break
		}
	}

	if len(ivTS) < 2 {
		return signalResult{0, "No IV term structure data", nil}
	}

	// Group by source, prefer deribit
	bySource := map[string][]map[string]interface{}{}
	for _, entry := range ivTS {
		src, _ := entry["source"].(string)
		if src == "" {
			src = "ibit"
		}
		bySource[src] = append(bySource[src], entry)
	}

	source := "ibit"
	if len(bySource["deribit"]) >= 2 {
		source = "deribit"
	}
	entries := bySource[source]
	sort.Slice(entries, func(i, j int) bool {
		di, _ := entries[i]["dte"].(float64)
		dj, _ := entries[j]["dte"].(float64)
		return di < dj
	})

	if len(entries) < 2 {
		return signalResult{0, "Need at least 2 expiries for term structure", nil}
	}

	// Build history for charts
	var historyMap []map[string]interface{}
	histRows := db.GetCacheRowsForDTE(ticker, usedDTE, days)
	for i := len(histRows) - 1; i >= 0; i-- {
		hr := histRows[i]
		var hd struct {
			IVTermStructure []map[string]interface{} `json:"iv_term_structure"`
		}
		if err := json.Unmarshal(hr.DataJSON, &hd); err != nil {
			continue
		}
		var hEntries []map[string]interface{}
		for _, e := range hd.IVTermStructure {
			if src, _ := e["source"].(string); src == source {
				hEntries = append(hEntries, e)
			}
		}
		sort.Slice(hEntries, func(a, b int) bool {
			da, _ := hEntries[a]["dte"].(float64)
			db, _ := hEntries[b]["dte"].(float64)
			return da < db
		})
		if len(hEntries) >= 2 {
			shortIV, _ := hEntries[0]["atm_iv"].(float64)
			longIV, _ := hEntries[len(hEntries)-1]["atm_iv"].(float64)
			if shortIV > 0 {
				slope := (longIV - shortIV) / shortIV
				shortDTE, _ := hEntries[0]["dte"].(float64)
				longDTE, _ := hEntries[len(hEntries)-1]["dte"].(float64)
				historyMap = append(historyMap, map[string]interface{}{
					"date":      hr.Date,
					"short_iv":  shortIV,
					"long_iv":   longIV,
					"short_dte": shortDTE,
					"long_dte":  longDTE,
					"slope":     math.Round(slope*10000) / 10000,
				})
			}
		}
	}

	shortEntry := entries[0]
	longEntry := entries[len(entries)-1]
	shortIV, _ := shortEntry["atm_iv"].(float64)
	longIV, _ := longEntry["atm_iv"].(float64)

	if shortIV <= 0 {
		return signalResult{0, "Invalid short-dated IV", historyMap}
	}

	slope := (longIV - shortIV) / shortIV
	shortDTE, _ := shortEntry["dte"].(float64)
	longDTE, _ := longEntry["dte"].(float64)

	detail := fmt.Sprintf("Short IV %.1f%% (%.0fd) vs Long IV %.1f%% (%.0fd), slope %+.1f%% [%s]",
		shortIV*100, shortDTE, longIV*100, longDTE, slope*100, source)

	score := 0
	if slope < -0.10 {
		score = 8
		detail += " | Strong backwardation (fear, contrarian bullish)"
	} else if slope < -0.03 {
		score = 4
		detail += " | Mild backwardation (hedging demand)"
	} else if slope > 0.25 {
		score = -8
		detail += " | Steep contango (complacency, contrarian bearish)"
	} else if slope > 0.15 {
		score = -4
		detail += " | Elevated contango (low near-term fear)"
	}

	return signalResult{score, detail, historyMap}
}
