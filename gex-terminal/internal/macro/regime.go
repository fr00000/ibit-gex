package macro

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"time"

	"gex-terminal/internal/config"
	"gex-terminal/internal/db"
	"gex-terminal/internal/model"
)

func nowET() time.Time {
	loc, _ := time.LoadLocation("America/New_York")
	return time.Now().In(loc)
}

// ComputeMacroRegime computes the 11-signal macro scoring system.
// Returns a MacroRegime with score from -100 (top) to +100 (bottom).
func ComputeMacroRegime(ticker string, days int) *model.MacroRegime {
	if days <= 0 {
		days = 30
	}
	cfg := config.TickerConfig[ticker]
	if cfg == nil {
		cfg = config.TickerConfig["IBIT"]
	}
	btcPerShare := cfg.PerShareDefault

	// ── Signal 1: Regime Persistence & Transition (-12 to +12) ──
	regimeScore := 0
	regimeDetail := ""
	var regimeHistory []map[string]interface{}

	snapshots, err := db.GetHistory(ticker, days)
	if err == nil && len(snapshots) > 0 {
		regimeHistory = make([]map[string]interface{}, len(snapshots))
		for i, s := range snapshots {
			regimeHistory[i] = map[string]interface{}{
				"date": s.Date, "regime": s.Regime, "net_gex": s.NetGEX,
			}
		}
		total := len(snapshots)
		negCount := 0
		for _, s := range snapshots {
			if s.Regime == "negative_gamma" {
				negCount++
			}
		}
		posCount := total - negCount

		flips := 0
		for i := 1; i < len(snapshots); i++ {
			if snapshots[i].Regime != snapshots[i-1].Regime {
				flips++
			}
		}

		negPct := float64(negCount) / float64(total)
		posPct := float64(posCount) / float64(total)

		if negPct > 0.70 {
			regimeScore = 8
			regimeDetail = fmt.Sprintf("%d/%d days negative gamma (%.0f%%)", negCount, total, negPct*100)
		} else if posPct > 0.70 {
			regimeScore = -8
			regimeDetail = fmt.Sprintf("%d/%d days positive gamma (%.0f%%)", posCount, total, posPct*100)
		} else {
			regimeDetail = fmt.Sprintf("Oscillating: %d pos / %d neg, %d flips", posCount, negCount, flips)
		}

		// Transition bonus
		if len(snapshots) >= 4 {
			recentRegime := snapshots[0].Regime
			recentStreak := 1
			for i := 1; i < min(4, len(snapshots)); i++ {
				if snapshots[i].Regime == recentRegime {
					recentStreak++
				} else {
					break
				}
			}

			if recentStreak <= 3 {
				oldRegimeCount := 0
				for i := recentStreak; i < len(snapshots); i++ {
					if snapshots[i].Regime != recentRegime {
						oldRegimeCount++
					} else {
						break
					}
				}
				if oldRegimeCount >= 7 {
					if recentRegime == "positive_gamma" {
						regimeScore = 12
						regimeDetail += fmt.Sprintf(" | TRANSITION: %dd neg->pos (streak %dd)", oldRegimeCount, recentStreak)
					} else {
						regimeScore = -12
						regimeDetail += fmt.Sprintf(" | TRANSITION: %dd pos->neg (streak %dd)", oldRegimeCount, recentStreak)
					}
				}
			}
		}
	}

	// ── Signal 2: Structural Wall Migration (-12 to +12) ──
	wallMigScore := 0
	wallDetail := ""
	var wallHistory []map[string]interface{}

	cacheRows := db.GetCacheRowsForDTE(ticker, 45, days)
	if len(cacheRows) >= 7 {
		for i := len(cacheRows) - 1; i >= 0; i-- {
			row := cacheRows[i]
			var d struct {
				CombinedBTC map[string]interface{} `json:"combined_levels_btc"`
				Levels      struct {
					CallWall float64 `json:"call_wall"`
					PutWall  float64 `json:"put_wall"`
				} `json:"levels"`
			}
			if err := json.Unmarshal(row.DataJSON, &d); err != nil {
				continue
			}
			cw := 0.0
			pw := 0.0
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
			if cw > 0 && pw > 0 {
				wallHistory = append(wallHistory, map[string]interface{}{
					"date": row.Date, "call_wall": cw, "put_wall": pw,
				})
			}
		}

		if len(wallHistory) >= 7 {
			half := len(wallHistory) / 2
			firstHalf := wallHistory[:half]
			lastHalf := wallHistory[half:]

			avgPWFirst := avgField(firstHalf, "put_wall")
			avgPWLast := avgField(lastHalf, "put_wall")
			avgCWFirst := avgField(firstHalf, "call_wall")
			avgCWLast := avgField(lastHalf, "call_wall")

			pwChange := 0.0
			if avgPWFirst > 0 {
				pwChange = (avgPWLast - avgPWFirst) / avgPWFirst
			}
			cwChange := 0.0
			if avgCWFirst > 0 {
				cwChange = (avgCWLast - avgCWFirst) / avgCWFirst
			}

			pwPart := 0
			if pwChange > 0.02 {
				pwPart = 6
			} else if pwChange < -0.02 {
				pwPart = -6
			}
			cwPart := 0
			if cwChange > 0.02 {
				cwPart = 6
			} else if cwChange < -0.02 {
				cwPart = -6
			}

			pwDir := "stable"
			if pwChange > 0.02 {
				pwDir = "rising"
			} else if pwChange < -0.02 {
				pwDir = "falling"
			}
			cwDir := "stable"
			if cwChange > 0.02 {
				cwDir = "rising"
			} else if cwChange < -0.02 {
				cwDir = "falling"
			}

			wallMigScore = pwPart + cwPart
			wallDetail = fmt.Sprintf("Put wall %s (%+.1f%%), Call wall %s (%+.1f%%)", pwDir, pwChange*100, cwDir, cwChange*100)
		}
	}
	if wallDetail == "" {
		wallDetail = fmt.Sprintf("Insufficient data (%d days, need 7+ of 31-45d cache)", len(cacheRows))
	}

	// ── Signal 3: Range Compression + Spot Position (-12 to +12) ──
	compScore := 0
	compDetail := ""

	// Get spot BTC and current walls
	var spotBTC, currentCW, currentPW float64
	for _, tryDTE := range []int{3, 7, 14, 30, 45} {
		rows := db.GetCacheRowsForDTE(ticker, tryDTE, 1)
		if len(rows) == 0 {
			continue
		}
		var d struct {
			BTCSpot     float64                `json:"btc_spot"`
			CombinedBTC map[string]interface{} `json:"combined_levels_btc"`
			Levels      struct {
				CallWall float64 `json:"call_wall"`
				PutWall  float64 `json:"put_wall"`
			} `json:"levels"`
		}
		if err := json.Unmarshal(rows[0].DataJSON, &d); err != nil {
			continue
		}
		spotBTC = d.BTCSpot
		if d.CombinedBTC != nil {
			if v, ok := d.CombinedBTC["call_wall"].(float64); ok {
				currentCW = v
			}
			if v, ok := d.CombinedBTC["put_wall"].(float64); ok {
				currentPW = v
			}
		}
		if currentCW == 0 && btcPerShare > 0 {
			currentCW = d.Levels.CallWall / btcPerShare
		}
		if currentPW == 0 && btcPerShare > 0 {
			currentPW = d.Levels.PutWall / btcPerShare
		}
		if spotBTC > 0 && currentCW > 0 && currentPW > 0 {
			break
		}
	}

	// Build range history from same DTE window
	var rangeRows []db.CacheRow
	for _, tryDTE := range []int{3, 7, 14, 30, 45} {
		rangeRows = db.GetCacheRowsForDTE(ticker, tryDTE, days)
		if len(rangeRows) >= 7 {
			break
		}
	}

	var ranges []float64
	for _, row := range rangeRows {
		var d struct {
			CombinedBTC map[string]interface{} `json:"combined_levels_btc"`
			Levels      struct {
				CallWall float64 `json:"call_wall"`
				PutWall  float64 `json:"put_wall"`
			} `json:"levels"`
		}
		if err := json.Unmarshal(row.DataJSON, &d); err != nil {
			continue
		}
		cw := 0.0
		pw := 0.0
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
		if cw > 0 && pw > 0 && cw > pw {
			ranges = append(ranges, cw-pw)
		}
	}

	if len(ranges) > 0 {
		currentRange := ranges[0] // newest first
		sorted := make([]float64, len(ranges))
		copy(sorted, ranges)
		sort.Float64s(sorted)
		below := 0
		for _, r := range sorted {
			if r < currentRange {
				below++
			}
		}
		percentile := float64(below) / float64(len(sorted)) * 100

		if percentile <= 20 && spotBTC > 0 && currentCW > 0 && currentPW > 0 && currentCW > currentPW {
			spotPct := (spotBTC - currentPW) / (currentCW - currentPW)
			spotPct = math.Max(0, math.Min(1, spotPct))
			currentRegime := ""
			if len(snapshots) > 0 {
				currentRegime = snapshots[0].Regime
			}

			if currentRegime == "positive_gamma" {
				if spotPct > 0.75 {
					compScore = -12
					compDetail = fmt.Sprintf("%.0fth pctl range, spot at %.0f%% (call wall) + pos \u03b3 \u2192 rejection", percentile, spotPct*100)
				} else if spotPct > 0.60 {
					compScore = -6
					compDetail = fmt.Sprintf("%.0fth pctl range, spot at %.0f%% (lean resistance) + pos \u03b3", percentile, spotPct*100)
				} else if spotPct < 0.25 {
					compScore = 12
					compDetail = fmt.Sprintf("%.0fth pctl range, spot at %.0f%% (put wall) + pos \u03b3 \u2192 bounce", percentile, spotPct*100)
				} else if spotPct < 0.40 {
					compScore = 6
					compDetail = fmt.Sprintf("%.0fth pctl range, spot at %.0f%% (lean support) + pos \u03b3", percentile, spotPct*100)
				} else {
					compDetail = fmt.Sprintf("%.0fth pctl range, spot at %.0f%% (mid-range) + pos \u03b3 \u2192 range-bound", percentile, spotPct*100)
				}
			} else if currentRegime == "negative_gamma" {
				if spotPct > 0.75 {
					compScore = 12
					compDetail = fmt.Sprintf("%.0fth pctl range, spot at %.0f%% (call wall) + neg \u03b3 \u2192 breakout", percentile, spotPct*100)
				} else if spotPct > 0.60 {
					compScore = 6
					compDetail = fmt.Sprintf("%.0fth pctl range, spot at %.0f%% (lean resistance) + neg \u03b3", percentile, spotPct*100)
				} else if spotPct < 0.25 {
					compScore = -12
					compDetail = fmt.Sprintf("%.0fth pctl range, spot at %.0f%% (put wall) + neg \u03b3 \u2192 breakdown", percentile, spotPct*100)
				} else if spotPct < 0.40 {
					compScore = -6
					compDetail = fmt.Sprintf("%.0fth pctl range, spot at %.0f%% (lean support) + neg \u03b3", percentile, spotPct*100)
				} else {
					compDetail = fmt.Sprintf("%.0fth pctl range, spot at %.0f%% (mid-range) + neg \u03b3 \u2192 volatile", percentile, spotPct*100)
				}
			} else {
				compDetail = fmt.Sprintf("%.0fth pctl range, spot at %.0f%%, oscillating regime", percentile, spotPct*100)
			}
		} else if percentile <= 20 {
			compDetail = fmt.Sprintf("%.0fth pctl range (compressed, no wall data for position)", percentile)
		} else {
			compDetail = fmt.Sprintf("%.0fth pctl range (not compressed)", percentile)
		}
	}
	if compDetail == "" {
		compDetail = "Insufficient data for range analysis"
	}

	// ── Signal 4: ETF Flow Momentum (-12 to +12) ──
	etfFlowScore := 0
	flowDetail := ""
	var flowHistory []map[string]interface{}

	flowRaws := db.GetFlowsRaw(ticker, days)
	if len(flowRaws) > 0 {
		// Build history oldest-first
		flowHistory = make([]map[string]interface{}, len(flowRaws))
		for i, r := range flowRaws {
			flowHistory[len(flowRaws)-1-i] = map[string]interface{}{
				"date":          r.Date,
				"ibit_flow":     r.DailyFlowDollars,
				"total_flow":    r.TotalBTCETFFlow,
				"cumulative_10d": nil,
			}
		}
		// Compute cumulative 10d
		for i := range flowHistory {
			start := i - 9
			if start < 0 {
				start = 0
			}
			var cum float64
			for j := start; j <= i; j++ {
				if tf := flowHistory[j]["total_flow"]; tf != nil {
					if v, ok := tf.(*float64); ok && v != nil {
						cum += *v
					}
				} else if ibf, ok := flowHistory[j]["ibit_flow"].(float64); ok {
					cum += ibf
				}
			}
			flowHistory[i]["cumulative_10d"] = cum
		}

		// Use total_btc_etf_flow, fall back to daily_flow_dollars (newest first from DB)
		flows := make([]float64, len(flowRaws))
		for i, r := range flowRaws {
			if r.TotalBTCETFFlow != nil {
				flows[i] = *r.TotalBTCETFFlow
			} else {
				flows[i] = r.DailyFlowDollars
			}
		}

		flow10d := 0.0
		for i := 0; i < min(10, len(flows)); i++ {
			flow10d += flows[i]
		}
		flow30d := 0.0
		for i := 0; i < min(30, len(flows)); i++ {
			flow30d += flows[i]
		}

		if len(flows) >= 15 {
			flow5dRecent := 0.0
			for i := 0; i < 5; i++ {
				flow5dRecent += flows[i]
			}
			flow5dPrior := 0.0
			for i := 5; i < min(10, len(flows)); i++ {
				flow5dPrior += flows[i]
			}

			if flow5dPrior < 0 && flow5dRecent > 0 {
				etfFlowScore = 12
				flowDetail = fmt.Sprintf("Flow reversal: prior $%.0fM -> recent $%.0fM", flow5dPrior/1e6, flow5dRecent/1e6)
			} else if flow5dPrior > 0 && flow5dRecent < 0 {
				etfFlowScore = -12
				flowDetail = fmt.Sprintf("Flow reversal: prior $%.0fM -> recent $%.0fM", flow5dPrior/1e6, flow5dRecent/1e6)
			} else if flow30d != 0 {
				pace10d := flow10d / float64(min(10, len(flows)))
				pace30d := flow30d / float64(min(30, len(flows)))

				if flow10d > 0 && pace10d > pace30d {
					etfFlowScore = 8
					flowDetail = fmt.Sprintf("Accelerating inflows: 10d $%.0fM", flow10d/1e6)
				} else if flow10d < 0 && math.Abs(pace10d) > math.Abs(pace30d) {
					etfFlowScore = -8
					flowDetail = fmt.Sprintf("Accelerating outflows: 10d $%.0fM", flow10d/1e6)
				} else if pace30d != 0 {
					ratio := pace10d / pace30d
					etfFlowScore = clampInt(int(ratio*8), -8, 8)
					flowDetail = fmt.Sprintf("10d flow $%.0fM, 10d/30d ratio %.2f", flow10d/1e6, ratio)
				}
			} else {
				flowDetail = "Flat flows"
			}
		} else if len(flows) > 0 {
			flowDetail = fmt.Sprintf("Only %d days of flow data", len(flows))
		}
	}
	if flowDetail == "" {
		flowDetail = "No ETF flow data"
	}

	// ── Signal 5: Venue Wall Convergence (-12 to +12) ──
	venueScore := 0
	venueDetail := ""
	convergenceThresholds := map[int]float64{3: 0.02, 7: 0.025, 14: 0.03, 30: 0.035, 45: 0.04}

	type venueDataT struct {
		CombinedCW float64
		CombinedPW float64
		DeribitCW  float64
		DeribitPW  float64
	}
	var venueData *venueDataT
	venueDTELabel := ""
	venueDTEUsed := 3

	for _, tryDTE := range []int{3, 7, 14, 30, 45} {
		rows := db.GetCacheRowsForDTE(ticker, tryDTE, 1)
		if len(rows) == 0 {
			continue
		}
		var d struct {
			CombinedBTC map[string]interface{} `json:"combined_levels_btc"`
			DeribitBTC  map[string]interface{} `json:"deribit_levels_btc"`
		}
		if err := json.Unmarshal(rows[0].DataJSON, &d); err != nil {
			continue
		}
		if d.CombinedBTC != nil && d.DeribitBTC != nil {
			ccw, _ := d.CombinedBTC["call_wall"].(float64)
			cpw, _ := d.CombinedBTC["put_wall"].(float64)
			dcw, _ := d.DeribitBTC["call_wall"].(float64)
			dpw, _ := d.DeribitBTC["put_wall"].(float64)
			venueData = &venueDataT{ccw, cpw, dcw, dpw}
			venueDTEUsed = tryDTE
			dteLabels := map[int]string{3: "0-3d", 7: "4-7d", 14: "8-14d", 30: "15-30d", 45: "31-45d"}
			venueDTELabel = dteLabels[tryDTE]
			if venueDTELabel == "" {
				venueDTELabel = fmt.Sprintf("%dd", tryDTE)
			}
			break
		}
	}

	if venueData != nil {
		convPct := convergenceThresholds[venueDTEUsed]
		if convPct == 0 {
			convPct = 0.04
		}

		pwConverging := false
		cwConverging := false
		pwDiffPct := 0.0
		cwDiffPct := 0.0

		if venueData.CombinedPW > 0 && venueData.DeribitPW > 0 {
			pwDiffPct = math.Abs(venueData.CombinedPW-venueData.DeribitPW) / venueData.CombinedPW
			if pwDiffPct <= convPct {
				pwConverging = true
			}
		}
		if venueData.CombinedCW > 0 && venueData.DeribitCW > 0 {
			cwDiffPct = math.Abs(venueData.CombinedCW-venueData.DeribitCW) / venueData.CombinedCW
			if cwDiffPct <= convPct {
				cwConverging = true
			}
		}

		threshLabel := fmt.Sprintf("%.0f%% thresh", convPct*100)

		if pwConverging && cwConverging {
			if len(wallHistory) >= 7 {
				lastPW, _ := wallHistory[len(wallHistory)-1]["put_wall"].(float64)
				firstPW, _ := wallHistory[0]["put_wall"].(float64)
				lastCW, _ := wallHistory[len(wallHistory)-1]["call_wall"].(float64)
				firstCW, _ := wallHistory[0]["call_wall"].(float64)

				pwRising := lastPW > firstPW*1.02
				cwFalling := lastCW < firstCW*0.98
				pwFalling := lastPW < firstPW*0.98
				cwRising := lastCW > firstCW*1.02

				if pwRising && cwFalling {
					venueScore = 12
					venueDetail = fmt.Sprintf("Venue range pinch upward — PW conv %.1f%%, CW conv %.1f%% [%s] (%s, migration from 31-45d)", pwDiffPct*100, cwDiffPct*100, venueDTELabel, threshLabel)
				} else if pwFalling && cwRising {
					venueScore = -12
					venueDetail = fmt.Sprintf("Venue range pinch downward — PW conv %.1f%%, CW conv %.1f%% [%s] (%s, migration from 31-45d)", pwDiffPct*100, cwDiffPct*100, venueDTELabel, threshLabel)
				} else if pwRising {
					venueScore = 8
					venueDetail = fmt.Sprintf("Venue range confirmed, support rising — PW %.1f%%, CW %.1f%% [%s] (%s, migration from 31-45d)", pwDiffPct*100, cwDiffPct*100, venueDTELabel, threshLabel)
				} else if cwFalling {
					venueScore = -8
					venueDetail = fmt.Sprintf("Venue range confirmed, resistance falling — PW %.1f%%, CW %.1f%% [%s] (%s, migration from 31-45d)", pwDiffPct*100, cwDiffPct*100, venueDTELabel, threshLabel)
				} else {
					venueDetail = fmt.Sprintf("Venue range confirmed, stable — PW %.1f%%, CW %.1f%% [%s] (%s, migration from 31-45d)", pwDiffPct*100, cwDiffPct*100, venueDTELabel, threshLabel)
				}
			} else {
				venueDetail = fmt.Sprintf("Venue range confirmed (no migration data) — PW %.1f%%, CW %.1f%% [%s] (%s)", pwDiffPct*100, cwDiffPct*100, venueDTELabel, threshLabel)
			}
		} else if pwConverging {
			if len(wallHistory) >= 7 {
				lastPW, _ := wallHistory[len(wallHistory)-1]["put_wall"].(float64)
				firstPW, _ := wallHistory[0]["put_wall"].(float64)
				if lastPW > firstPW*1.02 {
					venueScore = 12
					venueDetail = fmt.Sprintf("Put walls converging (%.1f%%) + rising [%s] (%s, migration from 31-45d)", pwDiffPct*100, venueDTELabel, threshLabel)
				} else {
					venueScore = 6
					venueDetail = fmt.Sprintf("Put walls converging (%.1f%%), stable/falling [%s] (%s, migration from 31-45d)", pwDiffPct*100, venueDTELabel, threshLabel)
				}
			} else {
				venueScore = 4
				venueDetail = fmt.Sprintf("Put walls converging (%.1f%%) [%s] (%s)", pwDiffPct*100, venueDTELabel, threshLabel)
			}
		} else if cwConverging {
			if len(wallHistory) >= 7 {
				lastCW, _ := wallHistory[len(wallHistory)-1]["call_wall"].(float64)
				firstCW, _ := wallHistory[0]["call_wall"].(float64)
				if lastCW < firstCW*0.98 {
					venueScore = -12
					venueDetail = fmt.Sprintf("Call walls converging (%.1f%%) + falling [%s] (%s, migration from 31-45d)", cwDiffPct*100, venueDTELabel, threshLabel)
				} else {
					venueScore = -6
					venueDetail = fmt.Sprintf("Call walls converging (%.1f%%), stable/rising [%s] (%s, migration from 31-45d)", cwDiffPct*100, venueDTELabel, threshLabel)
				}
			} else {
				venueScore = -4
				venueDetail = fmt.Sprintf("Call walls converging (%.1f%%) [%s] (%s)", cwDiffPct*100, venueDTELabel, threshLabel)
			}
		} else {
			venueDetail = fmt.Sprintf("Venue walls not converging — PW diff %.1f%%, CW diff %.1f%% [%s] (%s)", pwDiffPct*100, cwDiffPct*100, venueDTELabel, threshLabel)
		}
	} else {
		venueDetail = "Deribit data not available in any DTE window"
	}

	// ── Phase 2: Coinglass signals ──
	fundingResult := computeFundingSignal()
	oiResult := computeOISignal()
	liqResult := computeLiquidationSignal()

	// ── Phase 3: Options-derived signals ──
	pcrResult := computePCRDirection(ticker, btcPerShare)
	breachResult := computeWallBreach(ticker, btcPerShare)

	// ── Phase 4: Volatility signals ──
	ivTermResult := computeIVTermSignal(ticker, btcPerShare)

	// ── Final Score ──
	totalScore := regimeScore + wallMigScore + compScore + etfFlowScore + venueScore +
		fundingResult.Score + oiResult.Score + liqResult.Score +
		pcrResult.Score + breachResult.Score + ivTermResult.Score
	totalScore = clampInt(totalScore, -100, 100)

	// ── Score History ──
	scoreHistory := ComputeScoreHistory(ticker, days)

	// ── Structural entry / invalidation ──
	var structuralEntry, invalidation *float64
	if math.Abs(float64(totalScore)) > 50 {
		if totalScore > 50 && len(wallHistory) > 0 {
			pw, _ := wallHistory[len(wallHistory)-1]["put_wall"].(float64)
			if pw > 0 {
				structuralEntry = &pw
				inv := pw * 0.97
				invalidation = &inv
			}
		} else if totalScore < -50 && len(wallHistory) > 0 {
			cw, _ := wallHistory[len(wallHistory)-1]["call_wall"].(float64)
			if cw > 0 {
				structuralEntry = &cw
				inv := cw * 1.03
				invalidation = &inv
			}
		}
	}

	// ── Assemble result ──
	bias := "NEUTRAL"
	if totalScore > 30 {
		bias = "BOTTOMING"
	} else if totalScore < -30 {
		bias = "TOPPING"
	}

	swingSignal := "false"
	if totalScore > 50 || totalScore < -50 {
		swingSignal = "true"
	}

	historyMap := map[string]interface{}{
		"wall_migration":        wallHistory,
		"regime_history":        regimeHistory,
		"etf_flows":             flowHistory,
		"funding_rates":         fundingResult.History,
		"oi_history":            oiResult.History,
		"liquidation_history":   liqResult.History,
		"pcr_history":           pcrResult.History,
		"iv_term_history":       ivTermResult.History,
		"score_components_daily": scoreHistory,
	}

	return &model.MacroRegime{
		Score:           totalScore,
		Bias:            bias,
		SwingSignal:     swingSignal,
		HighConviction:  math.Abs(float64(totalScore)) > 75,
		CoinglassAvail:  config.CoinglassAPIKey != "",
		StructuralEntry: structuralEntry,
		Invalidation:    invalidation,
		Signals: map[string]*model.MacroSignal{
			"regime_persistence": {Score: regimeScore, Max: 12, Detail: regimeDetail},
			"wall_migration":     {Score: wallMigScore, Max: 12, Detail: wallDetail},
			"range_compression":  {Score: compScore, Max: 12, Detail: compDetail},
			"etf_flow_momentum":  {Score: etfFlowScore, Max: 12, Detail: flowDetail},
			"venue_convergence":  {Score: venueScore, Max: 12, Detail: venueDetail},
			"funding_rate":       {Score: fundingResult.Score, Max: 13, Detail: fundingResult.Detail, Source: "coinglass"},
			"aggregate_oi":       {Score: oiResult.Score, Max: 13, Detail: oiResult.Detail, Source: "coinglass"},
			"liquidation":        {Score: liqResult.Score, Max: 13, Detail: liqResult.Detail, Source: "coinglass"},
			"pcr_direction":      {Score: pcrResult.Score, Max: 8, Detail: pcrResult.Detail},
			"wall_breach":        {Score: breachResult.Score, Max: 13, Detail: breachResult.Detail},
			"iv_term_structure":  {Score: ivTermResult.Score, Max: 8, Detail: ivTermResult.Detail},
		},
		History:   historyMap,
		Timestamp: nowET().Format(time.RFC3339),
	}
}

func avgField(entries []map[string]interface{}, field string) float64 {
	if len(entries) == 0 {
		return 0
	}
	var sum float64
	for _, e := range entries {
		if v, ok := e[field].(float64); ok {
			sum += v
		}
	}
	return sum / float64(len(entries))
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
