package macro

import (
	"encoding/json"
	"sort"

	"gex-terminal/internal/config"
	"gex-terminal/internal/db"
)

// parsedCacheEntry holds pre-parsed data from the data_cache table.
type parsedCacheEntry struct {
	CallWall       float64
	PutWall        float64
	IBITCallWall   float64
	IBITPutWall    float64
	DeribitCW      float64
	DeribitPW      float64
	HasDeribit     bool
	PCR            *float64
	BTCSpot        float64
	Regime         string
	IVTermStructure []map[string]interface{}
}

// ComputeScoreHistory recomputes full macro score (all 11 signals) for each of the last N days.
func ComputeScoreHistory(ticker string, days int) []map[string]interface{} {
	if days <= 0 {
		days = 30
	}
	cfg := config.TickerConfig[ticker]
	if cfg == nil {
		cfg = config.TickerConfig["IBIT"]
	}
	btcPerShare := cfg.PerShareDefault

	lookback := days + 30

	// Pre-load all data
	allSnapshots, err := db.GetHistory(ticker, lookback)
	if err != nil {
		return nil
	}

	allCache := db.GetCacheRowsForDTE(ticker, 45, lookback)

	allFlows := db.GetFlowsRaw(ticker, lookback)

	allFunding, _ := db.GetCoinglassHistory("BTC", "avg_funding_rate", 90)
	allOI, _ := db.GetCoinglassHistory("BTC", "total_oi_usd", 90)
	allLiquidation, _ := db.GetCoinglassHistory("BTC", "total_liquidation_usd", 90)
	allLiqSnapshots, _ := db.GetCoinglassHistory("BTC", "liquidation_snapshot", 90)

	// Get dates we want scores for
	if len(allSnapshots) == 0 {
		return nil
	}
	seen := map[string]bool{}
	var dates []string
	limit := min(days, len(allSnapshots))
	for i := 0; i < limit; i++ {
		d := allSnapshots[i].Date
		if !seen[d] {
			seen[d] = true
			dates = append(dates, d)
		}
	}
	// Reverse to oldest first
	for i, j := 0, len(dates)-1; i < j; i, j = i+1, j-1 {
		dates[i], dates[j] = dates[j], dates[i]
	}

	// Pre-parse cache JSON
	parsedCache := map[string]*parsedCacheEntry{}
	for _, row := range allCache {
		if _, exists := parsedCache[row.Date]; exists {
			continue
		}
		var d struct {
			CombinedBTC map[string]interface{} `json:"combined_levels_btc"`
			DeribitBTC  map[string]interface{} `json:"deribit_levels_btc"`
			BTCSpot     float64                `json:"btc_spot"`
			Levels      struct {
				CallWall float64  `json:"call_wall"`
				PutWall  float64  `json:"put_wall"`
				Regime   string   `json:"regime"`
				PCR      *float64 `json:"pcr"`
			} `json:"levels"`
			IVTermStructure []map[string]interface{} `json:"iv_term_structure"`
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

		ibitCW := 0.0
		ibitPW := 0.0
		if btcPerShare > 0 {
			ibitCW = d.Levels.CallWall / btcPerShare
			ibitPW = d.Levels.PutWall / btcPerShare
		}

		deribitCW := 0.0
		deribitPW := 0.0
		if d.DeribitBTC != nil {
			if v, ok := d.DeribitBTC["call_wall"].(float64); ok {
				deribitCW = v
			}
			if v, ok := d.DeribitBTC["put_wall"].(float64); ok {
				deribitPW = v
			}
		}

		parsedCache[row.Date] = &parsedCacheEntry{
			CallWall:        cw,
			PutWall:         pw,
			IBITCallWall:    ibitCW,
			IBITPutWall:     ibitPW,
			DeribitCW:       deribitCW,
			DeribitPW:       deribitPW,
			HasDeribit:      d.CombinedBTC != nil && d.DeribitBTC != nil,
			PCR:             d.Levels.PCR,
			BTCSpot:         d.BTCSpot,
			Regime:          d.Levels.Regime,
			IVTermStructure: d.IVTermStructure,
		}
	}

	var scoreHistory []map[string]interface{}

	for _, dateStr := range dates {
		// ── Signal 1: Regime Persistence & Transition ──
		var snaps []struct{ Regime string }
		for _, s := range allSnapshots {
			if s.Date <= dateStr {
				snaps = append(snaps, struct{ Regime string }{s.Regime})
				if len(snaps) >= days {
					break
				}
			}
		}

		regimeS := 0
		if len(snaps) > 0 {
			total := len(snaps)
			negCount := 0
			for _, s := range snaps {
				if s.Regime == "negative_gamma" {
					negCount++
				}
			}
			posCount := total - negCount
			negPct := float64(negCount) / float64(total)
			posPct := float64(posCount) / float64(total)

			if negPct > 0.70 {
				regimeS = 8
			} else if posPct > 0.70 {
				regimeS = -8
			}

			// Transition bonus
			if len(snaps) >= 4 {
				recentRegime := snaps[0].Regime
				recentStreak := 1
				for i := 1; i < min(4, len(snaps)); i++ {
					if snaps[i].Regime == recentRegime {
						recentStreak++
					} else {
						break
					}
				}
				if recentStreak <= 3 {
					oldCount := 0
					for i := recentStreak; i < len(snaps); i++ {
						if snaps[i].Regime != recentRegime {
							oldCount++
						} else {
							break
						}
					}
					if oldCount >= 7 {
						if recentRegime == "positive_gamma" {
							regimeS = 12
						} else {
							regimeS = -12
						}
					}
				}
			}
		}

		// ── Signal 2: Wall Migration ──
		wallS := 0
		var cacheDates []string
		for dt := range parsedCache {
			if dt <= dateStr {
				cacheDates = append(cacheDates, dt)
			}
		}
		sort.Strings(cacheDates)

		var wallEntries []*parsedCacheEntry
		for _, dt := range cacheDates {
			e := parsedCache[dt]
			if e.CallWall > 0 && e.PutWall > 0 {
				wallEntries = append(wallEntries, e)
			}
		}

		if len(wallEntries) >= 7 {
			half := len(wallEntries) / 2
			firstHalf := wallEntries[:half]
			lastHalf := wallEntries[half:]

			avgPWF := avgParsed(firstHalf, func(e *parsedCacheEntry) float64 { return e.PutWall })
			avgPWL := avgParsed(lastHalf, func(e *parsedCacheEntry) float64 { return e.PutWall })
			avgCWF := avgParsed(firstHalf, func(e *parsedCacheEntry) float64 { return e.CallWall })
			avgCWL := avgParsed(lastHalf, func(e *parsedCacheEntry) float64 { return e.CallWall })

			pwChg := 0.0
			if avgPWF > 0 {
				pwChg = (avgPWL - avgPWF) / avgPWF
			}
			cwChg := 0.0
			if avgCWF > 0 {
				cwChg = (avgCWL - avgCWF) / avgCWF
			}

			pwPart := 0
			if pwChg > 0.02 {
				pwPart = 6
			} else if pwChg < -0.02 {
				pwPart = -6
			}
			cwPart := 0
			if cwChg > 0.02 {
				cwPart = 6
			} else if cwChg < -0.02 {
				cwPart = -6
			}
			wallS = pwPart + cwPart
		}

		// ── Signal 3: Range Compression ──
		compS := 0
		if len(wallEntries) >= 10 {
			var rangeVals []float64
			for _, w := range wallEntries {
				if w.CallWall > w.PutWall {
					rangeVals = append(rangeVals, w.CallWall-w.PutWall)
				}
			}
			if len(rangeVals) > 0 {
				currentRange := rangeVals[len(rangeVals)-1]
				sorted := make([]float64, len(rangeVals))
				copy(sorted, rangeVals)
				sort.Float64s(sorted)
				below := 0
				for _, r := range sorted {
					if r < currentRange {
						below++
					}
				}
				percentile := float64(below) / float64(len(sorted)) * 100
				if percentile <= 20 && len(snaps) > 0 {
					curRegime := snaps[0].Regime
					if curRegime == "positive_gamma" {
						compS = -12
					} else if curRegime == "negative_gamma" {
						compS = 12
					}
				}
			}
		}

		// ── Signal 4: ETF Flow Momentum ──
		flowS := 0
		var flowsUpTo []float64
		for _, r := range allFlows {
			if r.Date <= dateStr {
				if r.TotalBTCETFFlow != nil {
					flowsUpTo = append(flowsUpTo, *r.TotalBTCETFFlow)
				} else {
					flowsUpTo = append(flowsUpTo, r.DailyFlowDollars)
				}
			}
		}

		if len(flowsUpTo) > 0 {
			flow10d := 0.0
			for i := 0; i < min(10, len(flowsUpTo)); i++ {
				flow10d += flowsUpTo[i]
			}
			if len(flowsUpTo) >= 15 {
				flow30d := 0.0
				for i := 0; i < min(30, len(flowsUpTo)); i++ {
					flow30d += flowsUpTo[i]
				}
				f5Recent := 0.0
				for i := 0; i < 5; i++ {
					f5Recent += flowsUpTo[i]
				}
				f5Prior := 0.0
				for i := 5; i < min(10, len(flowsUpTo)); i++ {
					f5Prior += flowsUpTo[i]
				}

				if f5Prior < 0 && f5Recent > 0 {
					flowS = 12
				} else if f5Prior > 0 && f5Recent < 0 {
					flowS = -12
				} else if flow30d != 0 {
					pace10d := flow10d / float64(min(10, len(flowsUpTo)))
					pace30d := flow30d / float64(min(30, len(flowsUpTo)))
					if flow10d > 0 && pace10d > pace30d {
						flowS = 8
					} else if flow10d < 0 && pace10d < -pace30d {
						flowS = -8
					} else if pace30d != 0 {
						flowS = clampInt(int(pace10d/pace30d*8), -8, 8)
					}
				}
			}
		}

		// ── Signal 5: Venue Wall Convergence ──
		venueS := 0
		if len(cacheDates) > 0 {
			latest := parsedCache[cacheDates[len(cacheDates)-1]]
			if latest.HasDeribit {
				convThresh := 0.04 // Use dte=45 threshold for score history

				if latest.CombinedPW() > 0 && latest.DeribitPW > 0 {
					pwDiff := abs64(latest.PutWall-latest.DeribitPW) / latest.PutWall
					if pwDiff <= convThresh && len(wallEntries) >= 7 {
						pwRising := wallEntries[len(wallEntries)-1].PutWall > wallEntries[0].PutWall*1.02
						if pwRising {
							venueS += 12
						} else {
							venueS += 6
						}
					}
				}

				if latest.CombinedCW() > 0 && latest.DeribitCW > 0 {
					cwDiff := abs64(latest.CallWall-latest.DeribitCW) / latest.CallWall
					if cwDiff <= convThresh && len(wallEntries) >= 7 {
						cwFalling := wallEntries[len(wallEntries)-1].CallWall < wallEntries[0].CallWall*0.98
						if cwFalling {
							venueS -= 12
						} else {
							venueS -= 6
						}
					}
				}

				venueS = clampInt(venueS, -12, 12)
			}
		}

		// ── Signal 6: Funding Rate ──
		fundingS := 0
		var frUpTo []float64
		for _, r := range allFunding {
			if r.Date <= dateStr {
				frUpTo = append(frUpTo, r.Value)
			}
		}
		if len(frUpTo) >= 7 {
			var sum7d float64
			for i := 0; i < 7; i++ {
				sum7d += frUpTo[i]
			}
			avg7d := sum7d / 7
			var sum30d float64
			for i := 0; i < min(30, len(frUpTo)); i++ {
				sum30d += frUpTo[i]
			}
			avg30d := sum30d / float64(min(30, len(frUpTo)))

			if avg7d < -0.0001 {
				if avg7d > avg30d {
					fundingS = 13
				} else {
					fundingS = 8
				}
			} else if avg7d > 0.0003 {
				if avg7d < avg30d {
					fundingS = -13
				} else {
					fundingS = -8
				}
			}
		}

		// ── Signal 7: Open Interest ──
		oiS := 0
		oiSlope := "unknown"
		var oiUpTo []float64
		for _, r := range allOI {
			if r.Date <= dateStr {
				oiUpTo = append(oiUpTo, r.Value)
			}
		}
		if len(oiUpTo) >= 7 {
			oiCurrent := oiUpTo[0]
			oiPeak := oiUpTo[0]
			for _, v := range oiUpTo {
				if v > oiPeak {
					oiPeak = v
				}
			}
			chgFromPeak := 0.0
			if oiPeak > 0 {
				chgFromPeak = (oiCurrent - oiPeak) / oiPeak * 100
			}

			oi7dAgo := oiUpTo[6]
			if oiCurrent > oi7dAgo*1.02 {
				oiSlope = "rising"
			} else if oiCurrent < oi7dAgo*0.98 {
				oiSlope = "falling"
			} else {
				oiSlope = "flat"
			}

			var frForOI []float64
			for _, r := range allFunding {
				if r.Date <= dateStr {
					frForOI = append(frForOI, r.Value)
					if len(frForOI) >= 7 {
						break
					}
				}
			}
			fundingAvg := 0.0
			if len(frForOI) > 0 {
				for _, v := range frForOI {
					fundingAvg += v
				}
				fundingAvg /= float64(len(frForOI))
			}

			if chgFromPeak < -20 {
				if oiSlope == "flat" || oiSlope == "rising" {
					oiS = 13
				} else {
					oiS = 6
				}
			} else if chgFromPeak > -5 {
				if fundingAvg > 0 {
					oiS = -13
				} else {
					oiS = -6
				}
			}
		}

		// ── Signal 8: Liquidation Intensity ──
		liqS := 0
		// Find liq snapshot up to this date
		var liqSnap *struct {
			Value     float64
			ExtraJSON string
		}
		for _, r := range allLiqSnapshots {
			if r.Date <= dateStr {
				liqSnap = &struct {
					Value     float64
					ExtraJSON string
				}{r.Value, r.ExtraJSON}
				break
			}
		}
		var liqHist []struct {
			Value float64
		}
		for _, r := range allLiquidation {
			if r.Date <= dateStr {
				liqHist = append(liqHist, struct{ Value float64 }{r.Value})
			}
		}

		if liqSnap != nil && liqSnap.ExtraJSON != "" && len(liqHist) >= 7 {
			var liqExtra map[string]interface{}
			if err := json.Unmarshal([]byte(liqSnap.ExtraJSON), &liqExtra); err == nil {
				long24h, _ := liqExtra["long_24h"].(float64)
				short24h, _ := liqExtra["short_24h"].(float64)
				total24h := long24h + short24h
				var sum30d float64
				for i := 0; i < min(30, len(liqHist)); i++ {
					sum30d += liqHist[i].Value
				}
				avg30d := sum30d / float64(min(30, len(liqHist)))
				intensity := 0.0
				if avg30d > 0 {
					intensity = total24h / avg30d
				}
				longRatio := 0.5
				if total24h > 0 {
					longRatio = long24h / total24h
				}

				if intensity >= 2.0 && (longRatio > 0.65 || longRatio < 0.35) {
					magnitude := 6
					if intensity >= 3.0 {
						magnitude = 13
					}
					if longRatio > 0.65 { // Long dominant
						if oiSlope != "falling" {
							liqS = magnitude
						} else {
							liqS = max(6, magnitude-4)
						}
					} else { // Short dominant
						if oiSlope != "rising" {
							liqS = -magnitude
						} else {
							liqS = -max(6, magnitude-4)
						}
					}
				}
			}
		}

		// ── Signal 9: PCR Direction ──
		pcrS := 0
		if len(cacheDates) > 0 {
			var pcrVals []float64
			for _, dt := range cacheDates {
				if p := parsedCache[dt].PCR; p != nil {
					pcrVals = append(pcrVals, *p)
				}
			}
			if len(pcrVals) >= 5 {
				recent3 := pcrVals[len(pcrVals)-3:]
				priorPCR := pcrVals[:len(pcrVals)-3]
				if len(priorPCR) > 0 {
					var sumR, sumP float64
					for _, v := range recent3 {
						sumR += v
					}
					for _, v := range priorPCR {
						sumP += v
					}
					avgR := sumR / float64(len(recent3))
					avgP := sumP / float64(len(priorPCR))
					pctChg := 0.0
					if avgP != 0 {
						pctChg = (avgR - avgP) / avgP
					}
					if pctChg > 0.15 {
						pcrS = 8
					} else if pctChg > 0.08 {
						pcrS = 4
					} else if pctChg < -0.15 {
						pcrS = -8
					} else if pctChg < -0.08 {
						pcrS = -4
					}
				}
			}
		}

		// ── Signal 10: Wall Breach Detection ──
		breachS := 0
		if len(cacheDates) > 0 {
			latestWB := parsedCache[cacheDates[len(cacheDates)-1]]
			wbSpot := latestWB.BTCSpot
			wbCW := latestWB.CallWall
			wbPW := latestWB.PutWall
			wbRegime := latestWB.Regime
			if wbSpot > 0 && wbCW > 0 && wbPW > 0 {
				cwDist := (wbSpot - wbCW) / wbCW
				pwDist := (wbPW - wbSpot) / wbPW
				if cwDist > 0.005 {
					if wbRegime == "negative_gamma" {
						breachS = 13
					} else {
						breachS = 6
					}
				} else if pwDist > 0.005 {
					if wbRegime == "negative_gamma" {
						breachS = -13
					} else {
						breachS = -6
					}
				}
			}
		}

		// ── Signal 11: IV Term Structure Shape ──
		ivTermS := 0
		if len(cacheDates) > 0 {
			latestIV := parsedCache[cacheDates[len(cacheDates)-1]]
			ivTSData := latestIV.IVTermStructure
			ivBySrc := map[string][]map[string]interface{}{}
			for _, e := range ivTSData {
				src, _ := e["source"].(string)
				if src == "" {
					src = "ibit"
				}
				ivBySrc[src] = append(ivBySrc[src], e)
			}
			ivSrc := "ibit"
			if len(ivBySrc["deribit"]) >= 2 {
				ivSrc = "deribit"
			}
			ivEntries := ivBySrc[ivSrc]
			sort.Slice(ivEntries, func(a, b int) bool {
				da, _ := ivEntries[a]["dte"].(float64)
				db, _ := ivEntries[b]["dte"].(float64)
				return da < db
			})
			if len(ivEntries) >= 2 {
				sIV, _ := ivEntries[0]["atm_iv"].(float64)
				lIV, _ := ivEntries[len(ivEntries)-1]["atm_iv"].(float64)
				if sIV > 0 {
					ivSlope := (lIV - sIV) / sIV
					if ivSlope < -0.10 {
						ivTermS = 8
					} else if ivSlope < -0.03 {
						ivTermS = 4
					} else if ivSlope > 0.25 {
						ivTermS = -8
					} else if ivSlope > 0.15 {
						ivTermS = -4
					}
				}
			}
		}

		dayTotal := clampInt(
			regimeS+wallS+compS+flowS+venueS+fundingS+oiS+liqS+pcrS+breachS+ivTermS,
			-100, 100,
		)

		scoreHistory = append(scoreHistory, map[string]interface{}{
			"date":        dateStr,
			"total_score": dayTotal,
			"regime":      regimeS,
			"walls":       wallS,
			"compression": compS,
			"flow":        flowS,
			"venue":       venueS,
			"funding":     fundingS,
			"oi":          oiS,
			"liquidation": liqS,
			"pcr":         pcrS,
			"breach":      breachS,
			"iv_term":     ivTermS,
		})
	}

	return scoreHistory
}

// Helper methods on parsedCacheEntry.
func (e *parsedCacheEntry) CombinedPW() float64 { return e.PutWall }
func (e *parsedCacheEntry) CombinedCW() float64 { return e.CallWall }

func avgParsed(entries []*parsedCacheEntry, getter func(*parsedCacheEntry) float64) float64 {
	if len(entries) == 0 {
		return 0
	}
	var sum float64
	for _, e := range entries {
		sum += getter(e)
	}
	return sum / float64(len(entries))
}

func abs64(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}
