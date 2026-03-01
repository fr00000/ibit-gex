package gex

import (
	"math"
	"sort"

	"gex-terminal/internal/model"
)

// ComputeLevels computes wall positions, flip points, regime, max pain from strike data.
// This is the Go port of _compute_levels_from_df.
func ComputeLevels(strikes []model.StrikeData, spot float64, etfFlows *model.ETFFlows) *model.Levels {
	if len(strikes) == 0 || spot <= 0 {
		return &model.Levels{
			PositioningWarnings: []string{},
			Resistance:          []float64{},
			Support:             []float64{},
		}
	}

	levels := &model.Levels{
		PositioningWarnings: []string{},
		PositioningConfidence: 100,
	}

	// Sort by strike price
	sorted := make([]model.StrikeData, len(strikes))
	copy(sorted, strikes)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Strike < sorted[j].Strike })

	// ── Totals ──
	var sumNetGEX, sumDealerDelta, sumVanna, sumCharm float64
	var sumCallOI, sumPutOI int64
	var sumVolGEX, sumActiveGEX float64

	for _, s := range sorted {
		sumNetGEX += s.NetGEX
		sumDealerDelta += s.NetDealerDelta
		sumVanna += s.NetVanna
		sumCharm += s.NetCharm
		sumCallOI += s.CallOI
		sumPutOI += s.PutOI
		sumVolGEX += s.CallVolGEX + s.PutVolGEX
		sumActiveGEX += s.ActiveGEX
	}

	levels.NetGEXTotal = sumNetGEX
	levels.ActiveGEXTotal = sumActiveGEX
	levels.NetDealerDelta = sumDealerDelta
	levels.NetVanna = sumVanna
	levels.NetCharm = sumCharm
	levels.TotalCallOI = sumCallOI
	levels.TotalPutOI = sumPutOI
	levels.VolumeGEXTotal = sumVolGEX

	if sumCallOI > 0 {
		levels.PCR = float64(sumPutOI) / float64(sumCallOI)
	}
	if sumNetGEX != 0 {
		levels.GEXActivityRatio = math.Abs(sumVolGEX) / math.Abs(sumNetGEX)
	}

	// ── Call Wall: highest net_gex strike at or above spot ──
	callWallGEX := math.Inf(-1)
	for _, s := range sorted {
		if s.Strike >= spot && s.NetGEX > callWallGEX {
			callWallGEX = s.NetGEX
			levels.CallWall = s.Strike
		}
	}
	if math.IsInf(callWallGEX, -1) {
		// No strikes above spot — use global max
		for _, s := range sorted {
			if s.NetGEX > callWallGEX {
				callWallGEX = s.NetGEX
				levels.CallWall = s.Strike
			}
		}
	}

	// ── Put Wall: most negative net_gex strike at or below spot ──
	putWallGEX := math.Inf(1)
	for _, s := range sorted {
		if s.Strike <= spot && s.NetGEX < putWallGEX {
			putWallGEX = s.NetGEX
			levels.PutWall = s.Strike
		}
	}
	if math.IsInf(putWallGEX, 1) {
		for _, s := range sorted {
			if s.NetGEX < putWallGEX {
				putWallGEX = s.NetGEX
				levels.PutWall = s.Strike
			}
		}
	}

	// ── Gamma Flip: interpolated zero crossing nearest spot ──
	levels.GammaFlip = computeGammaFlip(sorted, spot)

	// ── Max Pain ──
	levels.MaxPain = computeMaxPain(sorted)

	// ── Regime: net GEX within 2% of spot ──
	nearSpotGEX := 0.0
	for _, s := range sorted {
		if s.Strike >= spot*0.98 && s.Strike <= spot*1.02 {
			nearSpotGEX += s.NetGEX
		}
	}
	if nearSpotGEX > 0 {
		levels.Regime = "positive_gamma"
	} else {
		levels.Regime = "negative_gamma"
	}

	// ── Active GEX Walls ──
	activeCallGEX := math.Inf(-1)
	activePutGEX := math.Inf(1)
	for _, s := range sorted {
		gex := s.ActiveGEX
		if gex == 0 {
			gex = s.NetGEX
		}
		if gex > activeCallGEX {
			activeCallGEX = gex
			levels.ActiveCallWall = s.Strike
		}
		if gex < activePutGEX {
			activePutGEX = gex
			levels.ActivePutWall = s.Strike
		}
	}

	// ── Resistance / Support ──
	levels.Resistance = topNByGEX(sorted, 3, true)
	levels.Support = topNByGEX(sorted, 3, false)

	// ── Positioning Confidence ──
	computeConfidence(levels, sorted, etfFlows)

	return levels
}

func computeGammaFlip(sorted []model.StrikeData, spot float64) float64 {
	var bestFlip float64
	bestDist := math.Inf(1)

	for i := 0; i < len(sorted)-1; i++ {
		g1 := sorted[i].NetGEX
		g2 := sorted[i+1].NetGEX

		if (g1 > 0 && g2 < 0) || (g1 < 0 && g2 > 0) {
			s1 := sorted[i].Strike
			s2 := sorted[i+1].Strike
			// Linear interpolation
			flip := s1 + (s2-s1)*(-g1)/(g2-g1)
			dist := math.Abs(flip - spot)
			if dist < bestDist {
				bestDist = dist
				bestFlip = flip
			}
		}
	}

	if math.IsInf(bestDist, 1) {
		return spot // fallback
	}
	return bestFlip
}

func computeMaxPain(sorted []model.StrikeData) float64 {
	if len(sorted) == 0 {
		return 0
	}

	bestPain := math.Inf(1)
	bestStrike := sorted[len(sorted)/2].Strike

	for _, settle := range sorted {
		totalPain := 0.0
		for _, s := range sorted {
			// Call pain
			if settle.Strike > s.Strike {
				totalPain += float64(s.CallOI) * (settle.Strike - s.Strike) * 100
			}
			// Put pain
			if s.Strike > settle.Strike {
				totalPain += float64(s.PutOI) * (s.Strike - settle.Strike) * 100
			}
		}
		if totalPain < bestPain {
			bestPain = totalPain
			bestStrike = settle.Strike
		}
	}

	return bestStrike
}

func topNByGEX(sorted []model.StrikeData, n int, positive bool) []float64 {
	type entry struct {
		strike float64
		gex    float64
	}
	var candidates []entry
	for _, s := range sorted {
		if positive && s.NetGEX > 0 {
			candidates = append(candidates, entry{s.Strike, s.NetGEX})
		} else if !positive && s.NetGEX < 0 {
			candidates = append(candidates, entry{s.Strike, s.NetGEX})
		}
	}

	if positive {
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].gex > candidates[j].gex })
	} else {
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].gex < candidates[j].gex })
	}

	result := make([]float64, 0, n)
	for i := 0; i < n && i < len(candidates); i++ {
		result = append(result, candidates[i].strike)
	}
	return result
}

func computeConfidence(levels *model.Levels, sorted []model.StrikeData, etfFlows *model.ETFFlows) {
	confidence := 100

	// P/C ratio check
	if levels.PCR < 0.5 {
		confidence -= 40
		levels.PositioningWarnings = append(levels.PositioningWarnings, "Extreme call skew (P/C < 0.5)")
	} else if levels.PCR < 0.7 {
		confidence -= 25
		levels.PositioningWarnings = append(levels.PositioningWarnings, "Call-heavy positioning (P/C < 0.7)")
	} else if levels.PCR < 0.85 {
		confidence -= 10
		levels.PositioningWarnings = append(levels.PositioningWarnings, "Mild call skew (P/C < 0.85)")
	}

	// OTM call concentration check
	if levels.TotalCallOI > 0 {
		var otmCallOI int64
		for _, s := range sorted {
			// Simplified: check if heavily call-skewed
			if s.CallOI > 0 && s.CallOI > s.PutOI*3 {
				otmCallOI += s.CallOI
			}
		}
		if float64(otmCallOI) > float64(levels.TotalCallOI)*0.6 {
			confidence -= 15
			levels.PositioningWarnings = append(levels.PositioningWarnings, ">60% OTM call OI — speculative accumulation")
		}
	}

	// Single strike concentration check
	for _, s := range sorted {
		if levels.TotalCallOI > 0 && float64(s.CallOI) > float64(levels.TotalCallOI)*0.2 {
			confidence -= 10
			levels.PositioningWarnings = append(levels.PositioningWarnings, "Single strike >20% of call OI — concentrated bet")
			break
		}
	}

	// ETF flow modifier
	if etfFlows != nil && etfFlows.Streak <= -3 && (etfFlows.Strength == "moderate" || etfFlows.Strength == "strong") {
		confidence -= 15
		levels.PositioningWarnings = append(levels.PositioningWarnings, "3+ day outflow streak — institutional exit")
	}

	if confidence < 0 {
		confidence = 0
	}
	levels.PositioningConfidence = confidence
}
