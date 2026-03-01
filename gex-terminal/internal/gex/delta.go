package gex

import (
	"fmt"
	"math"
	"sort"

	"gex-terminal/internal/bs"
	"gex-terminal/internal/model"
)

// CachedChain holds the raw option data needed for dealer delta repricing.
type CachedChain struct {
	ExpiryStr string
	DTE       float64 // days
	Options   []ChainOption
}

// ChainOption is a single option used for delta repricing.
type ChainOption struct {
	Strike     float64
	OI         float64
	IV         float64 // decimal (0.65)
	OptionType string  // "call" or "put"
	Multiplier float64 // 100 for IBIT, 1 for Deribit
	IsBTC      bool    // true if strike is in BTC space
}

// ComputeDealerDeltaScenarios reprices dealer delta across a price grid.
func ComputeDealerDeltaScenarios(
	chains []CachedChain,
	spot float64,
	levels *model.Levels,
	expectedMove *model.ExpectedMove,
	rfr, refPerShare float64,
	isCrypto bool,
	deribitOptions []model.DeribitOption,
) *model.DealerDeltaScenarios {
	// Build price grid
	grid := buildPriceGrid(spot, levels, expectedMove)

	// Compute dealer delta at each grid point
	profile := make([]model.DealerDeltaProfile, 0, len(grid))
	for _, price := range grid {
		netDD, callDD, putDD := computeDeltaAtPrice(chains, price, rfr, deribitOptions, refPerShare, spot)
		profile = append(profile, model.DealerDeltaProfile{
			PriceIBIT:       price,
			PriceBTC:        price / refPerShare,
			NetDealerDelta:  netDD,
			CallDealerDelta: callDD,
			PutDealerDelta:  putDD,
		})
	}

	// Compute acceleration
	for i := 1; i < len(profile)-1; i++ {
		dp := profile[i+1].PriceIBIT - profile[i-1].PriceIBIT
		if dp > 0 {
			ddChange := math.Abs(profile[i+1].NetDealerDelta - profile[i-1].NetDealerDelta)
			profile[i].Acceleration = ddChange / dp
		}
		switch {
		case profile[i].Acceleration > 50000:
			profile[i].AccelClass = "HIGH"
		case profile[i].Acceleration > 10000:
			profile[i].AccelClass = "MODERATE"
		default:
			profile[i].AccelClass = "LOW"
		}
	}

	// Find flip points
	flips := findFlipPoints(profile, refPerShare)

	// Max acceleration price
	maxAccel := 0.0
	maxAccelPrice := spot
	for _, p := range profile {
		if p.Acceleration > maxAccel {
			maxAccel = p.Acceleration
			maxAccelPrice = p.PriceIBIT
		}
	}

	// Key level deltas
	keyDeltas := map[string]float64{}
	for _, name := range []string{"call_wall", "put_wall", "gamma_flip", "max_pain"} {
		var target float64
		switch name {
		case "call_wall":
			target = levels.CallWall
		case "put_wall":
			target = levels.PutWall
		case "gamma_flip":
			target = levels.GammaFlip
		case "max_pain":
			target = levels.MaxPain
		}
		if target > 0 {
			keyDeltas[name] = findClosestDelta(profile, target)
		}
	}

	return &model.DealerDeltaScenarios{
		Profile:              profile,
		FlipPoints:           flips,
		MaxAccelerationPrice: maxAccelPrice / refPerShare,
		KeyLevelDeltas:       keyDeltas,
	}
}

// GenerateDealerDeltaBriefing creates plain-English summary of dealer delta.
func GenerateDealerDeltaBriefing(
	scenarios *model.DealerDeltaScenarios,
	spot float64,
	levels *model.Levels,
	refPerShare float64,
	isCrypto bool,
) *model.DealerDeltaBriefing {
	briefing := &model.DealerDeltaBriefing{
		KeyLevelDeltas: scenarios.KeyLevelDeltas,
	}

	// Current delta (closest to spot)
	curDelta := findClosestDelta(scenarios.Profile, spot)
	curDeltaM := curDelta / 1e6

	if curDelta < 0 {
		briefing.CurrentDelta = fmt.Sprintf("Dealers are net SHORT ~$%.1fM — must BUY on dips (supportive)", math.Abs(curDeltaM))
		briefing.Direction = "short"
	} else {
		briefing.CurrentDelta = fmt.Sprintf("Dealers are net LONG ~$%.1fM — must SELL on rallies (resistive)", math.Abs(curDeltaM))
		briefing.Direction = "long"
	}

	absDelta := math.Abs(curDeltaM)
	switch {
	case absDelta > 10:
		briefing.Magnitude = "massive"
	case absDelta > 5:
		briefing.Magnitude = "substantial"
	case absDelta > 1:
		briefing.Magnitude = "moderate"
	default:
		briefing.Magnitude = "minor"
	}

	// Flip summary
	if len(scenarios.FlipPoints) > 0 {
		fp := scenarios.FlipPoints[0]
		price := fp.PriceBTC
		if !isCrypto {
			price = fp.PriceIBIT
		}
		briefing.FlipSummary = fmt.Sprintf("Dealer pressure flips from %sing to %sing at $%.0f",
			fp.FromDirection, fp.ToDirection, price)
	} else {
		if curDelta < 0 {
			briefing.FlipSummary = "No flip — dealers are net buyers across entire range"
		} else {
			briefing.FlipSummary = "No flip — dealers are net sellers across entire range"
		}
	}

	// Acceleration zone
	briefing.AccelerationZone = fmt.Sprintf("$%.0f", scenarios.MaxAccelerationPrice)

	return briefing
}

func buildPriceGrid(spot float64, levels *model.Levels, em *model.ExpectedMove) []float64 {
	pointSet := map[float64]bool{spot: true}

	// Add major levels
	for _, v := range []float64{levels.CallWall, levels.PutWall, levels.GammaFlip, levels.MaxPain} {
		if v > 0 {
			pointSet[v] = true
		}
	}
	for _, v := range levels.Resistance {
		pointSet[v] = true
	}
	for _, v := range levels.Support {
		pointSet[v] = true
	}

	// Expected move bounds
	if em != nil {
		pointSet[em.UpperIBIT] = true
		pointSet[em.LowerIBIT] = true
	}

	// 0.5% increments between put and call wall
	if levels.PutWall > 0 && levels.CallWall > levels.PutWall {
		step := spot * 0.005
		for p := levels.PutWall; p <= levels.CallWall; p += step {
			pointSet[p] = true
		}
	}

	// Extend ±2% beyond walls
	step := spot * 0.005
	loEnd := spot * 0.85
	hiEnd := spot * 1.15

	for p := levels.PutWall - step; p >= loEnd; p -= step {
		if p > 0 {
			pointSet[p] = true
		}
	}
	for p := levels.CallWall + step; p <= hiEnd; p += step {
		pointSet[p] = true
	}

	// Filter and sort
	grid := make([]float64, 0, len(pointSet))
	for p := range pointSet {
		if p >= loEnd && p <= hiEnd {
			grid = append(grid, p)
		}
	}
	sort.Float64s(grid)

	// Cap to 80 points
	if len(grid) > 80 {
		sampled := make([]float64, 0, 80)
		step := float64(len(grid)) / 78
		for i := 0.0; i < float64(len(grid)); i += step {
			idx := int(i)
			if idx >= len(grid) {
				idx = len(grid) - 1
			}
			sampled = append(sampled, grid[idx])
		}
		// Ensure spot is in the grid
		found := false
		for _, p := range sampled {
			if p == spot {
				found = true
				break
			}
		}
		if !found {
			sampled = append(sampled, spot)
			sort.Float64s(sampled)
		}
		grid = sampled
	}

	return grid
}

func computeDeltaAtPrice(chains []CachedChain, price, rfr float64, deribitOpts []model.DeribitOption, refPerShare, spotIBIT float64) (netDD, callDD, putDD float64) {
	for _, chain := range chains {
		T := math.Max(chain.DTE/365.0, 0.5/365.0)
		for _, opt := range chain.Options {
			if opt.IV <= 0 || opt.OI <= 0 {
				continue
			}
			evalPrice := price
			if opt.IsBTC {
				evalPrice = price / refPerShare
			}

			delta := bs.Delta(evalPrice, opt.Strike, T, rfr, opt.IV, opt.OptionType)
			dd := -delta * opt.OI * opt.Multiplier * evalPrice

			switch opt.OptionType {
			case "call":
				callDD += dd
			case "put":
				putDD += dd
			}
		}
	}

	// Deribit options
	for _, opt := range deribitOpts {
		T := math.Max(opt.DTE/365.0, 0.5/365.0)
		iv := opt.IV / 100.0 // percentage to decimal
		btcPrice := price / refPerShare

		delta := bs.Delta(btcPrice, opt.Strike, T, rfr, iv, opt.OptionType)
		dd := -delta * opt.OI * 1.0 * btcPrice

		switch opt.OptionType {
		case "call":
			callDD += dd
		case "put":
			putDD += dd
		}
	}

	netDD = callDD + putDD
	return
}

func findFlipPoints(profile []model.DealerDeltaProfile, refPerShare float64) []model.DeltaFlipPoint {
	var flips []model.DeltaFlipPoint
	for i := 0; i < len(profile)-1; i++ {
		d1 := profile[i].NetDealerDelta
		d2 := profile[i+1].NetDealerDelta

		if (d1 < 0 && d2 > 0) || (d1 > 0 && d2 < 0) {
			p1 := profile[i].PriceIBIT
			p2 := profile[i+1].PriceIBIT
			flip := p1 + (p2-p1)*(-d1)/(d2-d1)

			fromDir := "buy"
			toDir := "sell"
			if d1 > 0 {
				fromDir = "sell"
				toDir = "buy"
			}

			flips = append(flips, model.DeltaFlipPoint{
				PriceIBIT:     flip,
				PriceBTC:      flip / refPerShare,
				FromDirection: fromDir,
				ToDirection:   toDir,
			})
		}
	}
	return flips
}

func findClosestDelta(profile []model.DealerDeltaProfile, target float64) float64 {
	bestDist := math.Inf(1)
	bestDelta := 0.0
	for _, p := range profile {
		dist := math.Abs(p.PriceIBIT - target)
		if dist < bestDist {
			bestDist = dist
			bestDelta = p.NetDealerDelta
		}
	}
	return bestDelta
}
