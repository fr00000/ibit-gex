package gex

import (
	"fmt"
	"math"

	"gex-terminal/internal/model"
)

// ComputeFlowForecast computes charm/vanna flow predictions.
func ComputeFlowForecast(strikes []model.StrikeData, spot float64, levels *model.Levels, isCrypto bool) *model.FlowForecast {
	var sumVanna, sumCharm float64
	for _, s := range strikes {
		sumVanna += s.NetVanna
		sumCharm += s.NetCharm
	}

	forecast := &model.FlowForecast{}

	// Charm analysis
	charmDir := "buy"
	if sumCharm < 0 {
		charmDir = "sell"
	}
	charmMag := math.Abs(sumCharm)
	charmStrength := flowStrength(charmMag)

	charmNarrative := fmt.Sprintf("Dealers will %s ~$%.0fK notional overnight from charm decay",
		charmDir, charmMag/1000)

	forecast.Charm = &model.FlowComponent{
		Direction:   charmDir,
		NetNotional: sumCharm,
		Strength:    charmStrength,
		Narrative:   charmNarrative,
	}

	// Vanna analysis
	vannaStrength := flowStrength(math.Abs(sumVanna))

	// Vanna scenarios: 5-point IV moves
	crushNotional := sumVanna * 5   // -5pt IV crush
	spikeNotional := -sumVanna * 5  // +5pt IV spike

	crushDir := "buy"
	if crushNotional < 0 {
		crushDir = "sell"
	}
	spikeDir := "buy"
	if spikeNotional < 0 {
		spikeDir = "sell"
	}

	forecast.Vanna = &model.VannaForecast{
		NetNotional: sumVanna,
		Strength:    vannaStrength,
		CrushScenario: &model.FlowComponent{
			Direction:   crushDir,
			NetNotional: crushNotional,
			Narrative:   fmt.Sprintf("Vol crush (-5pt): dealers %s ~$%.0fK", crushDir, math.Abs(crushNotional)/1000),
		},
		SpikeScenario: &model.FlowComponent{
			Direction:   spikeDir,
			NetNotional: spikeNotional,
			Narrative:   fmt.Sprintf("Vol spike (+5pt): dealers %s ~$%.0fK", spikeDir, math.Abs(spikeNotional)/1000),
		},
	}

	// Overnight forecast (charm + conservative vanna decay ~0.5pt)
	overnightVannaAdj := sumVanna * 0.5
	overnightTotal := sumCharm + overnightVannaAdj
	overnightDir := "buy"
	if overnightTotal < 0 {
		overnightDir = "sell"
	}

	forecast.Overnight = &model.FlowComponent{
		Direction:   overnightDir,
		NetNotional: overnightTotal,
		Strength:    flowStrength(math.Abs(overnightTotal)),
	}

	// Regime note
	if levels.Regime == "positive_gamma" {
		forecast.RegimeNote = "Positive gamma reinforces mean-reversion. Charm and vanna flows add directional bias within the range."
	} else {
		forecast.RegimeNote = "Negative gamma amplifies moves. Vanna and charm flows can accelerate breakouts."
	}

	return forecast
}

func flowStrength(magnitude float64) string {
	switch {
	case magnitude < 50000:
		return "negligible"
	case magnitude < 500000:
		return "minor"
	case magnitude < 2500000:
		return "moderate"
	default:
		return "significant"
	}
}
