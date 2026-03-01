package bs

import "math"

// d1d2 computes the shared d1/d2 values for Black-Scholes.
func d1d2(S, K, T, r, sigma float64) (float64, float64) {
	sqrtT := sigma * math.Sqrt(T)
	d1 := (math.Log(S/K) + (r+0.5*sigma*sigma)*T) / sqrtT
	d2 := d1 - sqrtT
	return d1, d2
}

// normPDF is the standard normal probability density function.
func normPDF(x float64) float64 {
	return math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
}

// normCDF is the standard normal cumulative distribution function.
func normCDF(x float64) float64 {
	return 0.5 * math.Erfc(-x/math.Sqrt2)
}

// Gamma computes Black-Scholes gamma.
// Returns 0 if T <= 0, sigma <= 0, or S <= 0.
func Gamma(S, K, T, r, sigma float64) float64 {
	if T <= 0 || sigma <= 0 || S <= 0 {
		return 0
	}
	d1, _ := d1d2(S, K, T, r, sigma)
	return normPDF(d1) / (S * sigma * math.Sqrt(T))
}

// Delta computes Black-Scholes delta.
// optionType is "call" or "put".
func Delta(S, K, T, r, sigma float64, optionType string) float64 {
	if T <= 0 || sigma <= 0 || S <= 0 {
		return 0
	}
	d1, _ := d1d2(S, K, T, r, sigma)
	if optionType == "call" {
		return normCDF(d1)
	}
	return normCDF(d1) - 1
}

// Vanna computes dDelta/dVol — dealer rebalancing when IV moves.
func Vanna(S, K, T, r, sigma float64) float64 {
	if T <= 0 || sigma <= 0 || S <= 0 {
		return 0
	}
	d1, d2 := d1d2(S, K, T, r, sigma)
	return -normPDF(d1) * d2 / sigma
}

// Charm computes dDelta/dT — dealer rebalancing from time decay of delta.
// optionType is "call" or "put".
func Charm(S, K, T, r, sigma float64, optionType string) float64 {
	if T <= 0 || sigma <= 0 || S <= 0 {
		return 0
	}
	d1, d2 := d1d2(S, K, T, r, sigma)
	sqrtT := math.Sqrt(T)
	charm := -normPDF(d1) * (2*r*T - d2*sigma*sqrtT) / (2 * T * sigma * sqrtT)
	if optionType == "put" {
		charm += r * math.Exp(-r*T) * normCDF(-d2)
	}
	return charm
}
