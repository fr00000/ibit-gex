package bs

import (
	"math"
	"testing"
)

type testVector struct {
	S, K, T, r, sigma float64
	gamma             float64
	deltaCall         float64
	deltaPut          float64
	vanna             float64
	charmCall         float64
	charmPut          float64
}

// Test vectors generated from Python scipy.stats.norm.
var vectors = []testVector{
	{100, 100, 0.1, 0.05, 0.2, 0.06269313918221044, 0.5440648351212303, -0.4559351648787697, -0.09403970877331563, -0.21942598713773656, -0.1954917778609545},
	{100, 110, 0.5, 0.05, 0.3, 0.01833469247686783, 0.41084208775394915, -0.5891579122460509, 0.5672153684783687, -0.2618380729278498, -0.22920762959451008},
	{100, 90, 0.25, 0.043, 0.65, 0.01072416711297237, 0.6983856582408442, -0.3016143417591558, -0.10443361505139563, 0.08964978098103314, 0.10763532879428928},
	{50000, 55000, 0.02, 0.043, 0.8, 5.209680955119214e-05, 0.21820758389418154, -0.7817924161058185, 0.3283732881543373, -6.67947390362181, -6.644517365648494},
	{50000, 45000, 0.08, 0.043, 0.7, 3.26564884141943e-05, 0.741675834233225, -0.25832416576677497, -0.20806968615142593, 0.8400934268219709, 0.8540703126752719},
	{37.5, 38, 0.001369863, 0.043, 0.5, 0.44891435479889413, 0.24093542529121142, -0.7590645747087885, 0.4497298812123625, -82.79957853912641, -82.76669441756263},
	{100, 100, 1.0, 0.05, 0.2, 0.018762017345846895, 0.6368306511756191, -0.3631693488243809, -0.28143026018770345, -0.06566706071046413, -0.0447218302581166},
	{100, 150, 0.5, 0.05, 0.4, 0.006834838356028662, 0.11434776263615731, -0.8856522373638427, 0.718452762067355, -0.32155529660708526, -0.2761333773174828},
	// Edge cases: S=0, T=0, sigma=0 all return 0
	{0, 100, 0.1, 0.05, 0.2, 0, 0, 0, 0, 0, 0},
	{100, 100, 0, 0.05, 0.2, 0, 0, 0, 0, 0, 0},
	{100, 100, 0.1, 0.05, 0, 0, 0, 0, 0, 0, 0},
}

const tolerance = 1e-10

func closeEnough(a, b, tol float64) bool {
	if a == 0 && b == 0 {
		return true
	}
	diff := math.Abs(a - b)
	mag := math.Max(math.Abs(a), math.Abs(b))
	if mag == 0 {
		return diff < tol
	}
	return diff/mag < tol
}

func TestGamma(t *testing.T) {
	for i, v := range vectors {
		got := Gamma(v.S, v.K, v.T, v.r, v.sigma)
		if !closeEnough(got, v.gamma, tolerance) {
			t.Errorf("case %d: Gamma(%v,%v,%v,%v,%v) = %v, want %v", i, v.S, v.K, v.T, v.r, v.sigma, got, v.gamma)
		}
	}
}

func TestDelta(t *testing.T) {
	for i, v := range vectors {
		gotCall := Delta(v.S, v.K, v.T, v.r, v.sigma, "call")
		if !closeEnough(gotCall, v.deltaCall, tolerance) {
			t.Errorf("case %d: Delta call = %v, want %v", i, gotCall, v.deltaCall)
		}
		gotPut := Delta(v.S, v.K, v.T, v.r, v.sigma, "put")
		if !closeEnough(gotPut, v.deltaPut, tolerance) {
			t.Errorf("case %d: Delta put = %v, want %v", i, gotPut, v.deltaPut)
		}
	}
}

func TestVanna(t *testing.T) {
	for i, v := range vectors {
		got := Vanna(v.S, v.K, v.T, v.r, v.sigma)
		if !closeEnough(got, v.vanna, tolerance) {
			t.Errorf("case %d: Vanna = %v, want %v", i, got, v.vanna)
		}
	}
}

func TestCharm(t *testing.T) {
	for i, v := range vectors {
		gotCall := Charm(v.S, v.K, v.T, v.r, v.sigma, "call")
		if !closeEnough(gotCall, v.charmCall, tolerance) {
			t.Errorf("case %d: Charm call = %v, want %v", i, gotCall, v.charmCall)
		}
		gotPut := Charm(v.S, v.K, v.T, v.r, v.sigma, "put")
		if !closeEnough(gotPut, v.charmPut, tolerance) {
			t.Errorf("case %d: Charm put = %v, want %v", i, gotPut, v.charmPut)
		}
	}
}
