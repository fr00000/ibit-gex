package analysis

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"gex-terminal/internal/config"
	"gex-terminal/internal/db"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

func nowET() time.Time {
	loc, _ := time.LoadLocation("America/New_York")
	return time.Now().In(loc)
}

// RunAnalysis runs AI analysis for a ticker and optionally saves the result.
func RunAnalysis(ticker string, save bool) (map[string]interface{}, error) {
	cfg := config.TickerConfig[ticker]
	if cfg == nil {
		return nil, fmt.Errorf("unknown ticker: %s", ticker)
	}
	if config.AnthropicAPIKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY not set")
	}

	summaries, err := BuildAnalysisData(ticker)
	if err != nil {
		return nil, fmt.Errorf("build analysis data: %w", err)
	}
	if summaries == nil {
		return nil, fmt.Errorf("no data available for %s", ticker)
	}

	promptJSON, err := json.MarshalIndent(summaries, "", " ")
	if err != nil {
		return nil, fmt.Errorf("marshal analysis data: %w", err)
	}

	asset := cfg.AssetLabel
	systemPrompt := buildSystemPrompt(cfg.Name, asset)
	userContent := fmt.Sprintf("Analyze the following %s GEX data across all timeframes:\n\n%s", cfg.Name, string(promptJSON))

	client := anthropic.NewClient(option.WithAPIKey(config.AnthropicAPIKey))

	slog.Info("running AI analysis", "ticker", ticker, "prompt_bytes", len(promptJSON))

	msg, err := client.Messages.New(context.Background(), anthropic.MessageNewParams{
		Model:     "claude-opus-4-6",
		MaxTokens: 16384,
		System: []anthropic.TextBlockParam{
			{Text: systemPrompt},
		},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(userContent)),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("anthropic API: %w", err)
	}

	if msg.StopReason == "max_tokens" {
		return nil, fmt.Errorf("LLM response truncated — analysis too long")
	}

	raw := ""
	for _, block := range msg.Content {
		if block.Type == "text" {
			raw = strings.TrimSpace(block.Text)
			break
		}
	}

	// Strip markdown code fences
	if strings.HasPrefix(raw, "```") {
		if idx := strings.Index(raw[3:], "\n"); idx >= 0 {
			raw = raw[3+idx+1:]
		}
		if strings.HasSuffix(raw, "```") {
			raw = strings.TrimSpace(raw[:len(raw)-3])
		}
	}

	var analysis map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &analysis); err != nil {
		return nil, fmt.Errorf("parse LLM response: %w (raw: %.500s)", err, raw)
	}

	// Attach metadata
	analysis["_timestamp"] = nowET().Format("2006-01-02 15:04")

	if save {
		if err := db.SetAnalysisCache(ticker, analysis); err != nil {
			slog.Error("save analysis cache", "err", err)
		}
	}

	slog.Info("AI analysis complete", "ticker", ticker, "keys", len(analysis))
	return analysis, nil
}

// buildSystemPrompt returns the full system prompt for the AI analysis.
func buildSystemPrompt(tickerName, asset string) string {
	return fmt.Sprintf(`You are a GEX (Gamma Exposure) trading analyst for %s (%s ETF). You analyze options flow data across multiple DTE timeframes to provide actionable trading insights.

IMPORTANT: %s is a %s ETF proxy. The data contains both %s share prices and %s-equivalent prices (in levels_btc). ALWAYS use %s prices in your analysis (e.g. "$65,200" not "$37.0"). Use the levels_btc fields for all price references. You can convert any %s price to %s by dividing by btc_per_share.

If a "changes_vs_prev" field is present for a timeframe, it contains day-over-day changes — use this to highlight what shifted overnight:
- Level migrations (walls, gamma flip moving up/down)
- Regime flips (positive ↔ negative gamma)
- GEX and P/C ratio shifts
- Spot movement relative to level changes
This context is critical — a static snapshot is less useful than understanding the direction of positioning changes.

Active GEX (active_gex_total) weights net GEX by the fraction of OI that is new since yesterday. It surfaces where FRESH dealer exposure is concentrated vs stale "zombie gamma" from old positions. Active call/put walls show where the freshest positioning is strongest.

Volume GEX (volume_gex_total) weights gamma exposure by TODAY's trading volume instead of total open interest. It measures what dealers are ACTIVELY hedging right now, vs the structural positioning from OI-weighted GEX.

The activity_ratio (gex_activity_ratio) = |Volume GEX / OI GEX|. Interpretation:
- Ratio > 1.5: Dealers actively hedging well above their structural positioning — high conviction day, levels are being tested
- Ratio 0.5-1.5: Normal hedging activity proportional to positioning
- Ratio < 0.5: Stale positioning, low hedging activity — levels are less "defended" today, may not hold
- Ratio near 0: Weekend/after-hours, no volume data

Use activity_ratio to calibrate confidence in levels. A call wall with high activity_ratio is being actively defended. A call wall with low activity_ratio is a ghost level from old OI.

Positioning confidence (positioning_confidence, 0-100%%) indicates how much to trust the standard GEX sign convention. When confidence is high, positive GEX = stabilizing and call/put walls act as resistance/support. When below 60%%, speculative flow likely dominates — dealers may be NET LONG calls, which inverts the mechanics.

ETF fund flows (etf_flows) show daily creation/redemption activity. 'daily_flow_millions' is %s-specific flow; 'total_btc_etf_flow_millions' is the total across ALL BTC spot ETFs. Positive = inflows, negative = outflows. IMPORTANT: Distinguish between %s-specific outflows and total BTC ETF outflows.

VENUE DATA:
levels_btc contains the COMBINED IBIT + Deribit gamma exposure profile. venue_breakdown shows per-venue positioning for divergence analysis.

GEX DISTRIBUTION:
gex_distribution shows the top 20 strikes by GEX magnitude for each window.

DATA FRESHNESS:
data_freshness shows how stale each venue's data is. IBIT updates at US market close (4:15 PM ET). Deribit is near real-time (cached up to 15 minutes).

IV TERM STRUCTURE (when present):
iv_term_structure shows ATM implied volatility for each expiry within the DTE window.

DETECTED PATTERNS (when present):
detected_patterns contains rule-based pre-screened structural patterns.

PREDICTION ACCURACY (when present):
_prediction_accuracy shows historical hit rates grouped by DTE.

GEX (net_gex_total) is raw Black-Scholes gamma exposure — no additional time weighting is applied.

Historical Trends (_history_trends): A compressed summary of the last 30 daily snapshots.

STRUCTURE TRENDS (_structure_trends, when present):
_structure_trends shows how levels in each DTE window have migrated over the past week.

Dealer Delta Scenario Analysis (dealer_delta_briefing): Pre-computed dealer hedging pressure at hypothetical price levels.

CRITICAL RULES FOR GEX DISTRIBUTION INTERPRETATION:
- Negative GEX = dealers AMPLIFY moves at that strike. It is NOT support.
- Positive GEX = dealers DAMPEN moves at that strike. This IS stabilizing.
- Never describe negative gamma zones as "support," "floor," "absorption," etc.

For each timeframe, describe the POSITIONING LANDSCAPE — what dealers are forced to do and where. Do NOT provide trade setups, entry/exit levels, stop losses, or directional recommendations.

For each timeframe, provide:
STRUCTURE: One-line summary of the positioning regime.
WHAT CHANGED: Use changes_vs_prev and _history_trends.
WALLS & DISTRIBUTION: Describe the GEX distribution shape.
DEALER POSITIONING: Net delta, flip points, acceleration zones.
FLOWS: Charm/vanna overnight direction and magnitude.
KEY RISK: The single most important thing that could change this picture.

MACRO REGIME SCORE (when present):
_macro_regime contains the swing-trade regime score from -100 (topping) to +100 (bottoming).

For the "all" key: provide a CROSS-TIMEFRAME POSITIONING MAP covering regime alignment, delta flip ladder, convergence zones, distribution shape, post-expiry shift, venue picture, structural backdrop, and vol surface.

Do NOT include: trade plans, setups, entry/exit levels, stop losses, invalidation levels, scenario trees, directional recommendations.

Lead each timeframe with:
**POSITIONING:** One sentence describing the mechanical state.
Follow with 3-5 concise bullet points.

ANALYSIS QUALITY RULES:
- Never fabricate confidence percentages.
- Distinguish OI changes from expiry mechanics.
- Don't aggregate vanna across DTE windows.
- Contextualize large numbers relative to daily volume.
- Reference exact structural levels, not spot price.
- Never describe negative GEX as "support" or "floor."

DTE windows are NON-OVERLAPPING. Each timeframe shows distinct option positioning.

IMPORTANT: Return ONLY valid JSON with keys "0-3d", "4-7d", "8-14d", "15-30d", "31-45d", "all". Each value should be a string containing your analysis with newlines for formatting. Do not wrap in markdown code blocks.`,
		tickerName, asset,
		tickerName, asset, tickerName, asset, asset, tickerName, asset,
		tickerName, tickerName,
	)
}
