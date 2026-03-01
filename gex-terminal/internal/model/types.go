package model

// Levels holds the computed GEX levels for a strike set.
type Levels struct {
	CallWall              float64   `json:"call_wall"`
	PutWall               float64   `json:"put_wall"`
	GammaFlip             float64   `json:"gamma_flip"`
	MaxPain               float64   `json:"max_pain"`
	Regime                string    `json:"regime"` // "positive_gamma" or "negative_gamma"
	NetGEXTotal           float64   `json:"net_gex_total"`
	ActiveGEXTotal        float64   `json:"active_gex_total"`
	NetDealerDelta        float64   `json:"net_dealer_delta"`
	NetVanna              float64   `json:"net_vanna"`
	NetCharm              float64   `json:"net_charm"`
	TotalCallOI           int64     `json:"total_call_oi"`
	TotalPutOI            int64     `json:"total_put_oi"`
	PCR                   float64   `json:"pcr"`
	VolumeGEXTotal        float64   `json:"volume_gex_total"`
	GEXActivityRatio      float64   `json:"gex_activity_ratio"`
	ActiveCallWall        float64   `json:"active_call_wall"`
	ActivePutWall         float64   `json:"active_put_wall"`
	PositioningConfidence int       `json:"positioning_confidence"`
	PositioningWarnings   []string  `json:"positioning_warnings"`
	Resistance            []float64 `json:"resistance"`
	Support               []float64 `json:"support"`
	LevelTrajectory       map[string]*LevelTrajectoryEntry `json:"level_trajectory,omitempty"`
}

// LevelTrajectoryEntry tracks stability of a level vs yesterday.
type LevelTrajectoryEntry struct {
	Direction     string `json:"direction"`      // "up", "down", "stable"
	ChangePct     float64 `json:"change_pct"`
	DominantExpiry string `json:"dominant_expiry,omitempty"`
}

// StrikeData holds per-strike aggregated data.
type StrikeData struct {
	Strike         float64 `json:"strike"`
	BTC            float64 `json:"btc"`
	CallOI         int64   `json:"call_oi"`
	PutOI          int64   `json:"put_oi"`
	TotalOI        int64   `json:"total_oi"`
	CallGEX        float64 `json:"call_gex"`
	PutGEX         float64 `json:"put_gex"`
	NetGEX         float64 `json:"net_gex"`
	IBITGEX        float64 `json:"ibit_gex"`
	DeribitGEX     float64 `json:"deribit_gex"`
	ActiveGEX      float64 `json:"active_gex"`
	NetDealerDelta float64 `json:"net_dealer_delta"`
	NetVanna       float64 `json:"net_vanna"`
	NetCharm       float64 `json:"net_charm"`
	CallVanna      float64 `json:"call_vanna"`
	PutVanna       float64 `json:"put_vanna"`
	CallCharm      float64 `json:"call_charm"`
	PutCharm       float64 `json:"put_charm"`
	CallVolume     int64   `json:"call_volume"`
	PutVolume      int64   `json:"put_volume"`
	TotalVolume    int64   `json:"total_volume"`
	CallVolGEX     float64 `json:"call_vol_gex"`
	PutVolGEX      float64 `json:"put_vol_gex"`
	ExpiryBreakdown map[string]float64 `json:"expiry_breakdown,omitempty"`
}

// ExpectedMove holds straddle-derived expected move.
type ExpectedMove struct {
	UpperBTC  float64 `json:"upper_btc"`
	LowerBTC  float64 `json:"lower_btc"`
	UpperIBIT float64 `json:"upper_ibit"`
	LowerIBIT float64 `json:"lower_ibit"`
	Pct       float64 `json:"pct"`
	DTE       float64 `json:"dte,omitempty"`
}

// DeltaFlipPoint marks where dealer delta pressure reverses.
type DeltaFlipPoint struct {
	PriceBTC     float64 `json:"price_btc"`
	PriceIBIT    float64 `json:"price_ibit"`
	DistancePct  float64 `json:"distance_pct,omitempty"`
	FromDirection string `json:"from_direction"`
	ToDirection   string `json:"to_direction"`
}

// SignificantLevel is a high-OI strike with regime-adjusted behavior.
type SignificantLevel struct {
	Strike    float64 `json:"strike"`
	BTC       float64 `json:"btc"`
	StrikeBTC float64 `json:"strike_btc"`
	Type      string  `json:"type"` // "put_wall", "call_wall", "oi_magnet"
	CallOI    int64   `json:"call_oi"`
	PutOI     int64   `json:"put_oi"`
	TotalOI   int64   `json:"total_oi"`
	NetGEX    float64 `json:"net_gex"`
	NetVanna  float64 `json:"net_vanna"`
	NetCharm  float64 `json:"net_charm"`
	DistPct   float64 `json:"dist_pct"`
	IsMajor   bool    `json:"is_major"`
	Behavior  string  `json:"behavior"`
	GreeksNote *string `json:"greeks_note"` // nil = omit
	OIDelta   *OIDelta `json:"oi_delta,omitempty"`
}

// OIDelta tracks change vs yesterday.
type OIDelta struct {
	Delta  int64   `json:"delta"`
	Pct    float64 `json:"pct"`
	Status string  `json:"status"` // "BUILDING", "DECAYING", "stable"
}

// FlowForecast holds charm/vanna flow predictions.
type FlowForecast struct {
	Charm     *FlowComponent `json:"charm"`
	Vanna     *VannaForecast `json:"vanna"`
	Overnight *FlowComponent `json:"overnight"`
	RegimeNote string        `json:"regime_note"`
}

// FlowComponent is a single directional flow estimate.
type FlowComponent struct {
	Direction   string  `json:"direction"` // "buy" or "sell"
	NetNotional float64 `json:"net_notional"`
	Strength    string  `json:"strength"` // "negligible", "minor", "moderate", "significant"
	Narrative   string  `json:"narrative,omitempty"`
}

// VannaForecast includes vanna scenarios.
type VannaForecast struct {
	NetNotional    float64         `json:"net_notional"`
	Strength       string          `json:"strength"`
	CrushScenario  *FlowComponent  `json:"crush_scenario"`
	SpikeScenario  *FlowComponent  `json:"spike_scenario"`
}

// BreakoutAnalysis holds bullish/bearish signal aggregation.
type BreakoutAnalysis struct {
	UpSignals   []string        `json:"up_signals"`
	DownSignals []string        `json:"down_signals"`
	UpScore     int             `json:"up_score"`
	DownScore   int             `json:"down_score"`
	UpTargets   []BreakoutTarget `json:"up_targets"`
	DownTargets []BreakoutTarget `json:"down_targets"`
	Bias        string          `json:"bias"` // "upside", "downside", "balanced"
	EMNote      *string         `json:"em_note"`
}

// BreakoutTarget is a GEX magnet beyond a wall.
type BreakoutTarget struct {
	Strike  float64 `json:"strike"`
	BTC     float64 `json:"btc"`
	TotalOI int64   `json:"total_oi"`
	NetGEX  float64 `json:"net_gex"`
}

// DealerDeltaProfile holds repriced dealer delta across a price grid.
type DealerDeltaProfile struct {
	PriceIBIT        float64 `json:"price_ibit"`
	PriceBTC         float64 `json:"price_btc"`
	NetDealerDelta   float64 `json:"net_dealer_delta"`
	CallDealerDelta  float64 `json:"call_dealer_delta"`
	PutDealerDelta   float64 `json:"put_dealer_delta"`
	Acceleration     float64 `json:"acceleration,omitempty"`
	AccelClass       string  `json:"accel_class,omitempty"`
}

// DealerDeltaBriefing is a plain-English summary of dealer delta.
type DealerDeltaBriefing struct {
	CurrentDelta     string             `json:"current_delta"`
	Direction        string             `json:"direction"`
	Magnitude        string             `json:"magnitude"`
	FlipSummary      string             `json:"flip_summary"`
	AccelerationZone string             `json:"acceleration_zone"`
	RangeBias        string             `json:"range_bias,omitempty"`
	MorningTake      string             `json:"morning_take,omitempty"`
	KeyLevelDeltas   map[string]float64 `json:"key_level_deltas"`
}

// DealerDeltaScenarios wraps the full dealer delta analysis.
type DealerDeltaScenarios struct {
	Profile              []DealerDeltaProfile `json:"profile"`
	FlipPoints           []DeltaFlipPoint     `json:"flip_points"`
	MaxAccelerationPrice float64              `json:"max_acceleration_price"`
	KeyLevelDeltas       map[string]float64   `json:"key_level_deltas"`
}

// ETFFlows holds parsed ETF fund flow data.
type ETFFlows struct {
	DailyFlowShares    float64 `json:"daily_flow_shares"`
	DailyFlowDollars   float64 `json:"daily_flow_dollars"`
	TotalBTCETFFlow    float64 `json:"total_btc_etf_flow"`
	SharesOutstanding  *float64 `json:"shares_outstanding"`
	AUM                *float64 `json:"aum"`
	NAV                *float64 `json:"nav"`
	Direction          string  `json:"direction"` // "inflow" or "outflow"
	Strength           string  `json:"strength"`  // "negligible", "minor", "moderate", "strong"
	Streak             int     `json:"streak"`
	AvgFlow5D          float64 `json:"avg_flow_5d"`
}

// OIChanges tracks day-over-day open interest changes.
type OIChanges struct {
	CallOIChange int64   `json:"call_oi_change"`
	PutOIChange  int64   `json:"put_oi_change"`
	NetGEXChange float64 `json:"net_gex_change"`
}

// IVTermEntry is one point in the IV term structure.
type IVTermEntry struct {
	Expiry string  `json:"expiry"`
	DTE    int     `json:"dte"`
	ATMIV  float64 `json:"atm_iv"`
	CallIV float64 `json:"call_iv,omitempty"`
	PutIV  float64 `json:"put_iv,omitempty"`
	Source string  `json:"source"` // "ibit" or "deribit"
}

// DataFreshness tracks how stale each data source is.
type DataFreshness struct {
	IBIT    *SourceFreshness `json:"ibit,omitempty"`
	Deribit *SourceFreshness `json:"deribit,omitempty"`
}

// SourceFreshness for a single data source.
type SourceFreshness struct {
	AgeHours      float64 `json:"age_hours,omitempty"`
	AgeMinutes    float64 `json:"age_minutes,omitempty"`
	LastUpdate    string  `json:"last_update,omitempty"`
	InMarketHours bool    `json:"in_market_hours,omitempty"`
	CachedDate    string  `json:"cached_date,omitempty"`
}

// VenueBreakdown holds per-venue positioning data.
type VenueBreakdown struct {
	IBIT    *VenueData `json:"ibit,omitempty"`
	Deribit *VenueData `json:"deribit,omitempty"`
}

// VenueData holds venue-specific levels.
type VenueData struct {
	OIContracts  int64   `json:"oi_contracts,omitempty"`
	OIBTC        float64 `json:"oi_btc,omitempty"`
	NetGEX       float64 `json:"net_gex"`
	CallWallBTC  float64 `json:"call_wall_btc"`
	PutWallBTC   float64 `json:"put_wall_btc"`
	GammaFlipBTC float64 `json:"gamma_flip_btc"`
	Regime       string  `json:"regime"`
	NetVanna     float64 `json:"net_vanna"`
	NetCharm     float64 `json:"net_charm"`
}

// ChangesVsPrev holds comparison data vs previous session.
type ChangesVsPrev struct {
	PrevDate           string  `json:"prev_date"`
	SpotBTCPrev        float64 `json:"spot_btc_prev"`
	SpotBTCChange      float64 `json:"spot_btc_change"`
	RegimePrev         string  `json:"regime_prev"`
	RegimeChanged      bool    `json:"regime_changed"`
	CallWallBTCPrev    float64 `json:"call_wall_btc_prev"`
	CallWallBTCChange  float64 `json:"call_wall_btc_change"`
	PutWallBTCPrev     float64 `json:"put_wall_btc_prev"`
	PutWallBTCChange   float64 `json:"put_wall_btc_change"`
	GammaFlipBTCPrev   float64 `json:"gamma_flip_btc_prev"`
	GammaFlipBTCChange float64 `json:"gamma_flip_btc_change"`
	NetGEXPrev         float64 `json:"net_gex_prev"`
	NetGEXChange       float64 `json:"net_gex_change"`
	PCRPrev            float64 `json:"pcr_prev"`
	PCRChange          float64 `json:"pcr_change"`
}

// DetectedPattern is a structural pattern near key levels.
type DetectedPattern struct {
	Pattern    string  `json:"pattern"`
	Signal     string  `json:"signal"`
	DistancePct float64 `json:"distance_pct,omitempty"`
	PinnedTo   string  `json:"pinned_to,omitempty"`
}

// DataBlob is the comprehensive data structure returned by fetch_and_analyze.
// This is the primary cache unit stored in data_cache.
type DataBlob struct {
	Ticker       string  `json:"ticker"`
	AssetLabel   string  `json:"asset_label"`
	Date         string  `json:"date,omitempty"`
	SnapshotDate string  `json:"snapshot_date,omitempty"`
	DTEKey       int     `json:"dte_key"`
	Spot         float64 `json:"spot"`
	BTCSpot      float64 `json:"btc_spot"`
	BTCPerShare  float64 `json:"btc_per_share"`
	IsBTC        bool    `json:"is_btc"`
	Timestamp    string  `json:"timestamp"`

	DTEWindow struct {
		Min int `json:"min"`
		Max int `json:"max"`
	} `json:"dte_window"`

	Levels              *Levels              `json:"levels"`
	CombinedLevelsBTC   *Levels              `json:"combined_levels_btc"`
	DeribitLevelsBTC    *Levels              `json:"deribit_levels_btc"`
	ExpectedMove        *ExpectedMove        `json:"expected_move"`
	GEXChart            []StrikeData         `json:"gex_chart"`
	SignificantLevels   []SignificantLevel   `json:"significant_levels"`
	Breakout            *BreakoutAnalysis    `json:"breakout"`
	ETFFlows            *ETFFlows            `json:"etf_flows"`
	OIChanges           *OIChanges           `json:"oi_changes"`
	FlowForecast        *FlowForecast        `json:"flow_forecast"`
	DealerDeltaProfile  []DealerDeltaProfile `json:"dealer_delta_profile"`
	DealerDeltaBriefing *DealerDeltaBriefing `json:"dealer_delta_briefing"`
	DeltaFlipPoints     []DeltaFlipPoint     `json:"delta_flip_points"`
	IVTermStructure     []IVTermEntry        `json:"iv_term_structure"`
	DataFreshness       *DataFreshness       `json:"data_freshness"`

	Expirations   []string     `json:"expirations"`
	DeribitAvail  bool         `json:"deribit_available"`
	DeribitOIBTC  float64      `json:"deribit_oi_btc"`
	DeribitNetGEX float64      `json:"deribit_net_gex"`

	Stale        bool    `json:"stale,omitempty"`
	FallbackFrom *string `json:"fallback_from,omitempty"`
}

// Candle is an OHLCV price bar.
type Candle struct {
	Time  int64   `json:"time"`
	Open  float64 `json:"open"`
	High  float64 `json:"high"`
	Low   float64 `json:"low"`
	Close float64 `json:"close"`
}

// FlowRow is a single ETF flow history entry.
type FlowRow struct {
	Date      string  `json:"date"`
	Flow      float64 `json:"flow"`
	Shares    float64 `json:"shares"`
	AUM       float64 `json:"aum"`
	NAV       float64 `json:"nav"`
	TotalFlow float64 `json:"total_flow"`
}

// DeribitOption is a single parsed option from the Deribit API.
type DeribitOption struct {
	Strike         float64
	ExpiryStr      string
	ExpiryDate     int64 // unix timestamp
	DTE            float64
	OptionType     string // "call" or "put"
	OI             float64
	IV             float64 // percentage (65.0 = 65%)
	UnderlyingPrice float64
}

// MacroRegime holds the full macro scoring result.
type MacroRegime struct {
	Score           int                       `json:"score"`
	Bias            string                    `json:"bias"`
	SwingSignal     string                    `json:"swing_signal"`
	HighConviction  bool                      `json:"high_conviction"`
	CoinglassAvail  bool                      `json:"coinglass_available"`
	StructuralEntry *float64                  `json:"structural_entry"`
	Invalidation    *float64                  `json:"invalidation"`
	Signals         map[string]*MacroSignal   `json:"signals"`
	History         map[string]interface{}     `json:"history"`
	Timestamp       string                    `json:"timestamp,omitempty"`
	DataFreshness   *DataFreshness            `json:"data_freshness,omitempty"`
}

// MacroSignal is one component of the macro regime score.
type MacroSignal struct {
	Score  int    `json:"score"`
	Max    int    `json:"max"`
	Detail string `json:"detail"`
	Source string `json:"source,omitempty"`
}

// AnalysisData is the JSON blob sent to the AI for analysis.
type AnalysisData struct {
	HistoryTrends    interface{}           `json:"_history_trends"`
	StructureTrends  interface{}           `json:"_structure_trends"`
	MacroRegime      *MacroRegime          `json:"_macro_regime"`
	PredictionAccuracy interface{}         `json:"_prediction_accuracy"`
	Windows          map[string]interface{} `json:"-"` // Keyed by "0-3d" etc, merged at top level
}

// AnalysisResult is the AI analysis output.
type AnalysisResult struct {
	Window03  string  `json:"0-3d"`
	Window47  string  `json:"4-7d"`
	Window814 string  `json:"8-14d"`
	Window1530 string `json:"15-30d"`
	Window3145 string `json:"31-45d"`
	All       string  `json:"all"`
	BTCPrice  float64 `json:"_btc_price,omitempty"`
	Timestamp string  `json:"_timestamp,omitempty"`
}

// Snapshot is a daily level snapshot stored in the snapshots table.
type Snapshot struct {
	Date           string
	Ticker         string
	Spot           float64
	BTCPrice       float64
	GammaFlip      float64
	CallWall       float64
	PutWall        float64
	MaxPain        float64
	Regime         string
	NetGEX         float64
	TotalCallOI    int64
	TotalPutOI     int64
	WeightedNetGEX float64
}

// StrikeHistory is a per-strike OI record for day-over-day comparison.
type StrikeHistory struct {
	Strike         float64
	CallOI         int64
	PutOI          int64
	TotalOI        int64
	NetGEX         float64
	WeightedNetGEX float64
}

// Prediction is a saved level prediction for accuracy scoring.
type Prediction struct {
	AnalysisDate    string
	Ticker          string
	ExpiryDate      string
	DTE             int
	DTEWindow       string
	SpotBTC         float64
	CallWallBTC     float64
	PutWallBTC      float64
	GammaFlipBTC    float64
	MaxPainBTC      float64
	Regime          string
	NetGEX          float64
	NetDealerDelta  float64
	DealerDeltaDir  string
	CharmDirection  string
	CharmNotional   float64
	CharmStrength   string
	VannaStrength   string
	OvernightDir    string
	OvernightNotional float64
	DeribitAvailable bool
	IBITCallWallBTC  float64
	IBITPutWallBTC   float64
	DeribitCallWallBTC float64
	DeribitPutWallBTC  float64
	VenueWallsAgree  bool
	EMUpperBTC       float64
	EMLowerBTC       float64
	AIBottomLine     string
}

// CoinglassRow is a single metric stored in the coinglass_data table.
type CoinglassRow struct {
	Date      string
	Symbol    string
	Metric    string
	Value     float64
	ExtraJSON string
}

// OutlookWindow is one window's data for the /api/outlook response.
type OutlookWindow struct {
	Label          string   `json:"label"`
	MinDTE         int      `json:"min_dte"`
	MaxDTE         int      `json:"max_dte"`
	CallWall       float64  `json:"call_wall"`
	PutWall        float64  `json:"put_wall"`
	GammaFlip      float64  `json:"gamma_flip"`
	Regime         string   `json:"regime"`
	NetGEX         float64  `json:"net_gex"`
	EMUpper        float64  `json:"em_upper"`
	EMLower        float64  `json:"em_lower"`
	VenueAgree     bool     `json:"venue_agree"`
	IBITCallWall   float64  `json:"ibit_cw"`
	IBITPutWall    float64  `json:"ibit_pw"`
	DeribitCallWall float64 `json:"deribit_cw"`
	DeribitPutWall  float64 `json:"deribit_pw"`
	DealerDeltaDir *string  `json:"dealer_delta_dir"`
	CharmDir       *string  `json:"charm_dir"`
	DeltaFlipBTC   *float64 `json:"delta_flip_btc"`
}

// RangeConeWindow is one window for the /api/range-cone response.
type RangeConeWindow struct {
	Label     string          `json:"label"`
	MinDTE    int             `json:"min_dte"`
	MaxDTE    int             `json:"max_dte"`
	EMUpper   float64         `json:"em_upper"`
	EMLower   float64         `json:"em_lower"`
	GEXZones  []GEXZone       `json:"gex_zones"`
	DeltaFlip *float64        `json:"delta_flip"`
	GammaFlip float64         `json:"gamma_flip"`
	Regime    string          `json:"regime"`
}

// GEXZone is a strike with GEX magnitude for the range cone.
type GEXZone struct {
	BTC float64 `json:"btc"`
	GEX float64 `json:"gex"`
}
