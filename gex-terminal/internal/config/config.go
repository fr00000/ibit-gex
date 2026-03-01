package config

import (
	"os"
	"path/filepath"

	"github.com/joho/godotenv"
)

// DTEWindow defines a non-overlapping DTE window for options positioning.
type DTEWindow struct {
	Label  int // display label (3, 7, 14, 30, 45)
	MinDTE int // inclusive lower bound
	MaxDTE int // inclusive upper bound
}

// DTEWindows are the 5 non-overlapping windows used throughout the system.
var DTEWindows = []DTEWindow{
	{3, 0, 3},   // This week's expirations — immediate hedging
	{7, 4, 7},   // Next week's setup
	{14, 8, 14}, // Two weeks out
	{30, 15, 30}, // Monthly cycle positioning
	{45, 31, 45}, // Structural / quarterly
}

// TickerCfg holds per-ticker configuration.
type TickerCfg struct {
	Name            string
	AssetLabel      string // "BTC" or "ETH"
	RefTicker       string // e.g. "BTC-USD"
	BinanceSymbol   string // e.g. "BTCUSDT"
	PerShareDefault float64
	DeribitCurrency string // "BTC" or "ETH"
}

// TickerConfig maps ticker symbol to its configuration.
var TickerConfig = map[string]*TickerCfg{
	"IBIT": {
		Name:            "IBIT",
		AssetLabel:      "BTC",
		RefTicker:       "BTC-USD",
		BinanceSymbol:   "BTCUSDT",
		PerShareDefault: 0.000568,
		DeribitCurrency: "BTC",
	},
	"ETHA": {
		Name:            "ETHA",
		AssetLabel:      "ETH",
		RefTicker:       "ETH-USD",
		BinanceSymbol:   "ETHUSDT",
		PerShareDefault: 0.0091,
		DeribitCurrency: "ETH",
	},
}

// Constants matching Python config.
const (
	RiskFreeRateDefault = 0.043
	StrikeRangePct      = 0.35

	DeribitCacheSeconds = 900  // 15 minutes
	FarsideCacheSeconds = 3600 // 1 hour
	YahooCheckInterval  = 1800 // 30 minutes
)

var (
	DBPath          string
	AnthropicAPIKey string
	CoinglassAPIKey string
	AdminKey        string
	ServerPort      string
	ServerHost      string
)

// Load reads .env and sets package-level config vars.
func Load() {
	// Try loading .env from the working directory and project root.
	_ = godotenv.Load()
	_ = godotenv.Load(filepath.Join("..", ".env"))

	home, _ := os.UserHomeDir()
	DBPath = filepath.Join(home, ".ibit_gex_history.db")

	AnthropicAPIKey = os.Getenv("ANTHROPIC_API_KEY")
	CoinglassAPIKey = os.Getenv("COINGLASS_API_KEY")
	AdminKey = os.Getenv("ADMIN_KEY")
	ServerPort = envOrDefault("PORT", "5001")
	ServerHost = envOrDefault("HOST", "127.0.0.1")
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// WindowLabel returns the string label like "0-3d" for a DTE window.
func (w DTEWindow) WindowLabel() string {
	return windowLabel(w.MinDTE, w.MaxDTE)
}

func windowLabel(minDTE, maxDTE int) string {
	switch {
	case minDTE == 0 && maxDTE == 3:
		return "0-3d"
	case minDTE == 4 && maxDTE == 7:
		return "4-7d"
	case minDTE == 8 && maxDTE == 14:
		return "8-14d"
	case minDTE == 15 && maxDTE == 30:
		return "15-30d"
	case minDTE == 31 && maxDTE == 45:
		return "31-45d"
	default:
		return ""
	}
}
