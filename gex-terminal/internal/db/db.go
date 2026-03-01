package db

import (
	"database/sql"
	"fmt"
	"log/slog"
	"sync"

	"gex-terminal/internal/config"

	_ "modernc.org/sqlite"
)

var (
	pool *sql.DB
	mu   sync.Mutex
)

// Init opens the database and creates/migrates all tables.
func Init() error {
	mu.Lock()
	defer mu.Unlock()

	var err error
	pool, err = sql.Open("sqlite", config.DBPath+"?_busy_timeout=30000&_journal_mode=WAL")
	if err != nil {
		return fmt.Errorf("open db: %w", err)
	}
	pool.SetMaxOpenConns(1) // SQLite single-writer
	pool.SetMaxIdleConns(4)

	if _, err := pool.Exec("PRAGMA journal_mode=WAL"); err != nil {
		slog.Warn("PRAGMA WAL", "err", err)
	}
	if _, err := pool.Exec("PRAGMA busy_timeout=30000"); err != nil {
		slog.Warn("PRAGMA busy_timeout", "err", err)
	}

	return migrate()
}

// Get returns the shared database pool.
func Get() *sql.DB {
	return pool
}

// Close closes the database pool.
func Close() error {
	if pool != nil {
		return pool.Close()
	}
	return nil
}

func migrate() error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS snapshots (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			date TEXT NOT NULL, ticker TEXT NOT NULL,
			spot REAL, btc_price REAL, gamma_flip REAL,
			call_wall REAL, put_wall REAL, max_pain REAL,
			regime TEXT, net_gex REAL,
			total_call_oi INTEGER, total_put_oi INTEGER,
			UNIQUE(date, ticker)
		)`,
		`CREATE TABLE IF NOT EXISTS strike_history (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			date TEXT NOT NULL, ticker TEXT NOT NULL,
			strike REAL NOT NULL, call_oi INTEGER,
			put_oi INTEGER, total_oi INTEGER, net_gex REAL,
			UNIQUE(date, ticker, strike)
		)`,
		`CREATE TABLE IF NOT EXISTS data_cache (
			date TEXT NOT NULL, ticker TEXT NOT NULL,
			dte INTEGER NOT NULL, data_json TEXT NOT NULL,
			UNIQUE(date, ticker, dte)
		)`,
		`CREATE TABLE IF NOT EXISTS analysis_cache (
			date TEXT NOT NULL, ticker TEXT NOT NULL,
			analysis_json TEXT NOT NULL,
			UNIQUE(date, ticker)
		)`,
		`CREATE TABLE IF NOT EXISTS btc_candles (
			symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
			tf TEXT NOT NULL,
			time INTEGER NOT NULL,
			open REAL NOT NULL, high REAL NOT NULL,
			low REAL NOT NULL, close REAL NOT NULL,
			UNIQUE(symbol, tf, time)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_candles_sym_tf_time ON btc_candles(symbol, tf, time)`,
		`CREATE TABLE IF NOT EXISTS etf_flows (
			date TEXT NOT NULL, ticker TEXT NOT NULL,
			shares_outstanding REAL, aum REAL, nav REAL,
			daily_flow_shares REAL, daily_flow_dollars REAL,
			UNIQUE(date, ticker)
		)`,
		`CREATE TABLE IF NOT EXISTS predictions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			analysis_date TEXT NOT NULL,
			ticker TEXT NOT NULL,
			expiry_date TEXT NOT NULL,
			dte INTEGER NOT NULL,
			dte_window TEXT NOT NULL,
			spot_btc REAL,
			call_wall_btc REAL, put_wall_btc REAL,
			gamma_flip_btc REAL, max_pain_btc REAL,
			regime TEXT, net_gex REAL,
			net_dealer_delta REAL, dealer_delta_direction TEXT,
			charm_direction TEXT, charm_notional REAL,
			charm_strength TEXT, vanna_strength TEXT,
			overnight_direction TEXT, overnight_notional REAL,
			deribit_available INTEGER DEFAULT 0,
			ibit_call_wall_btc REAL, ibit_put_wall_btc REAL,
			deribit_call_wall_btc REAL, deribit_put_wall_btc REAL,
			venue_walls_agree INTEGER,
			em_upper_btc REAL, em_lower_btc REAL,
			ai_bottom_line TEXT,
			scored INTEGER DEFAULT 0, scored_date TEXT,
			btc_high_on_expiry REAL, btc_low_on_expiry REAL, btc_close_on_expiry REAL,
			btc_high_in_window REAL, btc_low_in_window REAL,
			call_wall_held INTEGER, put_wall_held INTEGER,
			range_held INTEGER, em_held INTEGER,
			regime_correct INTEGER, charm_correct INTEGER,
			venue_agree_held INTEGER,
			max_breach_call_pct REAL, max_breach_put_pct REAL,
			realized_range_pct REAL,
			call_wall_error_pct REAL, put_wall_error_pct REAL,
			UNIQUE(analysis_date, ticker, expiry_date)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_pred_expiry ON predictions(expiry_date, scored)`,
		`CREATE INDEX IF NOT EXISTS idx_pred_dte ON predictions(dte, scored)`,
		`CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(ticker, scored)`,
		`CREATE TABLE IF NOT EXISTS coinglass_data (
			date TEXT NOT NULL,
			symbol TEXT NOT NULL,
			metric TEXT NOT NULL,
			value REAL,
			extra_json TEXT,
			UNIQUE(date, symbol, metric)
		)`,
		`CREATE TABLE IF NOT EXISTS expiry_cache (
			date TEXT NOT NULL,
			ticker TEXT NOT NULL,
			expiry_date TEXT NOT NULL,
			data_json TEXT NOT NULL,
			UNIQUE(date, ticker, expiry_date)
		)`,
	}

	for _, s := range stmts {
		if _, err := pool.Exec(s); err != nil {
			return fmt.Errorf("migrate: %w\nSQL: %s", err, s)
		}
	}

	// Column migrations
	migrateAddColumn("snapshots", "weighted_net_gex", "REAL")
	migrateAddColumn("strike_history", "weighted_net_gex", "REAL")
	migrateAddColumn("etf_flows", "total_btc_etf_flow", "REAL")

	// T+2 scoring columns for predictions
	for _, col := range []struct{ name, typ string }{
		{"btc_high_t2", "REAL"}, {"btc_low_t2", "REAL"}, {"btc_close_t2", "REAL"},
		{"call_wall_held_t2", "INTEGER"}, {"put_wall_held_t2", "INTEGER"},
		{"range_held_t2", "INTEGER"}, {"regime_correct_t2", "INTEGER"},
	} {
		migrateAddColumn("predictions", col.name, col.typ)
	}

	// Migrate btc_candles if missing symbol column
	migrateBTCCandlesSymbol()

	return nil
}

func migrateAddColumn(table, column, colType string) {
	// Check if column exists
	rows, err := pool.Query(fmt.Sprintf("PRAGMA table_info(%s)", table))
	if err != nil {
		return
	}
	defer rows.Close()

	for rows.Next() {
		var cid int
		var name, typ string
		var notNull int
		var dfltValue *string
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notNull, &dfltValue, &pk); err != nil {
			continue
		}
		if name == column {
			return // column already exists
		}
	}

	_, err = pool.Exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", table, column, colType))
	if err != nil {
		slog.Warn("migrate add column", "table", table, "column", column, "err", err)
	}
}

func migrateBTCCandlesSymbol() {
	rows, err := pool.Query("PRAGMA table_info(btc_candles)")
	if err != nil {
		return
	}
	defer rows.Close()

	hasSymbol := false
	for rows.Next() {
		var cid int
		var name, typ string
		var notNull int
		var dfltValue *string
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notNull, &dfltValue, &pk); err != nil {
			continue
		}
		if name == "symbol" {
			hasSymbol = true
		}
	}

	if hasSymbol {
		return
	}

	slog.Info("migrating btc_candles: adding symbol column")
	tx, err := pool.Begin()
	if err != nil {
		return
	}
	defer tx.Rollback()

	stmts := []string{
		`CREATE TABLE btc_candles_new (
			symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
			tf TEXT NOT NULL, time INTEGER NOT NULL,
			open REAL NOT NULL, high REAL NOT NULL,
			low REAL NOT NULL, close REAL NOT NULL,
			UNIQUE(symbol, tf, time)
		)`,
		`INSERT INTO btc_candles_new (symbol, tf, time, open, high, low, close)
		 SELECT 'BTCUSDT', tf, time, open, high, low, close FROM btc_candles`,
		`DROP TABLE btc_candles`,
		`ALTER TABLE btc_candles_new RENAME TO btc_candles`,
	}
	for _, s := range stmts {
		if _, err := tx.Exec(s); err != nil {
			slog.Error("btc_candles migration", "err", err)
			return
		}
	}
	tx.Commit()
}
