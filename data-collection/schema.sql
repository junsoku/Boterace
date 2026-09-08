-- 競艇予想アプリ用 DBスキーマ (SQLite想定 / PostgreSQLへの移行も容易な設計)
-- 生JSONも別途保持し、パース漏れ・仕様変更時に再パースできるようにしている

-- 生データ保管(取得元API/日付/種別ごとにJSONをそのまま保存)
CREATE TABLE IF NOT EXISTS raw_json (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_date       TEXT NOT NULL,          -- 'YYYY-MM-DD'
    kind            TEXT NOT NULL,          -- 'programs' | 'previews' | 'results'
    fetched_at      TEXT NOT NULL,          -- 取得日時(ISO8601)
    payload         TEXT NOT NULL,          -- JSON文字列そのまま
    UNIQUE(race_date, kind)
);

-- レース場マスタ(24場)
CREATE TABLE IF NOT EXISTS stadiums (
    stadium_number  INTEGER PRIMARY KEY,    -- 1〜24
    stadium_name    TEXT NOT NULL
);

-- レース(1日1場1レース = 1行)
CREATE TABLE IF NOT EXISTS races (
    race_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    race_date           TEXT NOT NULL,      -- 'YYYY-MM-DD'
    stadium_number       INTEGER NOT NULL,
    race_number          INTEGER NOT NULL,  -- 1〜12R
    race_title           TEXT,              -- 大会名(該当する場合)
    race_grade            TEXT,             -- SG/G1/G2/G3/一般 など
    close_at              TEXT,             -- 締切予定時刻
    distance_m             INTEGER,         -- 通常1800m
    weather                TEXT,            -- 天候
    wind_direction          TEXT,
    wind_speed_m             REAL,
    wave_height_cm            REAL,
    temperature_c              REAL,
    water_temperature_c         REAL,
    UNIQUE(race_date, stadium_number, race_number)
);

-- 出走表:1レース6艇分のエントリ(選手・モーター・ボート情報)
CREATE TABLE IF NOT EXISTS entries (
    entry_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id              INTEGER NOT NULL REFERENCES races(race_id),
    boat_number            INTEGER NOT NULL,  -- 1〜6号艇
    racer_registration_number INTEGER,        -- 選手登録番号
    racer_name               TEXT,
    racer_branch              TEXT,           -- 支部
    racer_class                TEXT,          -- A1/A2/B1/B2
    racer_age                   INTEGER,
    racer_weight_kg               REAL,
    national_win_rate               REAL,     -- 全国勝率
    national_2連率                   REAL,     -- 全国2連率
    local_win_rate                    REAL,   -- 当地勝率
    local_2連率                        REAL,
    motor_number                        INTEGER,
    motor_2連率                          REAL,
    boat_hull_number                       INTEGER,
    boat_hull_2連率                         REAL,
    average_start_timing                     REAL,
    UNIQUE(race_id, boat_number)
);

-- 直前情報:展示タイム・スタート展示・体重調整など、締切直前に確定するデータ
CREATE TABLE IF NOT EXISTS previews (
    preview_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id               INTEGER NOT NULL REFERENCES entries(entry_id),
    exhibition_time           REAL,          -- 展示タイム
    tilt_angle                  REAL,        -- チルト角度
    weight_adjustment_kg          REAL,      -- 調整重量
    start_course                   INTEGER,  -- 展示での進入コース
    start_timing_preview             REAL,   -- 展示スタートタイミング
    parts_exchanged                    TEXT, -- 部品交換情報
    UNIQUE(entry_id)
);

-- 結果:着順・実際のスタート・払戻金の元となる実測値
CREATE TABLE IF NOT EXISTS results (
    result_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id                INTEGER NOT NULL REFERENCES entries(entry_id),
    arrival_order              INTEGER,      -- 着順(1〜6、失格等は別途コード化)
    actual_course                 INTEGER,   -- 実際の進入コース
    actual_start_timing              REAL,
    race_time                          REAL, -- 決まり手のタイム
    winning_technique                    TEXT, -- 決まり手(逃げ/差し/まくり等)
    remarks                                TEXT, -- 失格/転覆などの備考
    UNIQUE(entry_id)
);

-- 払戻金(3連単など)
CREATE TABLE IF NOT EXISTS payouts (
    payout_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id         INTEGER NOT NULL REFERENCES races(race_id),
    bet_type          TEXT NOT NULL,   -- '3連単' '2連単' '拡連複' 等
    combination         TEXT NOT NULL, -- '1-2-3' のような組み合わせ文字列
    payout_yen            INTEGER,
    popularity_rank          INTEGER  -- 人気順位(オッズ人気)
);

CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date);
CREATE INDEX IF NOT EXISTS idx_entries_race ON entries(race_id);
CREATE INDEX IF NOT EXISTS idx_results_entry ON results(entry_id);
