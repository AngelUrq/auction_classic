CREATE TABLE IF NOT EXISTS items (
    item_id             INTEGER PRIMARY KEY,
    name                TEXT NOT NULL,
    quality             TEXT,
    item_level          SMALLINT,
    required_level      SMALLINT,
    item_class          TEXT,
    item_subclass       TEXT,
    inventory_type      TEXT,
    purchase_price      NUMERIC(16, 4),
    sell_price          NUMERIC(16, 4),
    max_count           INTEGER,
    purchase_quantity   INTEGER,
    is_equippable       BOOLEAN,
    is_stackable        BOOLEAN,
    details             JSONB
);

CREATE TABLE IF NOT EXISTS auctions (
    auction_id      BIGINT PRIMARY KEY,
    item_id         INTEGER,
    context         SMALLINT,
    buyout          NUMERIC(16, 4),
    bid             NUMERIC(16, 4),
    quantity        SMALLINT NOT NULL,
    first_seen_ts   TIMESTAMPTZ,
    last_seen_ts    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS auction_observations (
    auction_id  BIGINT NOT NULL REFERENCES auctions (auction_id),
    snapshot_ts TIMESTAMPTZ NOT NULL,
    time_left   TEXT NOT NULL,
    PRIMARY KEY (auction_id, snapshot_ts)
);

CREATE TABLE IF NOT EXISTS auction_bonus (
    auction_id  BIGINT NOT NULL REFERENCES auctions (auction_id),
    bonus_id    INTEGER NOT NULL,
    PRIMARY KEY (auction_id, bonus_id)
);

CREATE TABLE IF NOT EXISTS auction_modifiers (
    auction_id      BIGINT NOT NULL REFERENCES auctions (auction_id),
    modifier_type   SMALLINT NOT NULL,
    modifier_value  INTEGER NOT NULL,
    PRIMARY KEY (auction_id, modifier_type)
);

CREATE TABLE IF NOT EXISTS commodities (
    auction_id  BIGINT PRIMARY KEY,
    item_id     INTEGER,
    unit_price  NUMERIC(16, 4) NOT NULL
);

CREATE TABLE IF NOT EXISTS commodity_observations (
    auction_id  BIGINT NOT NULL REFERENCES commodities (auction_id),
    snapshot_ts TIMESTAMPTZ NOT NULL,
    quantity    INTEGER NOT NULL,
    time_left   TEXT NOT NULL,
    PRIMARY KEY (auction_id, snapshot_ts)
);

CREATE TABLE IF NOT EXISTS processed_files (
    filename        TEXT PRIMARY KEY,
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_auctions_item_id ON auctions (item_id);
CREATE INDEX IF NOT EXISTS idx_auction_observations_snapshot_ts ON auction_observations (snapshot_ts);
CREATE INDEX IF NOT EXISTS idx_commodities_item_id ON commodities (item_id);
CREATE INDEX IF NOT EXISTS idx_commodity_observations_snapshot_ts ON commodity_observations (snapshot_ts);
