CREATE TEMPORARY TABLE AuctionsHours AS
SELECT
    a.auction_id,
    a.bid / 10000 AS bid_in_gold,
    a.buyout / 10000 AS buyout_in_gold,
    (a.buyout / 10000) / a.quantity AS unit_price,
    a.quantity,
    a.time_left,
    a.item_id,
    MIN(ae.record) AS first_appearance_timestamp
FROM Auctions a
JOIN ActionEvents ae ON a.auction_id = ae.auction_id
GROUP BY a.auction_id
LIMIT 100;

CREATE TEMPORARY TABLE AuctionsCount AS
SELECT
    ah.auction_id
FROM AuctionsHours ah
JOIN Auctions a ON ah.item_id = a.item_id
GROUP BY ah.auction_id
LIMIT 100;

CREATE TEMPORARY TABLE AuctionsPrice AS
SELECT
    ah.auction_id
FROM AuctionsHours ah
JOIN Auctions a ON ah.item_id = a.item_id AND ah.auction_id <> a.auction_id
GROUP BY ah.auction_id
LIMIT 100;

SELECT
    ah.auction_id,
    ah.bid_in_gold,
    ah.buyout_in_gold,
    ah.unit_price,
    ah.quantity,
    ah.time_left,
    ah.item_id,
    ah.first_appearance_timestamp,
    YEAR(ah.first_appearance_timestamp) AS appearance_year,
    MONTH(ah.first_appearance_timestamp) AS appearance_month,
    DAY(ah.first_appearance_timestamp) AS appearance_day,
    HOUR(ah.first_appearance_timestamp) AS appearance_hour
FROM AuctionsHours ah
JOIN AuctionsCount ac ON ah.auction_id = ac.auction_id
JOIN AuctionsPrice ap ON ah.auction_id = ap.auction_id
LIMIT 100;
