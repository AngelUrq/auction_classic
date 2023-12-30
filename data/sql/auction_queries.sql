
CREATE TEMPORARY TABLE TempAuctions AS
SELECT
    a.auction_id,
    a.bid / 10000 AS bid_in_gold,
    a.buyout / 10000 AS buyout_in_gold,
    (a.buyout / 10000) / a.quantity AS unit_price,
    a.quantity,
    a.time_left,
    a.item_id,
    COUNT(*) AS hours_on_sale,
    MIN(ae.record) AS first_appearance_timestamp
FROM Auctions a
JOIN ActionEvents ae ON a.auction_id = ae.auction_id
GROUP BY a.auction_id;

CREATE TEMPORARY TABLE TempAuctionsSameTime AS
SELECT
    ta.auction_id,
    COUNT(DISTINCT a.auction_id) AS auctions_at_same_time
FROM TempAuctions ta
JOIN Auctions a ON ta.item_id = a.item_id
GROUP BY ta.auction_id;

CREATE TEMPORARY TABLE TempAvgCompetitorUnitPrice AS
SELECT
    ta.auction_id,
    AVG(ta.unit_price) AS avg_competitor_unit_price,
    MIN(ta.unit_price) AS min_competitor_unit_price
FROM TempAuctions ta
JOIN Auctions a ON ta.item_id = a.item_id AND ta.auction_id <> a.auction_id
GROUP BY ta.auction_id;

SELECT
    ta.auction_id,
    ta.bid_in_gold,
    ta.buyout_in_gold,
    ta.unit_price,
    ta.quantity,
    ta.time_left,
    ta.item_id,
    ta.hours_on_sale,
    ta.first_appearance_timestamp,
    YEAR(ta.first_appearance_timestamp) AS appearance_year,
    MONTH(ta.first_appearance_timestamp) AS appearance_month,
    DAY(ta.first_appearance_timestamp) AS appearance_day,
    HOUR(ta.first_appearance_timestamp) AS appearance_hour,
    tst.auctions_at_same_time,
    acup.avg_competitor_unit_price,
    acup.min_competitor_unit_price
FROM TempAuctions ta
JOIN TempAuctionsSameTime tst ON ta.auction_id = tst.auction_id
JOIN TempAvgCompetitorUnitPrice acup ON ta.auction_id = acup.auction_id;
