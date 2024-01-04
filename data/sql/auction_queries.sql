

CREATE TEMPORARY TABLE AuctionsHours AS
SELECT
    a.auction_id,
    a.bid / 10000 AS bid_in_gold,
    a.buyout / 10000 AS buyout_in_gold,
    (a.buyout / 10000) / a.quantity AS unit_price,
    a.quantity,
    a.time_left,
    a.item_id,
    MIN(ae.record) AS first_appearance_timestamp,
    COUNT(*) AS hours_on_sale
FROM Auctions a
JOIN ActionEvents ae ON a.auction_id = ae.auction_id
GROUP BY a.auction_id

SELECT
    ah.auction_id,
    ah.bid_in_gold,
    ah.buyout_in_gold,
    ah.unit_price,
    ah.quantity,
    ah.time_left,
    ah.item_id,
    ah.first_appearance_timestamp,
    ah.hours_on_sale
FROM AuctionsHours ah

