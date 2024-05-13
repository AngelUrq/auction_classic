SELECT
    a.auction_id,
    a.bid / 10000.0 AS bid_in_gold,
    a.buyout / 10000.0 AS buyout_in_gold,
    (a.buyout / 10000.0) / a.quantity AS unit_price,
    a.quantity,
    a.time_left,
    a.item_id,
    i.item_name,
    i.quality,
    i.item_class,
    i.item_subclass,
    i.is_stackable,
    i.purchase_price_gold,
    i.required_level,
    i.item_level,
    i.sell_price_gold,
    MIN(ae.record) AS first_appearance_timestamp,
    YEAR(MIN(ae.record)) AS first_appearance_year,
    MONTH(MIN(ae.record)) AS first_appearance_month,
    DAY(MIN(ae.record)) AS first_appearance_day,
    HOUR(MIN(ae.record)) AS first_appearance_hour,
    COUNT(*) AS hours_on_sale
FROM Auctions a
JOIN ActionEvents ae ON a.auction_id = ae.auction_id
JOIN Items i on i.item_id = a.item_id
GROUP BY a.auction_id
HAVING DATE(first_appearance_timestamp) NOT IN (
    '2024-03-1', 
    '2024-03-2')