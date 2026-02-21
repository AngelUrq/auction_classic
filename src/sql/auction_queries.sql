SELECT
    a.auction_id,
    a.bid AS bid,
    a.buyout AS buyout,
    a.buyout / a.quantity AS unit_price,
    a.quantity,
    MAX(CASE
        WHEN ae.time_left = 'VERY_LONG' THEN 48
        WHEN ae.time_left = 'LONG' THEN 12
        WHEN ae.time_left = 'MEDIUM' THEN 2
        WHEN ae.time_left = 'SHORT' THEN 0.5
        ELSE NULL
    END) AS time_left,
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
    COUNT(*) AS listing_duration
FROM Auctions a
JOIN ActionEvents ae ON a.auction_id = ae.auction_id
JOIN Items i on i.item_id = a.item_id
GROUP BY 
    a.auction_id,
    a.bid,
    a.buyout,
    a.quantity,
    a.item_id,
    i.item_name,
    i.quality,
    i.item_class,
    i.item_subclass,
    i.is_stackable,
    i.purchase_price_gold,
    i.required_level,
    i.item_level,
    i.sell_price_gold
HAVING DATE(first_appearance_timestamp) NOT IN (
    '2024-08-11',
    '2024-08-12',
    '2024-07-23',
    '2024-07-24'
);