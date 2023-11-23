-- Creation of the Items table
CREATE TABLE Items (
    item_id INT PRIMARY KEY,
    name VARCHAR(100), -- Adjust as per actual maximum length
    quality VARCHAR(50),
    level INT,
    required_level INT,
    item_class VARCHAR(50),
    item_subclass VARCHAR(50),
    purchase_price_gold INT,
    purchase_price_silver INT,
    sell_price_gold INT,
    sell_price_silver INT,
    max_count INT,
    is_stackable INT
);

-- Creation of the Auctions table
CREATE TABLE Auctions (
    auction_id INT PRIMARY KEY,
    bid INT,
    buyout INT,
    quantity INT,
    time_left VARCHAR(20), -- Adjust as per actual maximum length
    item_id INT,
    FOREIGN KEY (item_id) REFERENCES Items(item_id)
);

-- Creation of the ActionEvents table
CREATE TABLE ActionEvents (
    action_event_id INT PRIMARY KEY,
    auction_id INT,
    timestamp DATETIME,
    hours_listed INT,
    item_id INT,
    FOREIGN KEY (auction_id) REFERENCES Auctions(auction_id),
    FOREIGN KEY (item_id) REFERENCES Items(item_id)
);

