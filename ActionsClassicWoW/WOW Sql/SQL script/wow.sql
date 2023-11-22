-- Creación de la tabla Items
CREATE TABLE Items (
    item_id INT PRIMARY KEY,
    name VARCHAR(255),
    quality VARCHAR(255),
    level INT,
    required_level INT,
    item_class VARCHAR(255),
    item_subclass VARCHAR(255),
    purchase_price_gold INT,
    purchase_price_silver INT,
    sell_price_gold INT,
    sell_price_silver INT,
    max_count INT,
    is_stackable INT
);

-- Creación de la tabla Auctions
CREATE TABLE Auctions (
    auction_id INT PRIMARY KEY,
    bid INT,
    buyout INT,
    quantity INT,
    time_left VARCHAR(255),
    item_id INT,
    FOREIGN KEY (item_id) REFERENCES Items(item_id)
);

-- Creación de la tabla ActionEvents
CREATE TABLE ActionEvents (
    action_event_id INT PRIMARY KEY,
    auction_id INT,
    timestamp DATETIME,
    hours_listed INT,
    item_id INT,
    FOREIGN KEY (auction_id) REFERENCES Auctions(auction_id),
    FOREIGN KEY (item_id) REFERENCES Items(item_id)
);
