-- Crear la tabla Items
CREATE TABLE Items (
  item_id INT PRIMARY KEY,
  item_name VARCHAR(100),
  quality VARCHAR(50),
  item_level INT,
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

-- Crear la tabla Auctions
CREATE TABLE Auctions (
  auction_id INT PRIMARY KEY,
  bid INT,
  buyout INT,
  quantity INT,
  time_left VARCHAR(20)
);

-- Crear la tabla ActionEvents
CREATE TABLE ActionEvents (
  auction_id INT,
  record DATETIME,
  PRIMARY KEY (auction_id, record),
  FOREIGN KEY (auction_id) REFERENCES Auctions(auction_id)
);

