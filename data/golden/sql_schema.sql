-- SQL golden dataset schema and seed data for text-to-SQL evaluation
-- This file is loaded into an in-memory or file-based SQLite database at evaluation time

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    signup_date DATE NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    total_price REAL NOT NULL,
    order_date DATE NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);

INSERT INTO customers (id, name, email, city, state, signup_date) VALUES
    (1, 'Alice Johnson', 'alice@example.com', 'Seattle', 'WA', '2023-01-15'),
    (2, 'Bob Smith', 'bob@example.com', 'Portland', 'OR', '2023-03-22'),
    (3, 'Carol Davis', 'carol@example.com', 'San Francisco', 'CA', '2023-05-10'),
    (4, 'David Wilson', 'david@example.com', 'Seattle', 'WA', '2023-07-08'),
    (5, 'Eva Martinez', 'eva@example.com', 'Los Angeles', 'CA', '2023-08-19'),
    (6, 'Frank Lee', 'frank@example.com', 'Austin', 'TX', '2023-09-01'),
    (7, 'Grace Kim', 'grace@example.com', 'San Francisco', 'CA', '2023-10-15'),
    (8, 'Henry Brown', 'henry@example.com', 'Denver', 'CO', '2023-11-20');

INSERT INTO products (id, name, category, unit_price) VALUES
    (1, 'Widget Pro', 'Electronics', 29.99),
    (2, 'Widget Lite', 'Electronics', 14.99),
    (3, 'Gadget Plus', 'Tools', 49.99),
    (4, 'Gadget Max', 'Tools', 79.99),
    (5, 'Super Cable', 'Accessories', 9.99),
    (6, 'Phone Stand', 'Accessories', 12.99);

INSERT INTO orders (id, customer_id, product_id, quantity, total_price, order_date, status) VALUES
    (1, 1, 1, 2, 59.98, '2024-01-10', 'completed'),
    (2, 1, 3, 1, 49.99, '2024-01-15', 'completed'),
    (3, 2, 2, 3, 44.97, '2024-02-03', 'completed'),
    (4, 3, 1, 1, 29.99, '2024-02-20', 'shipped'),
    (5, 3, 5, 4, 39.96, '2024-02-25', 'completed'),
    (6, 4, 4, 1, 79.99, '2024-03-01', 'completed'),
    (7, 4, 6, 2, 25.98, '2024-03-05', 'shipped'),
    (8, 5, 1, 1, 29.99, '2024-03-10', 'completed'),
    (9, 5, 3, 2, 99.98, '2024-03-12', 'completed'),
    (10, 6, 2, 5, 74.95, '2024-03-15', 'completed'),
    (11, 6, 6, 1, 12.99, '2024-03-18', 'shipped'),
    (12, 7, 1, 3, 89.97, '2024-03-20', 'completed'),
    (13, 8, 5, 2, 19.98, '2024-03-22', 'completed');
