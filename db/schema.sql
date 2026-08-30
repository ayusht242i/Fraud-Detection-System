CREATE TABLE Users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE Accounts (
    account_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES Users(user_id),
    balance NUMERIC(12,2) DEFAULT 0,
    account_type VARCHAR(30)
);

CREATE TABLE Merchants (
    merchant_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50)
);

CREATE TABLE Transactions (
    txn_id SERIAL PRIMARY KEY,
    account_id INT REFERENCES Accounts(account_id),
    merchant_id INT REFERENCES Merchants(merchant_id),
    amount NUMERIC(12,2) NOT NULL,
    txn_time TIMESTAMP DEFAULT NOW(),
    location VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending'
);

CREATE TABLE Audit_Logs (
    log_id SERIAL PRIMARY KEY,
    txn_id INT REFERENCES Transactions(txn_id),
    flag_reason VARCHAR(200),
    risk_score NUMERIC(4,3),
    flagged_at TIMESTAMP DEFAULT NOW()
);