CREATE OR REPLACE FUNCTION flag_suspicious_txn()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.amount > (
        SELECT AVG(amount) * 5 FROM Transactions
        WHERE account_id = NEW.account_id AND txn_id != NEW.txn_id
    ) THEN
        INSERT INTO Audit_Logs(txn_id, flag_reason, risk_score, flagged_at)
        VALUES (NEW.txn_id, 'High amount deviation', 0.7, NOW());
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_check_txn
AFTER INSERT ON Transactions
FOR EACH ROW EXECUTE FUNCTION flag_suspicious_txn();

CREATE OR REPLACE FUNCTION flag_impossible_travel()
RETURNS TRIGGER AS $$
DECLARE
    last_location VARCHAR(100);
    last_time TIMESTAMP;
BEGIN
    SELECT location, txn_time INTO last_location, last_time
    FROM Transactions
    WHERE account_id = NEW.account_id
      AND txn_id != NEW.txn_id
      AND txn_time < NEW.txn_time
    ORDER BY txn_time DESC LIMIT 1;

    IF last_location IS NOT NULL
       AND last_location != NEW.location
       AND NEW.txn_time - last_time < INTERVAL '2 hours' THEN
        INSERT INTO Audit_Logs(txn_id, flag_reason, risk_score, flagged_at)
        VALUES (NEW.txn_id, 'Impossible travel pattern', 0.75, NOW());
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_impossible_travel
AFTER INSERT ON Transactions
FOR EACH ROW EXECUTE FUNCTION flag_impossible_travel();
