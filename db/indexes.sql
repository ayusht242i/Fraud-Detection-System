CREATE INDEX idx_txn_account_time ON Transactions(account_id, txn_time);
CREATE INDEX idx_txn_timestamp ON Transactions(txn_time);
CREATE INDEX idx_audit_risk_score ON Audit_Logs(risk_score);
CREATE INDEX idx_audit_txn_id ON Audit_Logs(txn_id);