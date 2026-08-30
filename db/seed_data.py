import random
from datetime import datetime, timedelta

random.seed(42)

locations = ['Pune', 'Mumbai', 'Delhi', 'Bengaluru', 'Dubai', 'London']
statuses = ['confirmed', 'confirmed', 'confirmed', 'pending']

output = []

# Seed 5 users, 5 accounts, 5 merchants first
for i in range(1, 6):
    output.append(f"INSERT INTO Users (name, email) VALUES ('User{i}', 'user{i}@test.com');")
for i in range(1, 6):
    output.append(f"INSERT INTO Accounts (user_id, balance, account_type) VALUES ({i}, {random.randint(1000,50000)}, 'savings');")
for i in range(1, 6):
    output.append(f"INSERT INTO Merchants (name, category) VALUES ('Merchant{i}', 'retail');")

# Generate 200 normal transactions
base_time = datetime(2026, 1, 1)
for i in range(200):
    account_id = random.randint(1, 5)
    merchant_id = random.randint(1, 5)
    amount = round(random.uniform(50, 3000), 2)
    location = 'Pune'  # normal transactions from home city
    txn_time = base_time + timedelta(hours=random.randint(0, 720))
    output.append(
        f"INSERT INTO Transactions (account_id, merchant_id, amount, txn_time, location, status) "
        f"VALUES ({account_id}, {merchant_id}, {amount}, '{txn_time}', '{location}', '{random.choice(statuses)}');"
    )

# Plant deliberate edge cases for testing triggers later
output.append(
    "INSERT INTO Transactions (account_id, merchant_id, amount, txn_time, location, status) "
    "VALUES (1, 1, 45000, '2026-01-15 14:00:00', 'Dubai', 'pending');"  # high amount deviation
)
output.append(
    "INSERT INTO Transactions (account_id, merchant_id, amount, txn_time, location, status) "
    "VALUES (2, 2, 500, '2026-01-16 10:00:00', 'Pune', 'confirmed');"
)
output.append(
    "INSERT INTO Transactions (account_id, merchant_id, amount, txn_time, location, status) "
    "VALUES (2, 2, 600, '2026-01-16 10:45:00', 'London', 'pending');"  # impossible travel (45 min gap)
)

with open('db/seed_data.sql', 'w') as f:
    f.write('\n'.join(output))

print(f"Generated {len(output)} INSERT statements to db/seed_data.sql")