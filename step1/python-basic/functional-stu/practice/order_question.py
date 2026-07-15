from collections import defaultdict

orders = [
    {"id": 1, "user": "A", "amount": 120, "paid": True},
    {"id": 2, "user": "B", "amount": 80, "paid": False},
    {"id": 3, "user": "A", "amount": 200, "paid": True},
]

def paid_order(orders):
    paid_orders = [order for order in orders if order["paid"] == True]
    # 第二种写法
    # paid_orders = [order for order in orders if order["paid"]]
    print(paid_orders)

def total_amount(orders):
    print(sum([order["amount"] for order in orders]))

def orders_by_user(orders):
    orders_by_user = defaultdict(list)
    for order in orders:
        orders_by_user[order["user"]].append(order)
    print(orders_by_user)

def high_value_orders(orders, threshold):
    print([order for order in orders if order["amount"] >= threshold])

paid_order(orders)
total_amount(orders)
orders_by_user(orders)
high_value_orders(orders, 100)