from collections import defaultdict

products = [
    {"id": 1, "name": "iPhone", "category": "phone", "price": 5999, "stock": 10},
    {"id": 2, "name": "MacBook", "category": "laptop", "price": 12999, "stock": 5},
    {"id": 3, "name": "AirPods", "category": "audio", "price": 1299, "stock": 30},
    {"id": 4, "name": "iPad", "category": "tablet", "price": 3999, "stock": 8},
    {"id": 5, "name": "ThinkPad", "category": "laptop", "price": 8999, "stock": 3},
]

cart = [
    {"product_id": 1, "quantity": 2},
    {"product_id": 3, "quantity": 1},
    {"product_id": 5, "quantity": 1},
]

def product_names(products):
    return [product["name"] for product in products]

def products_by_category(products):
    products_by_category = defaultdict(list)
    for product in products:
        products_by_category[product["category"]].append(product)
    return products_by_category

def total_stock_value(products):
    total_stock_value = 0
    for product in products:
        total_stock_value += product["price"] * product["stock"]
    return total_stock_value

def find_product(products, product_id):
    product_by_id = {product["id"]: product for product in products}
    return product_by_id[product_id]

def cart_items(products, cart):
    cart_items = []
    for item in cart:
        cart_item = {"product": find_product(products, item["product_id"]), "quantity": item["quantity"]}
        cart_items.append(cart_item)
    return cart_items

def cart_total(products, cart):
    cart_total = 0
    for item in cart:
        product = find_product(products, item["product_id"])
        cart_total += product["price"] * item["quantity"]
    return cart_total

def is_cart_available(products, cart):
    is_cart_available = True
    for item in cart:
        product = find_product(products, item["product_id"])
        is_cart_available = is_cart_available and item["quantity"] <= product["stock"]
    return is_cart_available

def category_total_stock(products):
    category_total_stock = defaultdict(int)
    for product in products:
        category_total_stock[product["category"]] += product["stock"]
    return category_total_stock

def top_n_expensive_products(products, n):
    return sorted(products, key=lambda product: product["price"], reverse=True)[:n]