from collections import defaultdict

events = [
    {"user": "A", "event": "view", "page": "/home", "ts": 1000},
    {"user": "A", "event": "click", "page": "/home", "ts": 1010},
    {"user": "B", "event": "view", "page": "/product", "ts": 1020},
    {"user": "A", "event": "purchase", "page": "/checkout", "ts": 1050},
    {"user": "C", "event": "view", "page": "/home", "ts": 1060},
    {"user": "B", "event": "click", "page": "/product", "ts": 1080},
    {"user": "B", "event": "purchase", "page": "/checkout", "ts": 1120},
    {"user": "C", "event": "click", "page": "/home", "ts": 1130},
]


def users(events):
    return {event["user"] for event in events}


def events_by_user(events):
    events_by_user_dict = defaultdict(list)
    for event in events:
        events_by_user_dict[event["user"]].append(event)
    return events_by_user_dict


def count_by_event(events):
    count_by_user = defaultdict(int)
    for event in events:
        count_by_user[event["event"]] += 1
    return count_by_user


def count_by_page(events):
    count_by_page = defaultdict(int)
    for event in events:
        count_by_page[event["page"]] += 1
    return count_by_page


def purchase_users(events):
    return {event["user"] for event in events if event["event"] == "purchase"}


def user_events_sequence(events, user):
    events_by_user_dict = events_by_user(events)
    user_events = events_by_user_dict[user]
    sorted_events = sorted(user_events, key=lambda x: x["ts"])
    return [e["event"] for e in sorted_events]


def conversion_rate(events):
    return len(purchase_users(events)) / len(users(events))


def first_event_by_user(events):
    result = {}
    for user, user_events in events_by_user(events).items():
        result[user] = min(user_events, key=lambda e: e["ts"])
    return result


def last_event_by_user(events):
    result = {}
    for user, user_events in events_by_user(events).items():
        result[user] = max(user_events, key=lambda e: e["ts"])
    return result


def active_users(events, min_events):
    return [
        user
        for user, user_events in events_by_user(events).items()
        if len(user_events) >= min_events
    ]
