from collections import defaultdict

students = [
    {"name": "Alice", "class": "A", "score": 85, "passed": True},
    {"name": "Bob", "class": "A", "score": 59, "passed": False},
    {"name": "Charlie", "class": "B", "score": 92, "passed": True},
    {"name": "David", "class": "B", "score": 76, "passed": True},
    {"name": "Eve", "class": "A", "score": 68, "passed": True},
    {"name": "Frank", "class": "C", "score": 45, "passed": False},
]

def passed_students(students):
    return [student for student in students if student["passed"]]

def failed_students(students):
    return [student for student in students if not student["passed"]]

def student_name(students):
    return [student["name"] for student in students]

def average_score(students):
    return sum(student["score"] for student in students)/len(students)

def top_students(students):
    return max(students, key=lambda student: student["score"])

def students_by_class(students):
    students_by_class = defaultdict(list)
    for student in students:
        students_by_class[student["class"]].append(student)
    return students_by_class

def average_score_by_class(students):
    average_score_by_class = defaultdict(float)
    students_by_class = defaultdict(list)
    for student in students:
        average_score_by_class[student["class"]] += student["score"]
        students_by_class[student["class"]].append(student)
    for key, value in average_score_by_class.items():
        average_score_by_class[key] = value / len(students_by_class[key])
    return average_score_by_class

def top_n_students(students, n):
    return sorted(students, key=lambda student: student["score"], reverse=True)[:n]

def score_distribution(students):
    score_distribution = defaultdict(int)
    score_distribution["90-100"] = 0
    score_distribution["80-89"] = 0
    score_distribution["70-79"] = 0
    score_distribution["60-69"] = 0
    score_distribution["0-59"] = 0
    for student in students:
        if student["score"] >= 90:
            score_distribution["90-100"] += 1
        elif student["score"] >= 80:
            score_distribution["80-89"] += 1
        elif student["score"] >= 70:
            score_distribution["70-79"] += 1
        elif student["score"] >= 60:
            score_distribution["60-69"] += 1
        else:
            score_distribution["0-59"] += 1
    return score_distribution