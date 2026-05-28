
#This code was made by Alexis Rosenfeld for the eperiment conclude in the reinforcemnt leranign course
#given by prof Dimitrakakis at the neuhcatel university

#For reading simplification the code aim to reproduce report's structures
#It has indeed the folowing strucutre : 
#Utility 

#Some part was factorized thank to openAI codex but is flagged as 

#There the different library i'm using : sqlite3 is used for my 
import sqlite3
import math
import random
import time
import matplotlib.pyplot as plt


EPISODES = 20000
STEP = 5.0
DB_PATH = "data.db"


#This value are only used for the dispaly of the layout
FINGER_COLORS = {
    1: "red",
    2: "orange",
    3: "green",
    4: "blue",
    5: "purple",
}


#This is the 
FINGER_ERGO_FACTOR = {
    1: 0.5,
    2: 1.0,
    3: 1.5,
    4: 2.0,
    5: 3.0,
}



#I reported the percentage value form wiwkipeda 
LETTER_FREQUENCY = {
    "A": 7.636, "B": 0.901, "C": 3.260, "D": 3.669,
    "E": 14.715, "F": 1.066, "G": 0.866, "H": 0.737,
    "I": 7.529, "J": 0.613, "K": 0.074, "L": 5.456,
    "M": 2.968, "N": 7.095, "O": 5.796, "P": 2.521,
    "Q": 1.362, "R": 6.693, "S": 7.948, "T": 7.244,
    "U": 6.311, "V": 1.838, "W": 0.049, "X": 0.427,
    "Y": 0.128, "Z": 0.326,
}

AVERAGE_LETTER_FREQUENCY = sum(LETTER_FREQUENCY.values()) / len(LETTER_FREQUENCY)





def copy_keys(keys):
    return {k: v.copy() for k, v in keys.items()}


def load_data():

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT key, x, y, r, finger_id
        FROM layout
        WHERE layout_name='default'
    """)

    layout_rows = cur.fetchall()

    cur.execute("""
        SELECT finger_id, x, y
        FROM fingerprint
        WHERE user_id=1
    """)

    fp_rows = cur.fetchall()

    conn.close()

    keys = {
        key: {
            "x": float(x),
            "y": float(y),
            "radius": float(r),
            "finger_id": int(f)
        }
        for key, x, y, r, f in layout_rows
    }

    fingers = {
        int(f): {
            "x": float(x),
            "y": float(y)
        }
        for f, x, y in fp_rows
    }

    return keys, fingers

# I apply a correction In my dataset 
def fix_dataset_finger_assignment(keys):

    corrections = {
        "B": 3,
        "N": 4,
        "M": 5,
    }

    for key, finger in corrections.items():
        if key in keys:
            keys[key]["finger_id"] = finger

    return keys


#The eulideand disatnce that will be used in the utility fucniton

def dist(a, b):
    return math.sqrt((a["x"] - b["x"])**2 + (a["y"] - b["y"])**2)


#this is the basic cost fucntio whrer i only took the euliden diantac

def naive_cost(keys, fingers):

    cost = 0

    for key, data in keys.items():

        finger = fingers[data["finger_id"]]

        cost += dist(data, finger)

    return cost


def ergonomic_cost(keys, fingers):
    cost = 0
    count = {}
    for data in keys.values():
        finger = data["finger_id"]
        count[finger] = count.get(finger, 0) + 1

    for key, data in keys.items():
        finger_id = data["finger_id"]
        finger = fingers[finger_id]
        distance_cost = dist(data, finger)
        ergonomic_factor = FINGER_ERGO_FACTOR[finger_id]
        load_factor = count[finger_id]
        freq = LETTER_FREQUENCY.get(
            key.upper(),
            AVERAGE_LETTER_FREQUENCY
        )
        usage_factor = freq / AVERAGE_LETTER_FREQUENCY

        cost += (
            distance_cost
            * ergonomic_factor
            * load_factor
            * usage_factor
        )

    return cost



#tis two 
def overlap(k1, k2):

    return dist(k1, k2) < ( k1["radius"] + k2["radius"] )


def has_collision(keys):

    values = list(keys.values())

    for i in range(len(values)):
        for j in range(i + 1, len(values)):

            if overlap(values[i], values[j]):
                return True

    return False


#This is the MOVE action. 
def move_key(keys):
    new = copy_keys(keys)
    k = random.choice(list(new.keys()))
    dx, dy = random.choice([
        (STEP, 0),
        (-STEP, 0),
        (0, STEP),
        (0, -STEP),
    ])
    new[k]["x"] += dx
    new[k]["y"] += dy
    return new, "MOVE"


def switch_reference_only(keys):
    new = copy_keys(keys)
    k1, k2 = random.sample(list(new.keys()), 2)
    new[k1]["finger_id"], new[k2]["finger_id"] = (
        new[k2]["finger_id"],
        new[k1]["finger_id"]
    )
    return new, "SWITCH"


def switch_position_and_reference(keys):

    new = copy_keys(keys)

    k1, k2 = random.sample(list(new.keys()), 2)

    new[k1]["x"], new[k2]["x"] = (
        new[k2]["x"],
        new[k1]["x"]
    )

    new[k1]["y"], new[k2]["y"] = (
        new[k2]["y"],
        new[k1]["y"]
    )

    new[k1]["finger_id"], new[k2]["finger_id"] = (
        new[k2]["finger_id"],
        new[k1]["finger_id"]
    )

    return new, "SWITCH"


#the different model I run in the experiment . This factorization of the model meanign the 
#creation of a enviroemnt setting 
def run_model(initial_keys, fingers, cost_function, switch_function):

    keys = copy_keys(initial_keys)

    switch_history = []
    gain_history = []

    switch_count = 0

    initial_cost = ergonomic_cost(initial_keys, fingers)

    for _ in range(EPISODES):

        old_cost = cost_function(keys, fingers)

        if random.random() < 0.5:
            new_keys, action = move_key(keys)
        else:
            new_keys, action = switch_function(keys)
        
        #This aprt of the code was made to keep prevent aciotn 
        if has_collision(new_keys):

            switch_history.append(switch_count)

            gain_history.append(
                initial_cost - ergonomic_cost(keys, fingers)
            )

            continue

        new_cost = cost_function(new_keys, fingers)

        reward = old_cost - new_cost

        if reward > 0:

            if action == "SWITCH":
                switch_count += 1

            keys = new_keys

        switch_history.append(switch_count)

        gain_history.append(
            initial_cost - ergonomic_cost(keys, fingers)
        )

    return keys, switch_history, gain_history



#Those values aim to 
keys, fingers = load_data()
keys = fix_dataset_finger_assignment(keys)
initial_keys = copy_keys(keys)
start_time = time.time()


naive_keys, naive_switch_history, naive_gain_history = run_model(
    initial_keys,
    fingers,
    naive_cost,
    switch_reference_only
)

ergonomic_keys, ergonomic_switch_history, ergonomic_gain_history = run_model(
    initial_keys,
    fingers,
    ergonomic_cost,
    switch_reference_only
)

advanced_keys, advanced_switch_history, advanced_gain_history = run_model(
    initial_keys,
    fingers,
    ergonomic_cost,
    switch_position_and_reference
)


#keep track of 
execution_time = round(time.time() - start_time, 2)




fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(
    naive_switch_history,
    label="Naive utility"
)
ax.plot(
    ergonomic_switch_history,
    label="Ergonomic utility"
)
ax.plot(
    advanced_switch_history,
    label="Ergonomic + position switch"
)
ax.set_title(
    f"Profitable switch actions | episodes={EPISODES}"
)
ax.set_xlabel("Episode")
ax.set_ylabel("Switch count")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(
    naive_gain_history,
    label="Naive utility"
)
ax.plot(
    ergonomic_gain_history,
    label="Ergonomic utility"
)
ax.plot(advanced_gain_history,label="Ergonomic + position switch")
ax.set_title(f"Ergonomic benchmark gain | episodes={EPISODES}")
ax.set_xlabel("Episode")
ax.set_ylabel("Gain")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()



#This is the function used to plot the evolution of display keyboard. The factorisation of the fucntion meaning 
#it's expression as a funciton was made useing codex LLM . The idea reamin from Alexis Rosenfeld

def draw_layout(before_keys, after_keys, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    layouts = [
        ("Before", before_keys),
        ("After", after_keys),
    ]
    original_fingers = {
        k: v["finger_id"]
        for k, v in before_keys.items()
    }
    for ax, (name, current_keys) in zip(axes, layouts):
        for key, data in current_keys.items():
            x = data["x"]
            y = data["y"]
            current_finger = data["finger_id"]
            original_finger = original_fingers[key]
            color = FINGER_COLORS[original_finger]
            circle = plt.Circle(
                (x, y),
                data["radius"],
                color=color,
                alpha=0.35
            )
            ax.add_patch(circle)
            ax.text(
                x,
                y,
                f"{key}\nF{current_finger}",
                ha="center",
                va="center",
                fontsize=8
            )
        for finger_id, finger in fingers.items():
            color = FINGER_COLORS[finger_id]
            ax.scatter(
                finger["x"],
                finger["y"],
                color=color,
                marker="x",
                s=120
            )
            ax.text(
                finger["x"],
                finger["y"] - 20,
                f"FP{finger_id}",
                color=color,
                ha="center"
            )
        ax.set_xlim(0, 1300)
        ax.set_ylim(0, 1000)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(
            f"{name} | cost = {round(ergonomic_cost(current_keys, fingers), 2)}"
        )
    fig.suptitle(
        f"{title} | execution={execution_time}s"
    )
    plt.tight_layout()
    plt.show()



draw_layout(
    initial_keys,
    naive_keys,
    "Naive utility"
)

draw_layout(
    initial_keys,
    ergonomic_keys,
    "Ergonomic utility"
)

draw_layout(
    initial_keys,
    advanced_keys,
    "Ergonomic + position/reference switch"
)
