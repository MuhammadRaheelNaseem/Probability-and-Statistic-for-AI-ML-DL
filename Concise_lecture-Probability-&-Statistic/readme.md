### **Lecture 1: Introduction to Probability**


### **1. What is Probability?**

**Mathematical Explanation:**

Probability is the measure of how likely an event is to occur. It is a numerical value between 0 and 1:
- $ P(E) = 0 $ means the event will never happen.
- $ P(E) = 1 $ means the event will always happen.

The formula for calculating probability is:

$$
P(E) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}
$$

#### **Real-World Example:**
**Scenario**: Predicting Rain

- Imagine you’re trying to predict the weather. The weather forecaster gives a 70% chance of rain tomorrow.
- If there are 100 days with similar conditions, on average, it will rain 70 of those days.

In this case, the probability of rain tomorrow is 0.7, which represents 70%.

**Calculation**:

We can consider a real-world example of flipping a coin to explain the basics:
- Outcomes: Heads (H) and Tails (T)
- Probability of heads:

$$
P(H) = \frac{1}{2} = 0.5
$$

<img src="https://static.scientificamerican.com/sciam/cache/file/5DC7AECC-6B6A-489E-A11618E62DC7BC4F_source.jpg?w=300" style="text-align:center;">




```python
import random

def coin_toss_simulation(trials):
    heads = 0
    for _ in range(trials):
        toss = random.choice(["Heads", "Tails"])
        if toss == "Heads":
            heads += 1
    return heads / trials


# Simulate tossing a coin 1000 times
probability_heads = coin_toss_simulation(20)
print(f"Empirical Probability of Heads: {probability_heads}")
```

    Empirical Probability of Heads: 0.4
    

---

### **2. Real-World Examples**

#### **Example 1: Coin Toss**

- In real life, you may flip a coin to make a decision, like choosing between two options for dinner. 
- The probability of getting heads is 0.5, so the decision is equally balanced, which reflects how randomness can be a tool in everyday life.

<img src="https://images.squarespace-cdn.com/content/v1/54905286e4b050812345644c/1609608174920-34RZY6EW61N8185W83LY/Rule2.jpg?w=300" style="text-align:center;">


#### **Example 2: Dice Roll**


- In a game like Monopoly, you roll a fair die to move around the board. If you need a 6 to land on a specific property, the probability of rolling a 6 is:

##### `or`

**"Why Players Often Land on Orange Properties in Monopoly"**

Imagine you're playing Monopoly. One of your friends just got out of jail. Their next move will likely be 6 to 9 steps forward — because statistically, most dice rolls result in a total between 6 and 9.

🔸 The orange properties (St. James Place, Tennessee Avenue, New York Avenue) are exactly 6 to 9 spaces away from jail.

**According to dice probability:**
- 7 is the most frequent roll (with 6 possible combinations).
- So there's a high chance the player will land on one of the orange properties.

**Smart Strategy**: If you’ve built houses on orange properties, there's a strong likelihood you’ll collect rent frequently — increasing your income.

<img src="https://c8.alamy.com/comp/BF7EFJ/monopoly-BF7EFJ.jpg?w=400" style="text-align:center;">


When rolling two 6-sided dice, the total number of possible outcomes is:

```
6 × 6 = 36
```

Now, let’s see how many ways each sum (from 2 to 12) can be achieved:

| Sum | Combinations                               | No. of Ways |
|-----|--------------------------------------------|-------------|
| 2   | (1,1)                                       | 1           |
| 3   | (1,2), (2,1)                                | 2           |
| 4   | (1,3), (2,2), (3,1)                         | 3           |
| 5   | (1,4), (2,3), (3,2), (4,1)                  | 4           |
| 6   | (1,5), (2,4), (3,3), (4,2), (5,1)           | 5           |
| 7   | (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)    | 6           |
| 8   | (2,6), (3,5), (4,4), (5,3), (6,2)           | 5           |
| 9   | (3,6), (4,5), (5,4), (6,3)                  | 4           |
| 10  | (4,6), (5,5), (6,4)                         | 3           |
| 11  | (5,6), (6,5)                                | 2           |
| 12  | (6,6)                                       | 1           |

As you can see, **7** has the most combinations — that’s why it’s the most probable outcome when rolling two dice.




```python
# All possible combinations of 2 dice rolls
dice_totals = [i + j for i in range(1, 7) for j in range(1, 7)]

# Manually count frequency of each total
counted = {}

for total in dice_totals:
    if total in counted:
        counted[total] += 1
    else:
        counted[total] = 1

# print(counted)

# Print results
for total in sorted(counted):
    print(f"Sum: {total}, Ways: {counted[total]}")

```

    Sum: 2, Ways: 1
    Sum: 3, Ways: 2
    Sum: 4, Ways: 3
    Sum: 5, Ways: 4
    Sum: 6, Ways: 5
    Sum: 7, Ways: 6
    Sum: 8, Ways: 5
    Sum: 9, Ways: 4
    Sum: 10, Ways: 3
    Sum: 11, Ways: 2
    Sum: 12, Ways: 1
    


#### **Example 3: Predicting Traffic**

**Scenario**: Traffic Probability Model

- Imagine you're deciding when to leave for work. A traffic app might tell you that there's a 30% chance of heavy traffic at 8 AM. 
- The probability reflects real-time data gathered from previous traffic patterns, helping you make an informed decision.


<img src="https://straterix.com/wp-content/uploads/2023/12/shutterstock_1693832470-2048x1923-1.jpg?" width=400>
<img src="https://t3.ftcdn.net/jpg/05/07/34/00/360_F_507340069_TnRJ7EFXauTvenoB9VgOq5YQXg4hFEnt.jpg" width=400>

**Empirical Calculation**:  (verifiable by observation or experience)

You can observe past traffic patterns (say, over 100 days) and calculate the empirical probability based on how often heavy traffic occurs at that hour.

---

### **3. Sample Space & Events**

#### **Sample Space (S):**

The sample space is the set of all possible outcomes of an experiment.

- **Coin Toss**: $ S = \{ \text{Heads}, \text{Tails} \} $
- **Dice Roll**: $ S = \{ 1, 2, 3, 4, 5, 6 \} $

**Real-World Application:**
- In real life, understanding the sample space is crucial when making decisions based on risk and uncertainty. For example, in a medical trial, the sample space might be the different outcomes of a treatment: "Effective" or "Not Effective."

#### **Event (E):**

<img src="https://statisticstechs.weebly.com/uploads/6/5/2/4/65248169/published/picture2_13.png?1509950446" width=300>

An event is a specific outcome or set of outcomes from the sample space.

- **For Coin Toss**: $ E = \{ \text{Heads} \} $
- **For Dice Roll**: If you want to roll an even number, the event is $ E = \{ 2, 4, 6 \} $

---

### **4. Classical vs Empirical Probability**

#### **Classical Probability**:


<img src="https://itfeature.com/wp-content/uploads/2024/07/Emperical-Probability-and-Classical-Probability.jpg" width=600>


Classical probability assumes that all outcomes are equally likely. It's useful for experiments where we know the exact number of possible outcomes, such as tossing a fair coin or rolling a fair die.

- **Example (Die Roll)**: The probability of rolling a 3:
$$
P(3) = \frac{1}{6}
$$

#### **Empirical Probability**:

Empirical probability is based on real-world observations and is calculated as:

$$
P(E) = \frac{\text{Number of times event E occurs}}{\text{Total number of trials}}
$$


For example, if we conduct an experiment where we roll a die 100 times and observe how often we roll a 3, we can calculate the empirical probability based on the data.
$$
P(3) = \frac{18}{100} = 0.18
$$



```python

import random

# Classical probability
# dice has 6 sides so any has same chance to come e.g 3
classical_probability = 1/6
print(f"Classical Probability of rolling a 3: {classical_probability}")

# Empirical probability
# if we roll dice 1000 times than check how many time 3 will appear
total_rolls = 100
count_three = 0

for _ in range(total_rolls):
    roll = random.randint(1, 6)  # dice roll (1 se 6 tak random number)
    if roll == 3:
        count_three += 1

empirical_probability = count_three / total_rolls
print(f"Empirical Probability of rolling a 3 after {total_rolls} rolls: {empirical_probability}")


```

    Classical Probability of rolling a 3: 0.16666666666666666
    Empirical Probability of rolling a 3 after 100 rolls: 0.16
    

---

### **5. Types of Events**


<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR2rtIRo6Gr-d-xbQs54LBQyz487SUJ0D1J4Q&s" width=600>


#### **1. Independent Events**

**Independent events** are events where the occurrence of one does not affect the probability of the other event. In other words, the outcome of one event has **no impact** on the outcome of another event.

##### **Example: Coin Toss and Dice Roll**

- **Coin Toss**: You toss a coin, and it can either land on **Heads** or **Tails**.
- **Dice Roll**: You roll a die, and it can land on any number between 1 and 6.

**Why are these independent?**
- The result of the coin toss (whether it lands on heads or tails) **does not affect** the result of the die roll (whether it lands on a 1, 2, 3, 4, 5, or 6). These events are not related to each other.

**Mathematical Formula**:
If you want to calculate the probability of both events happening together:
$$
P(\text{Heads and 3}) = P(\text{Heads}) \times P(3)
$$

**Real-World Example**:  
- Your performance in a school exam and your friend's football match result are **independent** events. If you do well in your exam, it does not affect the outcome of your friend's football match.

---




```python
import random

# Coin Toss
coin_toss = random.choice(['Heads', 'Tails'])

# Dice Roll
dice_roll = random.randint(1, 6)

# Independent events ka example
print(f"Coin Toss: {coin_toss}")
print(f"Dice Roll: {dice_roll}")

```

    Coin Toss: Heads
    Dice Roll: 2
    


```python
def independent_events_simulation(trials):
    heads_and_three = 0
    for _ in range(trials):
        toss = random.choice(["Heads", "Tails"])
        roll = random.randint(1, 6)
        if toss == "Heads" and roll == 3:
            heads_and_three += 1
    return heads_and_three / trials

# Simulate 1000 independent events
probability_independent = independent_events_simulation(1000)
print(f"Empirical Probability of Heads and 3: {probability_independent}")

```

    Empirical Probability of Heads and 3: 0.081
    

<!-- 
2. **Dependent Events**:

- Events are dependent if the outcome of one event affects the other.
  **Example**: Drawing cards without replacement from a deck. Once you draw one card, the sample space for the next draw changes.
 -->


### **2. Dependent Events**

**Dependent events** are events where the outcome of one event **affects** the probability of the other event. In other words, the occurrence of the first event influences the likelihood of the second event.

#### **Example: Drawing Balls from a Bag**

- Imagine you have a bag with 3 red balls and 2 blue balls.
- You draw the first ball, and then draw the second ball.

**Why are these dependent?**
- When you draw the first ball, the total number of balls in the bag decreases, and this affects the outcome of the second draw.
- If you draw a red ball first, there will be fewer red balls left in the bag for the second draw, making the probability of drawing a red ball again different from the first time.

**Mathematical Formula**:
To calculate the probability of both events happening together:
$$
P(\text{First Red, Second Blue}) = P(\text{First Red}) \times P(\text{Second Blue after First Red})
$$



```python
bag = ['Amber','Blue','Charcoal','Gold','Orange','Red', 'pink','yellow','grey']
# Loop through the list, pick an item, remove it, and pick the next item
# We need at least 2 items left in the bag to select both
while len(bag) > 1:  
    first_ball = bag[0]  # Pick the first item
    bag.remove(first_ball)  # Remove the first item
    
    second_ball = bag[0]  # Pick the next item after removing the first one
    bag.remove(second_ball)  # Remove the second item
    
    print(f"First Ball: {first_ball} | Second Ball: {second_ball}")
```

    First Ball: Amber | Second Ball: Blue
    First Ball: Charcoal | Second Ball: Gold
    First Ball: Orange | Second Ball: Red
    First Ball: pink | Second Ball: yellow
    


```python
import random
import matplotlib.pyplot as plt

# Coin Simulation
def coin_toss(n=1000):
    results = {"Heads": 0, "Tails": 0}
    for _ in range(n):
        toss = random.choice(["Heads", "Tails"])  # Random choice between Heads and Tails
        results[toss] += 1
    return results

# Dice Simulation (Fair Dice)
def fair_dice_roll(n=1000):
    results = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for _ in range(n):
        roll = random.randint(1, 6)  # Random roll between 1 and 6
        results[roll] += 1
    return results

# Dice Simulation (Biased Dice)
def biased_dice_roll(n=1000):
    results = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    # biased_probabilities = [0.1, 0.2, 0.15, 0.2, 0.25, 0.1]  # Probabilities for 1 to 6 (biased)
    biased_probabilities = [0.05, 0.15, 0.3, 0.2, 0.2, 0.1]  # More likely to get 3
    
    for _ in range(n):
        roll = random.choices(range(1, 7), biased_probabilities)[0]
        # print(roll)
        # var = [x for x in roll]
        # print(var)
        results[roll] += 1
    return results

# Plotting function for fair vs biased dice results
def plot_results(fair_results, biased_results):
    # Prepare data for plotting
    fair_labels = list(fair_results.keys())
    fair_counts = list(fair_results.values())
    
    biased_labels = list(biased_results.keys())
    biased_counts = list(biased_results.values())
    
    # Create figure and axis objects for plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot for Fair Dice
    ax1.bar(fair_labels, fair_counts, color='green', alpha=0.7)
    ax1.set_title("Fair Dice Rolls")
    ax1.set_xlabel("Dice Faces")
    ax1.set_ylabel("Count")
    
    # Plot for Biased Dice
    ax2.bar(biased_labels, biased_counts, color='red', alpha=0.7)
    ax2.set_title("Biased Dice Rolls")
    ax2.set_xlabel("Dice Faces")
    ax2.set_ylabel("Count")
    
    # Show the plots
    plt.tight_layout()
    plt.show()

# Main Function to run simulations and plot results
def main():
    num_simulations = 1000  # Number of simulations (tosses or rolls)
    
    # Simulate Coin Toss
    coin_results = coin_toss(num_simulations)
    print(f"Coin Toss Results after {num_simulations} simulations: {coin_results}")
    
    # Simulate Fair Dice Rolls
    fair_dice_results = fair_dice_roll(num_simulations)
    print(f"Fair Dice Roll Results after {num_simulations} simulations: {fair_dice_results}")
    
    # Simulate Biased Dice Rolls
    biased_dice_results = biased_dice_roll(num_simulations)
    print(f"Biased Dice Roll Results after {num_simulations} simulations: {biased_dice_results}")
    
    # Plotting Fair vs Biased Dice Rolls
    plot_results(fair_dice_results, biased_dice_results)

if __name__ == "__main__":
    main()

```

    Coin Toss Results after 1000 simulations: {'Heads': 472, 'Tails': 528}
    Fair Dice Roll Results after 1000 simulations: {1: 166, 2: 166, 3: 173, 4: 173, 5: 150, 6: 172}
    Biased Dice Roll Results after 1000 simulations: {1: 65, 2: 153, 3: 307, 4: 173, 5: 184, 6: 118}
    


    
![png](output_11_1.png)
    



```python
def biased_dice_roll(n=10):
    results = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    # biased_probabilities = [0.1, 0.2, 0.15, 0.2, 0.25, 0.1]  # Probabilities for 1 to 6 (biased)
    # biased_probabilities = [0.05, 0.15, 0.3, 0.2, 0.2, 0.1]  # More likely to get 3
    
    for _ in range(n):
        biased_probabilities = [0,0,0,0,0.4,0.8]  # More likely to get 3
        roll = random.choices(range(1, 7), biased_probabilities)[0]
        
        results[roll] += 1
        print(roll)
    return results

print(biased_dice_roll())
```

    5
    6
    6
    6
    5
    6
    5
    5
    6
    6
    {1: 0, 2: 0, 3: 0, 4: 0, 5: 4, 6: 6}
    


```python
# import random
# import plotly.graph_objects as go

# # Coin Toss Simulation
# def coin_toss(n=1000):
#     heads = 0
#     tails = 0
#     for _ in range(n):
#         toss = random.choice(["Heads", "Tails"])
#         if toss == "Heads":
#             heads += 1
#         else:
#             tails += 1
#     return heads, tails

# # Fair Dice Roll Simulation (Equal chance for 1 to 6)
# def fair_dice_roll(n=1000):
#     counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
#     for _ in range(n):
#         roll = random.randint(1, 6)
#         counts[roll] += 1
#     return counts

# # Biased Dice Roll Simulation (Different chances for 1 to 6)
# def biased_dice_roll(n=1000):
#     counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
#     biased_probabilities = [0.1, 0.2, 0.15, 0.2, 0.25, 0.1]  # Some numbers are more likely
#     for _ in range(n):
#         roll = random.choices(range(1, 7), biased_probabilities)[0]
#         counts[roll] += 1
#     return counts

# # Plotting the results (for fair vs biased dice) using Plotly
# def plot_results(fair_results, biased_results):
#     # Data for the graph
#     fair_faces = list(fair_results.keys())
#     fair_counts = list(fair_results.values())
    
#     biased_faces = list(biased_results.keys())
#     biased_counts = list(biased_results.values())

#     # Fair Dice Graph
#     fair_trace = go.Bar(
#         x=fair_faces,
#         y=fair_counts,
#         name='Fair Dice Rolls',
#         marker=dict(color='green', opacity=0.7)
#     )

#     # Biased Dice Graph
#     biased_trace = go.Bar(
#         x=biased_faces,
#         y=biased_counts,
#         name='Biased Dice Rolls',
#         marker=dict(color='red', opacity=0.7)
#     )

#     # Layout for the graph
#     layout = go.Layout(
#         title="Fair vs Biased Dice Rolls",
#         xaxis=dict(title="Dice Faces (1 to 6)"),
#         yaxis=dict(title="Count"),
#         barmode='group',  # Group the bars for fair and biased dice together
#         hovermode='closest'  # Hover mode for more interactivity
#     )

#     # Plot the graph
#     fig = go.Figure(data=[fair_trace, biased_trace], layout=layout)
#     fig.show()

# # Main function to run simulations and display results
# def main():
#     num_simulations = 1000  # Number of tosses or rolls
    
#     # Coin Toss Simulation
#     heads, tails = coin_toss(num_simulations)
#     print(f"Coin Toss Results: Heads: {heads}, Tails: {tails}")
    
#     # Fair Dice Simulation
#     fair_dice_results = fair_dice_roll(num_simulations)
#     print(f"Fair Dice Roll Results: {fair_dice_results}")
    
#     # Biased Dice Simulation
#     biased_dice_results = biased_dice_roll(num_simulations)
#     print(f"Biased Dice Roll Results: {biased_dice_results}")
    
#     # Plot results for fair vs biased dice
#     plot_results(fair_dice_results, biased_dice_results)

# if __name__ == "__main__":
#     main()

```

# 

## Lecture 2: Conditional Probability & Bayes’ Theorem

### 1. Conditional Probability

**Definition**  
> Conditional probability is the probability of an event **A** occurring given that another event **B** has already occurred.

**Formula**  
$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$
- $ P(A|B) $: Probability of A given B.  
- $ P(A \cap B) $: Probability that both A and B occur.  
- $ P(B) $: Probability that B occurs.

---

### 2. Scenario: Email Spam Detection

- **A** = Email is **spam**.  
- **B** = Email contains the word **“offer”**.

**Assumptions for 100 emails**:
- $30\%$ are spam → 30 spam, 70 non-spam.  
- Among spam, $70\%$ contain “offer” → 21 spam with offer.  
- Among non-spam, $10\%$ contain “offer” → 7 non-spam with offer.

---

### 3. Manual Calculation (using a table)

|                   | “offer” (B) | No “offer” (¬B) | Total |
|-------------------|-------------|-----------------|-------|
| **Spam (A)**      | 21          | 9               | 30    |
| **Non-Spam (¬A)** | 7           | 63              | 70    |
| **Total**         | 28          | 72              | 100   |

$$
P(A|B) = \frac{\text{Spam \& offer}}{\text{Total offer}} = \frac{21}{28} = 0.75
$$

**Interpretation**  
> If an email contains “offer,” there is a **75%** chance it is spam.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*Fm58r_RQ53sEHfwFa28LpA.png" width=700>



```python

import random
import matplotlib.pyplot as plt

def simulate_emails(trials=10000):
    offer_spam = 0
    offer_nonspam = 0
    for _ in range(trials):
        is_spam = random.random() < 0.3
        if is_spam:
            has_offer = (random.random() < 0.7)
        else:
            has_offer = (random.random() < 0.1)
        if has_offer:
            if is_spam:
                offer_spam += 1
            else:
                offer_nonspam += 1
    return offer_spam, offer_nonspam

# Run simulation
offer_spam, offer_nonspam = simulate_emails(10000)
total_offer = offer_spam + offer_nonspam
p_spam_given_offer = offer_spam / total_offer

print(f"P(Spam | 'offer') ≈ {p_spam_given_offer:.2f} ({offer_spam}/{total_offer})")

# Plot results
labels = ['Spam (with offer)', 'Non-Spam (with offer)']
counts = [offer_spam, offer_nonspam]

plt.bar(labels, counts, edgecolor='black')
plt.title("Emails with 'offer': Spam vs Non-Spam")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

    P(Spam | 'offer') ≈ 0.75 (2091/2795)
    


    
![png](output_16_1.png)
    


---

#### **2. Bayes' Theorem**:

   - **Manual Calculation**:
     - **Definition**: Bayes' Theorem allows you to revise your initial beliefs (prior probability) in the light of new evidence (likelihood).
<div style="display: flex; justify-content: space-between; align-items: center; width: 900px;">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/20250203174215734570/Bayes-theorem-1.webp" alt="Image 1" style="width: 300px;">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/20250203174215531466/Bayes-theorem-2.webp" alt="Image 2" style="width: 300px;">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/20250203174215336554/Bayes-theorem-3.webp" alt="Image 3" style="width: 300px;">
</div>


**Applying Bayes' Theorem to the Cat and Dog Example**

- **Formula**:
    $$
    P(\text{Cat}|\text{Quiet}) = \frac{P(\text{Quiet}|\text{Cat}) \times P(\text{Cat})}{P(\text{Quiet})}
    $$   $$
    P(\text{Dog}|\text{Quiet}) = \frac{P(\text{Quiet}|\text{Dog}) \times P(\text{Dog})}{P(\text{Quiet})}
    $$

    Where:
    - $P(\text{Cat}|\text{Quiet})$: Probability it's a cat given it's quiet (posterior probability for cat).
    - $P(\text{Dog}|\text{Quiet})$: Probability it's a dog given it's quiet (posterior probability for dog).
    - $P(\text{Quiet}|\text{Cat})$: Probability of being quiet given it's a cat (likelihood for cat).
    - $P(\text{Quiet}|\text{Dog})$: Probability of being quiet given it's a dog (likelihood for dog).
    - $P(\text{Cat})$: Initial probability it's a cat (prior probability for cat).
    - $P(\text{Dog})$: Initial probability it's a dog (prior probability for dog).
    - $P(\text{Quiet})$: Total probability of the pet being quiet.

- **Given Information from the Image**:
    - Prior probability of a cat: $P(\text{Cat}) = 0.5$
    - Prior probability of a dog: $P(\text{Dog}) = 0.5$
    - Probability of being quiet given it's a cat: $P(\text{Quiet}|\text{Cat}) = 0.8$
    - Probability of being quiet given it's a dog: $P(\text{Quiet}|\text{Dog}) = 0.3$

- **Step 1: Calculate the Total Probability of the Pet Being Quiet ($P(\text{Quiet})$)**

    The pet can be quiet in two ways: either it's a cat and it's quiet, or it's a dog and it's quiet. We need to sum the probabilities of these two mutually exclusive scenarios:

    $$
    P(\text{Quiet}) = P(\text{Quiet}|\text{Cat}) \times P(\text{Cat}) + P(\text{Quiet}|\text{Dog}) \times P(\text{Dog})
    $$   $$
    P(\text{Quiet}) = (0.8 \times 0.5) + (0.3 \times 0.5)
    $$   $$
    P(\text{Quiet}) = 0.4 + 0.15
    $$   $$
    P(\text{Quiet}) = 0.55
    $$

- **Step 2: Calculate the Posterior Probability for the Cat ($P(\text{Cat}|\text{Quiet})$)**

    Now we can plug the values into Bayes' Theorem for the cat:

    $$
    P(\text{Cat}|\text{Quiet}) = \frac{P(\text{Quiet}|\text{Cat}) \times P(\text{Cat})}{P(\text{Quiet})}
    $$   $$
    P(\text{Cat}|\text{Quiet}) = \frac{0.8 \times 0.5}{0.55}
    $$   $$
    P(\text{Cat}|\text{Quiet}) = \frac{0.4}{0.55}
    $$   $$
    P(\text{Cat}|\text{Quiet}) \approx 0.72727...
    $$

    Converting this to a percentage and rounding to one decimal place gives us **72.7%**.

- **Step 3: Calculate the Posterior Probability for the Dog ($P(\text{Dog}|\text{Quiet})$)**

    We can do the same for the dog:

    $$
    P(\text{Dog}|\text{Quiet}) = \frac{P(\text{Quiet}|\text{Dog}) \times P(\text{Dog})}{P(\text{Quiet})}
    $$   $$
    P(\text{Dog}|\text{Quiet}) = \frac{0.3 \times 0.5}{0.55}
    $$   $$
    P(\text{Dog}|\text{Quiet}) = \frac{0.15}{0.55}
    $$   $$
    P(\text{Dog}|\text{Quiet}) \approx 0.27272...
    $$


---

   **Real-World Scenario (Manual)**:
   - Consider the case of **spam email detection**:
     - You want to determine the probability that an email is spam, given that it contains the word "free."
     - **Prior probability**: P(Spam) = 0.4, P(Not Spam) = 0.6
     - **Likelihood**:
       - P("free"|Spam) = 0.7, P("free"|Not Spam) = 0.2
     - You want to calculate the probability that an email with the word "free" is spam.
     - Applying Bayes’ Theorem:
       $$
       P(Spam|“free”) = \frac{P(“free”|Spam) P(Spam)}{P(“free”)}
       $$
       Where \( P(“free”) = P(“free”|Spam) P(Spam) + P(“free”|Not Spam) P(Not Spam) \).
       - So, calculating \( P(“free”) \):
       $$
       P(“free”) = (0.7 * 0.4) + (0.2 * 0.6) = 0.38
       $$
       - Now calculate \( P(Spam|“free”) \):
       $$
       P(Spam|“free”) = \frac{(0.7 * 0.4)}{0.38} = 0.7368 \text{ or 73.68%}
       $$
     - This means there’s a 73.68% chance that an email with the word "free" is spam.

---



```python
# Given probabilities
prior_cat = 0.5
prior_dog = 0.5
prob_quiet_given_cat = 0.8 # 80%
prob_quiet_given_dog = 0.3 # 30%

# Evidence: The pet is quiet

# Calculate the probability of the evidence (pet being quiet)
prob_quiet = (prob_quiet_given_cat * prior_cat) + (prob_quiet_given_dog * prior_dog)

# Calculate the posterior probability of it being a cat given it's quiet
posterior_cat_given_quiet = (prob_quiet_given_cat * prior_cat) / prob_quiet

# Calculate the posterior probability of it being a dog given it's quiet
posterior_dog_given_quiet = (prob_quiet_given_dog * prior_dog) / prob_quiet

print(f"Probability (Cat | Quiet): {posterior_cat_given_quiet:.3f}")
print(f"Probability (Dog | Quiet): {posterior_dog_given_quiet:.3f}")
```

    Probability (Cat | Quiet): 0.727
    Probability (Dog | Quiet): 0.273
    

### **Activity**:

#### 1. **Spam Email Classifier with Bayes’ Formula**:
     - Create a Python function that takes the word occurrence and returns the spam probability based on Bayes' formula.

---



```python
# Given probabilities (let's say we already figured these out)
prior_spam = 0.3       # Initial guess: 30% of emails are spam
prob_free_given_spam = 0.7  # If it's spam, 70% chance it has "free"
prob_free_given_not_spam = 0.05 # If it's NOT spam, 5% chance it has "free"

# Evidence: The email contains the word "free"

# Calculate the probability of seeing "free" in any email
prob_free = (prob_free_given_spam * prior_spam) + (prob_free_given_not_spam * (1 - prior_spam))

# Apply Bayes' Theorem to find the probability of spam GIVEN "free" is present
if prob_free == 0:
    probability_is_spam = None
else:
    probability_is_spam = (prob_free_given_spam * prior_spam) / prob_free

# Print the result
if probability_is_spam is not None:
    print(f"Given that the email contains 'free', the probability of it being spam is: {probability_is_spam:.3f}")
else:
    print("It seems the word 'free' never appears based on our probabilities.")
```

    Given that the email contains 'free', the probability of it being spam is: 0.857
    

### **Lecture 3: Probability Distributions**


### **1. Uniform Distribution**

<img src="https://www.investopedia.com/thmb/V5KuQyN9rehNkaLAJjm1zs7znMI=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/uniform-distribution.asp-final-18a25d70966246ed8eec2ca04602b5d0.png" width=400>

- **Definition**: A **uniform distribution** is when all outcomes are equally likely.
  
  For example, when rolling a fair die, the probability of each number from 1 to 6 is the same: $ \frac{1}{6} $.
  
- **Manual Calculation for a Die**:
  
  **Scenario**: You roll a fair die. The probability of getting any particular outcome (e.g., 3) is the same as the others. So, the probability of getting a 3 is:
  
  $$
  P(X = 3) = \frac{1}{6}
  $$

  This is because each of the six outcomes (1, 2, 3, 4, 5, 6) has an equal probability of $ \frac{1}{6} $.

  **Output**: The probability of getting any number on a fair die is $ \frac{1}{6} $.



```python
# Python Code for Uniform Distribution (Fair Die)
import numpy as np

# Number of sides on the die
sides = 6

# Uniform probability (each outcome has equal probability)
probability = 1 / sides

# Display result
print(f"Probability of rolling any specific number: {probability}")
```

    Probability of rolling any specific number: 0.16666666666666666
    


```python
import matplotlib.pyplot as plt

# Possible results when you roll a die
results = [1, 2, 3, 4, 5, 6]

# Since it's a fair die, each result has the same chance
probability = 1 / 6

# Now, let's create the probabilities for each result
probabilities = [probability] * 6  # This makes a list like [0.166..., 0.166..., ...]

# Let's show this in a graph
plt.bar(results, probabilities)
plt.title('Uniform Distribution - Dice Roll')
plt.xlabel('Outcomes (What you can get on the die)')
plt.ylabel('Probability (How likely each outcome is)')
plt.show()
```


    
![png](output_23_0.png)
    


### **2. Normal (Gaussian) Distribution**

- **Definition**: The **Normal distribution** is a bell-shaped curve, where most of the data points are around the mean, and fewer points are found as you move farther from the mean.
  
- **Formula** for the Normal distribution:
  $$
  f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
  $$
  Where:
  - $ \mu $ is the mean (average).
  - $ \sigma $ is the standard deviation (spread of data).

**Scenario**:  
- Suppose the heights of a population of adults follow a normal distribution with a mean height \( \mu = 170 \, \text{cm} \) and a standard deviation $ \sigma = 10 \, \text{cm} $.
  
**Question**: What is the probability of a person having a height between 160 cm and 180 cm?

To find this manually, we would calculate the **z-scores** for 160 cm and 180 cm and find the area between them using the **Standard Normal Distribution**.

**Z-Score Formula**:
$$
Z = \frac{X - \mu}{\sigma}
$$

For $ X = 160 $ cm:
$$
Z_1 = \frac{160 - 170}{10} = -1
$$

For $ X = 180 $ cm:
$$
Z_2 = \frac{180 - 170}{10} = 1
$$

Using a Z-table or Python, we find that the probability between these two Z-scores is approximately 68.2%.




```python
from scipy.stats import norm

# Parameters for the Normal distribution
mu = 170
sigma = 10

# Calculate cumulative probabilities for 160 cm and 180 cm
P_160 = norm.cdf(160, mu, sigma)  # CDF at 160
# print(P_160)
P_180 = norm.cdf(180, mu, sigma)  # CDF at 180

# The probability of being between 160 and 180 cm
probability = P_180 - P_160
print(f"Probability of height being between 160 cm and 180 cm: {probability:.4f}")
```

    0.15865525393145707
    Probability of height being between 160 cm and 180 cm: 0.6827
    


```python
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# # Generate values for the x-axis (height range)
# x = np.linspace(140, 200, 1000)

# # Probability density function for Normal distribution
# y = norm.pdf(x, mu, sigma)

# # Plotting the Normal Distribution
# plt.plot(x, y)
# plt.title('Normal Distribution - Height')
# plt.xlabel('Height (cm)')
# plt.ylabel('Probability Density')
# plt.show()


import matplotlib.pyplot as plt
from scipy.stats import norm

# Let's say the average height (mean) is 170 cm
average_height = 170

# And the spread of heights (standard deviation) is 10 cm
height_spread = 10

# Generate some height values to plot (from 140 cm to 200 cm)
heights = [h for h in range(140, 201)] # This creates a list of heights

# Get the probability of each height based on the normal distribution
# Probability density function for Normal distribution
probabilities = norm.pdf(heights, average_height, height_spread)

# Plotting the Normal Distribution (the bell curve)
plt.plot(heights, probabilities)
plt.title('Normal Distribution - Height')
plt.xlabel('Height (cm)')
plt.ylabel('Probability Density (How common each height is)')
plt.show()

```


    
![png](output_26_0.png)
    


### **3. Binomial Distribution**

- **Definition**: The **Binomial distribution** models the number of successes in a fixed number of trials, where each trial has two possible outcomes (success or failure).

- **Formula**:
  $$
  P(X = x) = \binom{n}{x} p^x (1 - p)^{n-x}
  $$
  Where:
  - $ n $ is the number of trials.
  - $ x $ is the number of successes.
  - $ p $ is the probability of success on each trial.

**Scenario**:  
- You flip a fair coin 10 times. What is the probability of getting exactly 6 heads?

For this, $ n = 10 $, $ x = 6 $, and $ p = 0.5 $.

Using the binomial formula:

$$
P(X = 6) = \binom{10}{6} (0.5)^6 (0.5)^4 = 210 \times 0.5^{10} = 0.205
$$



```python
from scipy.stats import binom
import matplotlib.pyplot as plt

# Parameters for the Binomial distribution
n = 10  # Number of trials
p = 0.5  # Probability of success (head)
x = np.arange(0, n+1)  # Number of successes (heads)

# Binomial probability mass function
pmf = binom.pmf(x, n, p)

# Display result for exactly 6 heads
print(f"Probability of getting exactly 6 heads: {pmf[6]:.4f}")

# Plotting the Binomial Distribution
plt.bar(x, pmf)
plt.title('Binomial Distribution - Coin Flip')
plt.xlabel('Number of Heads')
plt.ylabel('Probability')
plt.show()

```

    Probability of getting exactly 6 heads: 0.2051
    


    
![png](output_28_1.png)
    


### **4. Poisson Distribution**

- **Definition**: The **Poisson distribution** models the number of events that occur in a fixed interval of time or space, given a known average rate.

- **Formula**:
  $$
  P(X = x) = \frac{\lambda^x e^{-\lambda}}{x!}
  $$
  Where:
  - $ \lambda $ is the average number of events in a fixed interval.
  - $ x $ is the number of events.

**Scenario**:  
- Suppose 3 cars pass a checkpoint every minute. What is the probability that exactly 5 cars pass the checkpoint in the next minute?

Using $ \lambda = 3 $ and $ x = 5 $:

$$
P(X = 5) = \frac{3^5 e^{-3}}{5!} = \frac{243 \times e^{-3}}{120} \approx 0.1008
$$



```python
from scipy.stats import poisson

# Parameters for the Poisson distribution
lambda_ = 3  # Average rate
x = 5  # Number of events (cars)

# Poisson probability mass function
pmf = poisson.pmf(x, lambda_)

# Display result
print(f"Probability of exactly 5 cars passing the checkpoint: {pmf:.4f}")

```

    Probability of exactly 5 cars passing the checkpoint: 0.1008
    


```python
x = np.arange(0, 10)
pmf = poisson.pmf(x, lambda_)

# Plotting the Poisson Distribution
plt.bar(x, pmf)
plt.title('Poisson Distribution - Cars Passing Checkpoint')
plt.xlabel('Number of Cars')
plt.ylabel('Probability')
plt.show()

```


    
![png](output_31_0.png)
    

