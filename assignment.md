This post-class practice helps you reinforce the core NumPy skills we used:

- Creating and inspecting arrays  
- Indexing, slicing, and boolean filtering  
- Aggregations over rows/columns  
- Reshaping and simple matrix operations

You can work through these in the same environment you used in class (Colab or VS Code).

---

## 1. Warm-up: Rebuild the basics

**Goal:** Make array creation and inspection feel automatic.

1. Create each of the following arrays and print its `shape`, `ndim`, and `dtype`:
   - A 1D array of integers from 10 to 19  
   - A 2D array of shape `(4, 3)` filled with 1.5  
   - A 3D array of zeros with shape `(2, 2, 3)`

2. Convert the `(4, 3)` float array to integers using `.astype(int)`.  
3. Write down (in a markdown cell) when you would prefer `float` vs `int` in real analyst work (e.g., prices vs counts).

---

## 2. Sales table drill: Indexing and slicing

**Scenario:** You have quarterly sales numbers for 4 regions (rows) over 5 quarters (columns).

```python
import numpy as np

sales = np.array([
    [120, 135, 110, 145, 130],  # Region A
    [ 90,  85,  95, 100, 110],  # Region B
    [200, 215, 190, 225, 245],  # Region C
    [155, 140, 135, 160, 170]   # Region D
])
regions = np.array(["A", "B", "C", "D"])
quarters = np.array(["Q1", "Q2", "Q3", "Q4", "Q5"])
```

Do the following:

1. Select all quarters for **Region B** as a 1D array.  
2. Select **Q2 to Q4** (inclusive) for all regions as a 2D subarray.  
3. Select **Q5** sales for **Regions A and D only** (use slicing or fancy indexing).  
4. Compute the **shape** of each result and add a brief comment: “Is this 1D or 2D, and why?”

---

## 3\. Boolean masks: Work with targets

**Scenario:** You have campaign response data.

```py
names = np.array(["Ana", "Ben", "Chen", "Dana", "Eli", "Fatima", "George", "Hui"])
spend = np.array([200, 150, 300, 120, 180, 220, 160, 310])      # marketing spend
revenue = np.array([400, 180, 500, 100, 220, 260, 150, 600])   # revenue
```

1. Compute the **ROI** for each person: `roi = revenue / spend`.  
2. Create a boolean mask for customers with `roi >= 2.0`.  
3. Use the mask to:  
   - List their names  
   - List their spend and revenue  
4. Create a second mask for customers with `spend >= 200`.  
5. Combine the two masks to find customers who have **roi \>= 2.0 AND spend \>= 200**.  
6. In a markdown cell, answer: *“What business insight do you get from this filtered group?”*

---

## 4\. Gradebook: Aggregations and broadcasting

Reuse the “Gradebook” idea from class and extend it.

```py
students = np.array(["S1", "S2", "S3", "S4", "S5"])
subjects = np.array(["Math", "Stats", "Python"])

scores = np.array([
    [75, 80, 85],
    [60, 65, 70],
    [90, 88, 92],
    [82, 79, 84],
    [70, 72, 78]
])
```

1. Compute each student’s **average score** (per row).  
2. Compute each subject’s **average score** (per column).  
3. Create `scores_centered` by subtracting the **subject mean** from each column (broadcasting).  
4. For `scores_centered`, compute per-student averages again.  
5. Compare which student looks best by **raw average** vs **centered average**.  
6. In a markdown cell, explain briefly why centered scores might give a fairer comparison.

---

## 5\. Reshape and flatten: From daily to weekly

**Scenario:** You have 30 days of daily website visits, and you want to summarize them by week.

```py
daily_visits = np.array([
    120, 130, 125, 140, 150,
    160, 170, 155, 145, 135,
    128, 132, 138, 142, 148,
    152, 158, 162, 168, 172,
    180, 190, 185, 175, 165,
    155, 145, 135, 125, 115
])
```

1. Reshape `daily_visits` into a `(6, 5)` array representing **6 weeks × 5 days**.  
2. Compute **total visits per week**.  
3. Compute **average visits per day of the week** (e.g., all “day 1 of week” together, all “day 2 of week” together, etc.).  
4. Flatten the reshaped array back to 1D and confirm it matches the original `daily_visits`.

Hint: Pay attention to how reshaping groups the data; mention any assumptions you’re making about which days belong to which week.

---

## 6\. Mini-project: Simple scoring model

This mirrors the tiny matrix multiply example from class, but with more features.

```py
# Each row: [page_views, time_on_site (minutes), past_purchases]
X = np.array([
    [10,  3.5,  0],
    [25,  5.0,  1],
    [40,  2.0,  0],
    [15, 10.0,  3],
    [30,  4.0,  2]
])

customers = np.array(["C1", "C2", "C3", "C4", "C5"])
```

1. Choose a weight vector `w = [w_views, w_time, w_purchases]` (for example `[0.1, 0.5, 1.0]`).  
2. Compute a **score** for each customer using `scores = X @ w`.  
3. Rank customers by score (highest first).  
4. Change the weights to emphasize **past\_purchases** more than other features, and recompute.  
5. In a markdown cell, answer:  
   - Which customer is top-ranked before vs after changing weights?  
   - In what real-world situation might you prefer each weighting?

---

## 7\. Stretch ideas (optional)

If you’re comfortable with the above, try:

- Generate 1,000 random test scores with `np.random.randn`, scale them to have mean 70 and standard deviation 10, then:  
  - Clip scores to the range `[0, 100]`  
  - Compute min, max, mean, and standard deviation  
- Simulate a small A/B test:  
  - Two groups of 20 users each  
  - Randomly generate conversions (0 or 1\) for each group  
  - Compute conversion rate per group using pure NumPy operations

---

## How to get help

If you get stuck:

- Add a markdown cell describing what you tried and what you expected vs what happened.
- Post your notebook in **#questions** on Discord for help, or in **#peer-reviews** to share your work.

Practising these patterns now will make later topics (like Pandas and modelling) feel much more natural.

---

## ✅ Solutions

*Attempt all exercises before expanding the solutions below.*

<details>
<summary>Click to reveal all solutions</summary>

## Solution

## **1: Warm-up: Rebuild the Basics**

### **1\. Array Creation & Inspection**

**Why `arange` instead of `range`?** `np.arange` returns a NumPy array (not a Python list), so it already supports fast math operations. `range` returns a lazy iterator — you'd have to convert it with `np.array(range(...))`.

**Why check `shape`, `ndim`, and `dtype` first?** These three attributes give you a complete picture of what you're working with before touching the data. Missing this step is like opening a spreadsheet without knowing how many columns it has.

```
# 1D array 10 to 19
arr_1d = np.arange(10, 20)

# 2D array (4, 3) filled with 1.5
arr_2d = np.full((4, 3), 1.5)

# 3D array of zeros (2, 2, 3)
arr_3d = np.zeros((2, 2, 3))

for a in [arr_1d, arr_2d, arr_3d]:
    print(f"Shape: {a.shape}, Dim: {a.ndim}, Dtype: {a.dtype}")

# 2. Convert 2D float to int
arr_2d_int = arr_2d.astype(int)
print("\nConverted to int:\n", arr_2d_int)
```

### **3\. Analyst Reflection: Float vs Int**

* **Use `float`** for metrics requiring precision or where division is involved, such as **stock prices, temperature readings, or conversion rates**.  
* **Use `int`** for discrete units that cannot be subdivided, such as **website visitor counts, number of units sold, or user IDs**.

## **Part 2: Sales Table Drill**

**Why does selecting a single row give a 1D result?** When you pick one row (`sales[1, :]`), NumPy removes that dimension — you asked for one row, so the "rows" dimension collapses and you get a flat 1D array. This is called "dimension reduction." If you want to keep it 2D (e.g., shape `(1, 5)` instead of `(5,)`), use `sales[1:2, :]` (a slice of one row, not a single index).

**Why does `sales[[0, 3], 4]` give a 1D array?** You passed a list of row indices and a single column — so the result is a 1D array of the values at (row 0, col 4) and (row 3, col 4). This is fancy indexing — NumPy picks the elements in the order you listed them.

```
sales = np.array([
    [100, 120, 110, 150, 130], # Region A
    [90, 80, 95, 100, 110],    # Region B
    [200, 210, 190, 220, 250], # Region C
    [150, 140, 130, 160, 170]  # Region D
])

# 1. Select Region B (Row index 1)
region_b = sales[1, :] 
# Result: 1D. We selected a single row across all columns.

# 2. Select Q2 to Q4 for all regions (Cols 1 to 4)
q2_q4 = sales[:, 1:4]
# Result: 2D. We selected multiple rows and multiple columns.

# 3. Q5 sales for Regions A and D (Row 0 & 3, Col 4)
a_d_q5 = sales[[0, 3], 4]
# Result: 1D. We extracted specific scalar values into a new vector.

print("Region B Shape:", region_b.shape)
print("Q2-Q4 Subarray Shape:", q2_q4.shape)
print("A & D Q5 Shape:", a_d_q5.shape)
```

## **Part 3: Boolean Masks**

```
names = np.array(["Ana", "Ben", "Chen", "Dana", "Eli", "Fatima", "George", "Hui"])
spend = np.array([100, 250, 50, 300, 150, 400, 120, 210])
revenue = np.array([250, 400, 120, 800, 200, 900, 150, 500])

# 1. Compute ROI
roi = revenue / spend

# 2. Mask for ROI >= 2.0
high_roi_mask = roi >= 2.0

# 3. Use mask
print("High ROI Customers:", names[high_roi_mask])
print("Their Spend:", spend[high_roi_mask])

# 4. Combined mask (ROI >= 2.0 AND spend >= 200)
high_value_mask = (roi >= 2.0) & (spend >= 200)
print("High Value/High Efficiency:", names[high_value_mask])
```

**Business Insight:** Customers in the filtered group (ROI \>= 2.0 AND Spend \>= 200\) represent our **Scalable Successes**. These are accounts where we spent a significant amount of money and still saw a high rate of return. We should prioritize these profiles for future budget increases.

## **Part 4: Gradebook Aggregations**

**Why `axis=1` for student averages and `axis=0` for subject averages?**
- Each student is a row. To get one average *per student*, we collapse across the columns → `axis=1`.
- Each subject is a column. To get one average *per subject*, we collapse down the rows → `axis=0`.

**Why does centering work with broadcasting?** `scores - subject_avg` subtracts a 1D array (shape `(3,)`) from a 2D array (shape `(5, 3)`). NumPy broadcasts the 1D array across all 5 rows automatically — each column gets its own subject mean subtracted. This is the standard way to normalise data.

```
scores = np.array([
    [85, 90, 78], # S1
    [70, 80, 85], # S2
    [95, 92, 98], # S3
    [60, 65, 60], # S4
    [88, 85, 90]  # S5
])

# 1. Student Avg (Row mean)
student_avg = scores.mean(axis=1)

# 2. Subject Avg (Col mean)
subject_avg = scores.mean(axis=0)

# 3. Centering (Broadcasting)
scores_centered = scores - subject_avg

# 4. Compare
print("Raw Averages:", student_avg)
print("Centered Averages:", scores_centered.mean(axis=1))
```

**Why centered scores?** Centering scores provides a fairer comparison because it accounts for the **difficulty of the subject**. If the "Stats" test was much harder than the "Python" test, a student's raw average might look low simply because they took a hard class. Centering shows how they performed relative to the class average for each specific subject.

## **Part 5: Reshape & Flatten**

```
daily_visits = np.array([
    120, 130, 125, 140, 150, 160, 170, 155, 145, 135,
    128, 132, 138, 142, 148, 152, 158, 162, 168, 172,
    180, 190, 185, 175, 165, 155, 145, 135, 125, 115
])

# Reshape (6 weeks, 5 days - assuming a 5-day work week)
weekly_view = daily_visits.reshape(6, 5)

# Total visits per week
total_per_week = weekly_view.sum(axis=1)

# Average visits per day of the week (Avg of all Day 1s, Day 2s, etc.)
avg_day_of_week = weekly_view.mean(axis=0)

# Flatten
original_shape = weekly_view.flatten()
print("Back to original match?", np.array_equal(daily_visits, original_shape))
```

**Assumption:** We assume the first 5 elements are Week 1, the next 5 are Week 2, and so on. If the data had weekends included but we reshaped to 5 columns, our "weeks" would drift chronologically.

## **Part 6: Mini-project: Scoring Model**

**Why `X @ w` instead of a loop?** `X @ w` is matrix-vector multiplication. For each customer (row), it multiplies the three feature values by the three weights and adds them up — in one expression, for all 5 customers simultaneously. The loop equivalent would be 5 iterations, each doing 3 multiplications and 2 additions. `@` does it all at once in optimised C code.

**Why does changing weights change rankings?** A weight is how much you "care about" each feature. `w_purchases = 5.0` means past purchases count 50× more than page views. Customers who bought before will leap to the top of the ranking. This is the foundation of feature engineering in machine learning — the weights are literally what the model learns during training.

```
X = np.array([
    [10,  3.5,  0],
    [25,  5.0,  1],
    [40,  2.0,  0],
    [15, 10.0,  3],
    [30,  4.0,  2]
])
customers = np.array(["C1", "C2", "C3", "C4", "C5"])

# Weight set 1: Balanced
w1 = np.array([0.1, 0.5, 1.0])
scores1 = X @ w1

# Weight set 2: Heavy on Purchases
w2 = np.array([0.05, 0.1, 5.0])
scores2 = X @ w2

print("Balanced Rankings:", customers[np.argsort(-scores1)])
print("Purchase-heavy Rankings:", customers[np.argsort(-scores2)])
```

**Analysis:**

* **Weighting preference:** `w1` (Balanced) is better for identifying **engaged** users who might be new (high time on site). `w2` (Purchase-heavy) is better for a **loyalty program** where we want to reward historical spend above all else.

## **Part 7: Stretch Idea (A/B Test)**

```
# Group sizes
n = 20

# Generate conversions (0 or 1)
# Using 0.3 probability for A and 0.4 for B
group_a = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
group_b = np.random.choice([0, 1], size=n, p=[0.6, 0.4])

print(f"Group A Conv Rate: {group_a.mean():.2%}")
print(f"Group B Conv Rate: {group_b.mean():.2%}")
```

</details>
