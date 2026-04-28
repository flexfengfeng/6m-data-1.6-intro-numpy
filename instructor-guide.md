# 🎓 Instructor Guide — Lesson 1.6: Introduction to NumPy

> **Branch:** `feature/instructor-guide`
> **Audience:** Instructors and teaching assistants
> **Companion to:** `lesson.md`, `pre-class.md`, `assignment.md`

---

## 1. Lesson Overview & Instructor Objectives

| | |
|---|---|
| **Duration** | 3 hours |
| **Format** | Flipped Classroom + Guided Coding in Jupyter/Colab |
| **Notebook** | `notebooks/numpy_lesson.ipynb` |
| **Learner entry point** | Completed SQL track (1.3–1.5); basic Python familiarity; no NumPy experience |

By the end of this lesson learners should be able to:

1. Create, inspect, and cast NumPy arrays (`shape`, `dtype`, `ndim`, `astype`).
2. Perform element-wise arithmetic and explain how broadcasting works.
3. Index and slice 1D and 2D arrays, understanding the view-vs-copy distinction.
4. Apply statistical aggregations and basic linear algebra operations.

**Instructor's primary job:** This lesson is the conceptual bridge between Python lists (familiar) and the numerical computing paradigm (new). The performance benchmark in Part 1 is your best hook — make it visceral. If learners understand *why* NumPy exists (speed, memory efficiency, vectorisation), every subsequent concept makes sense.

---

## 2. Concept Analogies

### NumPy vs. Python Lists — "The Cargo Ship vs. the Rowing Boat"

> "A Python list is a rowing boat — great for a few passengers, flexible, you can put different things in it. NumPy is a cargo ship — built for one type of cargo, loads it in parallel across the whole hold, and crosses the ocean in a fraction of the time."

The performance benchmark in Part 1 (1,000,000 element multiply) is the proof. Let learners run it themselves and share their timing results. Typical result: NumPy is 100–1000x faster. Ask: "If you're training a machine learning model on a million rows, which would you use?"

**Why the speed difference?**
- Python lists are arrays of *pointers* to objects scattered in memory. Iterating means following each pointer separately.
- NumPy arrays store values in *contiguous memory blocks* of a single type. Operations are applied in C or FORTRAN underneath — not Python loops.

---

### The ndarray — "The Spreadsheet That Knows Its Type"

> "Think of a NumPy array as a spreadsheet column that only accepts one data type. You declare it once — 'this is a column of integers' — and the computer reserves exactly the right amount of memory for every cell. No wasted space, no type uncertainty."

**Shape as rows × columns:** For 2D arrays, `shape` reads as `(rows, columns)`. Help learners see this by mapping it to a real table:
- `np.zeros((3, 4))` — a 3-row, 4-column table, all zeros.
- `array.shape` → `(3, 4)` — 3 rows, 4 columns.

**The dtype conversation:** Ask learners: "Why would you ever want `float32` instead of `float64`?" → Half the memory. For large datasets (images, medical scans, ML training data), switching from float64 to float32 halves your memory usage with minimal precision loss.

---

### Broadcasting — "Automatic Scaling"

> "Broadcasting is NumPy's superpower. If you multiply a 4-row array by a single number, NumPy automatically 'broadcasts' that number across every element — no loop needed. It's like telling a photocopier 'enlarge every page by 20%' — you set the rule once and it applies everywhere."

**The shop discount example:** `prices * 0.9` — if `prices` is an array of 1,000 product prices, this applies a 10% discount to all 1,000 in one line of code. In a Python loop, this would be 1,000 iterations.

**The shape compatibility rules** (briefly): Broadcasting works when arrays have compatible shapes — either the same dimension size, or one of them is 1 (scalars). Don't go deep into the rules; instead, show it working and failing, and ask learners to predict which will broadcast and which will error.

---

### Views vs. Copies — "The Mirror vs. the Photograph"

> "When you slice a NumPy array, you get a *mirror*, not a photograph. Changes to the mirror change the original. If you want a photograph — an independent copy — you have to say `.copy()` explicitly."

**The danger:** This is a source of subtle bugs that are hard to find. If a learner slices an array, modifies the slice, and is surprised that the original changed — they've been bitten by the view behaviour. Show it live: modify a slice, print the original, show the change propagated.

**When does it matter?** When you're transforming data and want to keep the original intact for comparison or rollback. The professional habit: use `.copy()` when in doubt.

---

### Matrix Multiplication — "@ vs. *"

> "`*` is element-wise — multiply each cell with its counterpart. `@` is matrix multiplication — the mathematical dot product. They look similar but produce completely different results."

Use a 2×2 example on the whiteboard to show both operations side-by-side. This is the kind of confusion that causes silent wrong results in ML code — the model trains "successfully" but on incorrect numbers.

---

## 3. Real-World Use Cases

### Why NumPy Exists — The Machine Learning Connection

Every major ML library (Pandas, scikit-learn, TensorFlow, PyTorch) uses NumPy arrays internally. When you train a model, your data is stored as a NumPy array. The model's weights are NumPy arrays. The predictions are NumPy arrays. Understanding NumPy is understanding the foundation of the entire Python data science stack.

### Image Processing

A grayscale image is a 2D NumPy array of pixel values (0–255). A colour image is a 3D array (height × width × 3 channels). Operations like brightness adjustment, cropping, and blurring are all NumPy operations:
- Brightness: `image_array * 1.2`
- Contrast: `(image_array - mean) * factor + mean`
- Crop: `image_array[100:400, 200:600]` (row slice, column slice)

### Financial Modelling

Portfolio calculations at investment banks use NumPy:
- Daily returns: `np.diff(prices) / prices[:-1]`
- Covariance matrix: `np.cov(returns_matrix)`
- Matrix multiplication for portfolio weights: `weights @ covariance_matrix @ weights.T`

The `@` operator in NumPy directly replaces pages of financial algebra.

---

## 4. Activity Facilitation Notes

### Part 1: Performance Benchmark (5–10 min)

**Run it before explaining anything.** The timing result *is* the explanation. After learners see the result, ask:
- "What does this number mean for your day-to-day work?"
- "If you're cleaning a million-row dataset and doing 10 operations — how much time are you saving?"

If the result is underwhelming (sometimes happens in Colab): the comparison still holds. Even if it's "only" 5x faster, at scale that's the difference between a 1-hour job and a 12-minute job.

---

### Part 2: ndarray Creation & Casting

**Reinforce shape reading:** Before any exercise, show a 2D array and ask: "How many rows? How many columns?" Do this with 2–3 different arrays. Building the habit of reading shape correctly prevents hours of debugging later.

**Exercise 1 setup check:** Confirm every learner can run `import numpy as np` successfully before moving on. Environment issues surface here and are better resolved early.

**The `astype` cast:** Show what happens when you cast a float array to int — the decimal is *truncated*, not rounded. `np.array([1.9, 2.7]).astype(int)` → `[1, 2]`. Ask: "Is this what you expected? When would this silent truncation cause a data problem?"

---

### Part 3: Arithmetic & Broadcasting

**Pair exercise:** Have learners write `arr * arr` (element-wise square) and then predict what `arr * 2` will do before running it. Prediction → run → verify. This "predict then confirm" loop builds intuition faster than just watching demonstrations.

**When broadcasting fails:** Intentionally create a shape mismatch error. Read the error message aloud: "operands could not be broadcast together with shapes (3,4) (3,5)." Ask: "What does this tell you about what NumPy needs?" → Compatible dimensions.

---

### Part 4: Indexing & Slicing

**The 2D indexing notation:** `array[row, col]` vs. Python list `array[row][col]`. They produce the same result, but `array[row, col]` is NumPy's preferred form — more efficient and more expressive.

**Exercise 3 (complex filtering):**

The Boolean mask approach (`array[mask]`) is the foundational skill for all future data filtering in Pandas. Spend time here. Ask: "How is this different from a WHERE clause in SQL?" → Conceptually the same — filter rows based on a condition — but expressed as an array operation rather than a string query.

**The view mutation demonstration:** Run the view-modification example live. Let learners predict whether the original will change. Most will be wrong. The surprise is the lesson.

---

### Part 7: Linear Algebra

**Keep this section brief for non-mathematical learners.** The key takeaway is: `*` ≠ `@`. Show the 2×2 example, verify learners understand the difference, move on. The mathematical proof of why matrix multiplication works the way it does is outside the scope of this lesson.

**The reshape + statistics exercise** is more practically relevant: create a 3×5 array (e.g., 3 products × 5 months of sales), calculate `axis=1` mean (average per product) vs. `axis=0` mean (average per month). This maps directly to what learners will do in Pandas EDA.

---

## 5. Timing & Pacing Notes

| Part | Planned | Common Overrun | Mitigation |
|------|---------|---------------|-----------|
| Parts 1–2: Benchmark + ndarray | 50 min | Environment setup issues eat 10–15 min | Have a pre-loaded Colab link ready as a fallback |
| Parts 3–4: Arithmetic + Indexing | 50 min | Boolean masking exercise can run long | Treat Exercise 3 as the "You do" phase — give 10 min then debrief together |
| Parts 6–7: ufuncs + Linear Algebra | 55 min | Linear algebra section confuses non-math learners | Cut the formal linear algebra theory; focus on `*` vs. `@` distinction and the axis aggregation exercise |

---

## 6. Common Learner Questions

**Q: "If NumPy is so much faster, why use Python at all?"**
A: Python provides the glue — the logic, control flow, and expressiveness. NumPy provides the performance for numerical operations. They're complementary. Python without NumPy for data science is possible but painfully slow; NumPy without Python's flexibility would be too rigid.

**Q: "Is a Pandas DataFrame just a NumPy array with labels?"**
A: Conceptually yes — Pandas was originally built on top of NumPy. A DataFrame is essentially a collection of NumPy arrays with labels (column names and an index). This is why NumPy operations often work directly on Pandas columns.

**Q: "When would I use NumPy directly vs. Pandas?"**
A: NumPy for pure numerical computing — matrix operations, linear algebra, signal processing, image manipulation. Pandas when you need labelled data, mixed types, or table-like operations. In practice, you'll use both: Pandas for loading/cleaning, NumPy operations on the numerical columns.

**Q: "What is `axis=0` vs. `axis=1`?"**
A: `axis=0` collapses rows — you get one result per column (e.g., average of each column). `axis=1` collapses columns — you get one result per row (e.g., average of each row). A helpful shorthand: `axis=0` is "along the rows" (top-to-bottom), `axis=1` is "along the columns" (left-to-right).
