# 📚 Lesson 1.6: Introduction to NumPy

## Session Overview

| | |
|---|---|
| **Duration** | 3 hours |
| **Format** | Flipped Classroom + Guided Coding in Jupyter |
| **Tools** | Google Colab (recommended) or VS Code + `pds` conda environment |
| **Notebook** | `notebooks/numpy_lesson.ipynb` |

## Agenda

| Time | Part | Topic |
|------|------|-------|
| 0:00 – 0:10 | Welcome | Session goals & setup check |
| 0:10 – 1:00 | Parts 1–2 | Performance Benchmark + The ndarray |
| 1:00 – 1:05 | Break | — |
| 1:05 – 1:55 | Parts 3–5 | Arithmetic & Broadcasting + Indexing & Slicing + Boolean Indexing |
| 1:55 – 2:00 | Break | — |
| 2:00 – 2:55 | Parts 6–7 | ufuncs & Methods + Linear Algebra |
| 2:55 – 3:00 | Wrap-Up | Key Takeaways |

## 🎯 Learning Objectives

By the end of this session, you will be able to:

1. Create NumPy arrays from Python sequences and inspect key attributes (`shape`, `dtype`, `ndim`).
2. Apply indexing, slicing, and Boolean masking to filter and extract data from arrays.
3. Use broadcasting to perform efficient element-wise arithmetic between arrays of different shapes.
4. Execute statistical aggregations and basic linear algebra operations using NumPy.

---

## 🖥️ Visual Companion — open this first

> **Before working through the notebook, open the interactive visual explainer. It shows you what arrays *look like* before you start writing code.**
>
> 📂 **[Open numpy-visual-explainer.html](../../../numpy-visual-explainer.html)** — or find it in your course folder.
>
> The page has 4 tabs: Shape & ndarray · Indexing & Slicing · Boolean Masking · Views vs Copies. Spend 5–10 minutes clicking through each tab before opening the notebook. Everything in this lesson will make more sense with the visual in your head.

---

## Before You Start

**Have you completed the pre-class reading?**
- ✓ Understand what NumPy is and why it matters
- ✓ Jupyter or Colab environment is set up and working

Open the notebook: `notebooks/numpy_lesson.ipynb`

Run the first cell (`import numpy as np` and print the version) to confirm your environment is ready.

---

> ## 🔤 Plain-English Jargon Buster
>
> Read this once before diving into the notebook. These 8 terms appear constantly — if you know what they mean in plain English, the syntax becomes much easier to follow.
>
> | Term | Plain-English meaning |
> |---|---|
> | **ndarray** | N-dimensional array. "N" just means it can have 1, 2, or more dimensions. For data work, 2D (a table of rows and columns) is most common. |
> | **shape** | A tuple showing the size of each dimension. `(4, 5)` = 4 rows, 5 columns. Always read left-to-right: first number = rows, second = columns. |
> | **dtype** | Data type — what kind of value is stored in every cell. `int64` = whole numbers. `float64` = decimals. All cells in an array must be the same type. |
> | **ndim** | Number of dimensions. A flat list = 1. A table = 2. You'll use 2D arrays most of the time. |
> | **axis 0** | The direction *down* the rows. Operations on `axis=0` collapse the rows — giving one result per column. Think: "axis 0 goes along dimension 0 (rows)." |
> | **axis 1** | The direction *across* the columns. Operations on `axis=1` collapse the columns — giving one result per row. |
> | **broadcasting** | NumPy's automatic scaling rule. When you do `array * 2`, NumPy applies `* 2` to every element automatically — no loop needed. When shapes don't match exactly, NumPy stretches the smaller array to fit. |
> | **view** | A slice of an array that shares the same memory as the original. Modifying the view modifies the original. Use `.copy()` when you want an independent duplicate. |

---

## 🏃 Part 1: Performance Benchmark

**Why this Part exists:** Before learning NumPy's syntax, it helps to see *why* anyone bothers learning a new library. This benchmark makes the performance difference visceral.

Notebook section: **"Part 1: Performance Benchmark"**

- Run the `%timeit` comparison between a NumPy vectorized multiply and a Python list comprehension on 1,000,000 elements.
- Observe the timing results.

> **Quick activity:** Run the timing cell and describe what you observe. How much faster is NumPy? What does this mean for data science at scale?

**Why is NumPy faster?** A Python list stores each number as a separate object scattered around memory — to multiply a list, Python visits each object one by one. A NumPy array stores all numbers in a single continuous block of memory of one fixed type, so the underlying C/Fortran code can process the entire block in one sweep. At 1 million elements, this difference is typically 100× or more.

---

## 🏃 Part 2: The ndarray (N-dimensional array)

**Why this Part exists:** Python lists work fine for 10 items. For 10 million numbers, you need a data structure designed for math — with a fixed type, a declared shape, and no Python overhead. That's the ndarray.

Notebook section: **"Part 2: The ndarray"**

**Before the notebook, picture this:**

```
A 2D array with shape (3, 4) looks like a 3-row, 4-column grid:

         col 0   col 1   col 2   col 3
row 0  [  10      20      30      40  ]
row 1  [  50      60      70      80  ]
row 2  [  90     100     110     120  ]

arr.shape  →  (3, 4)   ← 3 rows, 4 columns
arr.ndim   →  2        ← two dimensions
arr.dtype  →  int64    ← whole numbers
arr.size   →  12       ← total cells (3 × 4)
```

**Key concepts in this section:**
- Creating arrays from Python sequences — why `np.array([1, 2, 3])` vs a plain list
- `shape`, `dtype`, `ndim` — understanding what you have before you analyse it
- Casting with `astype` — why you sometimes need to change dtype (e.g., `float32` uses half the memory of `float64`, which matters for large datasets)

### 🛠️ Exercise 1: Creation & Casting

- Create a 3×4 array of all ones using `np.ones()`
- Cast this array to `float32`
- Create an array of strings representing numbers and cast to `float`

<details>
<summary>💡 Solution — try it yourself first!</summary>

```python
import numpy as np

# 3×4 array of ones
arr = np.ones((3, 4))
print(arr.shape, arr.dtype)  # (3, 4) float64

# Cast to float32 (half the memory)
arr_f32 = arr.astype(np.float32)
print(arr_f32.dtype)  # float32

# String array cast to float
string_nums = np.array(['1.5', '2.7', '3.1'])
floats = string_nums.astype(float)
print(floats)  # [1.5 2.7 3.1]
```

**Why cast to float32?** For large arrays (think millions of rows), switching from float64 to float32 halves your memory usage with negligible precision loss for most data science tasks.

</details>

---

## 🏃 Part 3: Arithmetic & Broadcasting

**Why this Part exists:** The whole point of NumPy is to do math on many numbers at once — without writing loops. Broadcasting is the rule that makes this work even when two arrays have different shapes.

Notebook section: **"Part 3: Arithmetic & Broadcasting"**

**Before the notebook, picture this:**

```
Element-wise arithmetic — every cell pairs with the matching cell:

[10, 20, 30] * [2, 3, 4]  →  [20, 60, 120]   ← position 0 pairs with 0, etc.
```

```
Broadcasting — a scalar stretches to match the array:

[10, 20, 30] * 2  →  [20, 40, 60]   ← NumPy automatically applies * 2 to every cell
```

**Why does broadcasting exist?** Without it, you'd need a loop: `for x in arr: x * 2`. With broadcasting, one line does the same thing — and 100× faster. The rule: if dimensions don't match, NumPy stretches the smaller array along the size-1 dimensions.

Key concepts:
- `arr * arr` — element-wise multiplication (same shape required, or broadcastable)
- `1 / arr` — scalar broadcasts across every cell
- `arr + scalar`, `arr - mean` — the most common data normalisation pattern

> **Discussion:** How does broadcasting make code cleaner and more efficient? What would the loop equivalent look like?

---

## 🏃 Part 4: Indexing and Slicing

**Why this Part exists:** Selecting specific rows, columns, or cells is the most basic data operation. NumPy's 2D indexing gives you precise control in one expression.

Notebook section: **"Part 4: Indexing and Slicing"**

**Before the notebook, picture this:**

```
For a 2D array, indexing uses [row, col] syntax:

arr[1, 2]      →  single value at row 1, column 2
arr[0:2, :]    →  rows 0 and 1, all columns (2D result)
arr[:, 1:4]    →  all rows, columns 1, 2, 3 (stop=4 is excluded)
arr[2, :]      →  all of row 2 as a 1D array
```

**The stop index is excluded** — this matches Python list slicing. `0:3` gives indices 0, 1, 2.

**Critical: slices are views, not copies.** When you do `sub = arr[0:2, :]`, `sub` is a window into the original array's memory. If you modify `sub`, you modify `arr`. Use `.copy()` to get an independent duplicate:

```python
safe = arr[0:2, :].copy()   # now modifying safe will NOT affect arr
```

---

## 🏃 Part 5: Boolean Indexing

**Why this Part exists:** Selecting by position is useful, but selecting by *condition* is what you do in real analysis — "give me all rows where sales > 100", "flag all readings outside the normal range."

Notebook section: **"Part 5: Boolean Indexing"**

A comparison like `arr > 50` produces a Boolean array (`True`/`False` per cell). Using that Boolean array as an index returns only the cells where the value is `True`.

```python
names  = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
scores = np.array([[75, 80], [85, 90], [95, 100], [100, 77], [85, 92], [95, 80], [72, 80]])

bob_mask = (names == 'Bob')           # [True, False, False, True, False, False, False]
print(scores[bob_mask])               # rows where name is 'Bob'
```

Key points:
- The boolean mask must have the same length as the dimension you are indexing
- Combine conditions with `&` (AND) and `|` (OR) — wrap each condition in parentheses
- Assigning to a boolean-indexed slice modifies the original array in place

### 🛠️ Exercise 3: Complex Filtering

Using the `names` and `scores` arrays from the demo cell:

1. Select all scores where the name is NOT `'Bob'` (use `!=` or `~`)
2. Select scores for `'Bob'` or `'Will'` using the `|` operator
3. Find all scores less than 80 and set them to 0

<details>
<summary>💡 Solution — try it yourself first!</summary>

```python
# 1. Scores where name is NOT 'Bob'
print(scores[names != 'Bob'])

# 2. Scores for 'Bob' or 'Will'
mask = (names == 'Bob') | (names == 'Will')
print(scores[mask])

# 3. Set all scores below 80 to 0 — this modifies the array in place
scores[scores < 80] = 0
print(scores)
```

**Why parentheses around each condition?** Python's `&` operator has higher precedence than `==`, so without parentheses, `names == 'Bob' | names == 'Will'` is parsed incorrectly and gives a cryptic error.

</details>

---

## 🏃 Part 6: Universal Functions (ufuncs) and Methods

**Why this Part exists:** NumPy has a family of pre-built mathematical functions — called ufuncs — that apply to every element automatically, without a loop. These are the functions you'll use to calculate statistics, normalise data, and prepare features for models.

Notebook section: **"Part 6: Universal Functions (ufuncs) and Methods"**

**What is a ufunc?** "Universal Function" — a function that automatically applies element-by-element across an entire array. `np.sqrt(arr)` calculates the square root of *every* value in `arr` in one call.

**The axis argument — which direction to collapse:**

```
arr.mean(axis=0)  →  one average per COLUMN (collapses down the rows ↓)
arr.mean(axis=1)  →  one average per ROW (collapses across the columns →)
arr.mean()        →  single average across ALL values
```

Memory tip: axis=0 is "along the rows" (so the result has one value per column). axis=1 is "along the columns" (so the result has one value per row).

Key concepts:
- Unary ufuncs: `np.sqrt()`, `np.exp()`, `np.abs()` — apply to every element
- Binary ufuncs: `np.add()`, `np.maximum()` — combine two arrays element-by-element
- Statistical methods: `mean()`, `sum()`, `std()`, `min()`, `max()` — with the `axis` argument

> **Discussion:** When should you use ufuncs vs. loops? (Answer: always ufuncs if one exists — they're pre-compiled C code and 10–100× faster.)

---

## 🏃 Part 7: Linear Algebra

**Why this Part exists:** Most ML models — from linear regression to neural networks — are just matrix operations under the hood. Understanding `*` vs `@` prevents the most common source of silent wrong results in ML code.

Notebook section: **"Part 7: Linear Algebra"**

**The critical distinction:**

```python
A * B    # Element-wise: every cell pairs with its matching cell
A @ B    # Matrix multiplication: the mathematical dot product

Example — 2×2 matrices:
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

A * B = [[1×5, 2×6], [3×7, 4×8]] = [[5, 12], [21, 32]]   ← element-wise
A @ B = [[1×5+2×7, 1×6+2×8], [3×5+4×7, 3×6+4×8]] = [[19, 22], [43, 50]]  ← matrix multiply
```

These produce completely different results. Getting them confused causes code that runs without error but produces wrong answers — the most dangerous kind of bug.

### 🛠️ Exercise 4: Reshaping & Statistics

- Create an array of 15 integers using `arange(15)` and reshape it to `(3, 5)`
- Calculate the average value of each row (which axis?)
- Use `np.unique()` to find distinct elements
- Transpose the reshaped array using `.T` and check the new shape

<details>
<summary>💡 Solution — try it yourself first!</summary>

```python
import numpy as np

arr = np.arange(15).reshape(3, 5)
# [[0, 1, 2, 3, 4],
#  [5, 6, 7, 8, 9],
#  [10, 11, 12, 13, 14]]

# Average per row → axis=1 (collapses columns, gives one result per row)
row_means = arr.mean(axis=1)
print(row_means)  # [2.0, 7.0, 12.0]

# Unique elements
print(np.unique(arr))  # [0, 1, 2, ..., 14] — same as input since all distinct

# Transpose — rows become columns
arr_T = arr.T
print(arr_T.shape)  # (5, 3) — was (3, 5)
```

**Why `axis=1` for row averages?** We want one number per row — that means we're collapsing (averaging) across the columns. Collapsing across columns = axis=1.

</details>

---

## 🎯 Wrap-Up

**Key Takeaways:**
1. NumPy arrays are the foundation — Pandas, scikit-learn, and TensorFlow all build on NumPy's ndarray.
2. Slices are **views**, not copies — use `.copy()` when you need independence from the source array.
3. `@` for matrix multiplication, `*` for element-wise — getting these confused causes silent wrong results.
4. `axis=0` collapses rows (result is per column); `axis=1` collapses columns (result is per row).

**Next Steps:**
- Complete the [Assignment](./assignment.md) — 7 NumPy practice exercises covering arrays, masking, broadcasting, and linear algebra.
- Next lesson: Lesson 1.7 introduces Pandas — a higher-level library built on NumPy that adds labelled indexing, mixed data types, and table-like operations.
