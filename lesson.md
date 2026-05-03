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

## Before You Start

**Have you completed the pre-class reading?**
- ✓ Understand what NumPy is and why it matters
- ✓ Jupyter or Colab environment is set up and working

Open the notebook: `notebooks/numpy_lesson.ipynb`

Run the first cell (`import numpy as np` and print the version) to confirm your environment is ready.

---

## 🏃 Part 1: Performance Benchmark

Notebook section: **"Part 1: Performance Benchmark"**

- Run the `%timeit` comparison between a NumPy vectorized multiply and a Python list comprehension on 1,000,000 elements.
- Observe the timing results.

> **Quick activity:** Run the timing cell and describe what you observe. How much faster is NumPy? What does this mean for data science at scale?

---

## 🏃 Part 2: The ndarray (N-dimensional array)

Notebook section: **"Part 2: The ndarray"**

- Creating arrays from Python sequences
- Array attributes: `shape`, `dtype`, `ndim`
- Data types and casting using `astype`

### 🛠️ Exercise 1: Creation & Casting

- Create a 3×4 array of all ones using `np.ones()`
- Cast this array to `float32`
- Create an array of strings representing numbers and cast to `float`

---

## 🏃 Part 3: Arithmetic & Broadcasting

Notebook section: **"Part 3: Arithmetic & Broadcasting"**

- Element-wise arithmetic operations (addition, multiplication, etc.)
- Broadcasting: how scalars and arrays of different shapes work together
- Examples: `arr * arr`, `1 / arr`

> **Discussion:** How does broadcasting make code cleaner and more efficient?

---

## 🏃 Part 4: Indexing and Slicing

Notebook section: **"Part 4: Indexing and Slicing"**

- 1D array indexing (similar to Python lists)
- 2D array indexing with `[row, col]` syntax
- Array slices are **views**, not copies — modifying a slice affects the original array

---

## 🏃 Part 5: Boolean Indexing

Notebook section: **"Part 5: Boolean Indexing"**

Comparisons on arrays (such as `==`, `>`, `<`) are vectorized — they produce a **Boolean array** of `True`/`False` values. That Boolean array can then be used directly to filter data, selecting only the rows where the condition is `True`.

```python
names  = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
scores = np.array([[75, 80], [85, 90], [95, 100], [100, 77], [85, 92], [95, 80], [72, 80]])

bob_mask = (names == 'Bob')          # array([True, False, False, True, False, False, False])
print(scores[bob_mask])              # rows where name is 'Bob'
```

Key points:
- The boolean mask must have the same length as the dimension you are indexing
- Combine conditions with `&` (AND) and `|` (OR) — each condition must be wrapped in parentheses
- Assigning to a boolean-indexed slice modifies the **original** array in place

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

# 3. Set all scores below 80 to 0
scores[scores < 80] = 0
print(scores)
```

</details>

---

## 🏃 Part 6: Universal Functions (ufuncs) and Methods

Notebook section: **"Part 6: Universal Functions (ufuncs) and Methods"**

- Unary ufuncs: `sqrt`, `exp`
- Binary ufuncs: `add`, `maximum`
- Statistical methods: `mean`, `sum`, `std`
- Computing statistics along axes (`axis=0`, `axis=1`)

> **Discussion:** When should you use ufuncs vs. loops for performance?

---

## 🏃 Part 7: Linear Algebra

Notebook section: **"Part 7: Linear Algebra"**

- Element-wise multiplication with `*` vs. matrix multiplication with `.dot()` or `@` operator
- Example: multiplying two matrices

### 🛠️ Exercise 4: Reshaping & Statistics

- Create an array of 15 integers using `arange(15)` and reshape it to `(3, 5)`
- Calculate the average value of each row
- Use `np.unique()` to find distinct elements
- Transpose the reshaped array using `.T` and check the new shape

---

## 🎯 Wrap-Up

**Key Takeaways:**
1. NumPy arrays are the foundation — Pandas, scikit-learn, and TensorFlow all build on NumPy's ndarray.
2. Slices are **views**, not copies — use `.copy()` when you need independence from the source array.
3. `@` for matrix multiplication, `*` for element-wise — getting these confused causes silent wrong results.

**Next Steps:**
- Complete the [Assignment](./assignment.md) — 7 NumPy practice exercises covering arrays, masking, broadcasting, and linear algebra.
- Next lesson: Lesson 1.7 introduces Pandas — a higher-level library built on NumPy that adds labelled indexing, mixed data types, and table-like operations.
