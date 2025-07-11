# Animated Gradient Descent Visualization

A Python project that demonstrates the inner workings of linear regression via gradient descent. It produces a 2×2 Matplotlib animation showing:

1. The fitted regression line adapting to the data.
2. The Mean Squared Error (MSE) cost decreasing over epochs.
3. The slope (m) parameter converging.
4. The intercept (b) parameter converging.

---

## 📂 Repository Contents

* `study.csv` – Sample dataset with columns `x` and `y`.
* `Gradient_Descent.gif` - This file
* `Linear Regression.ipynb` - For experiment purpose
* `Linear Regression.py` – Main script defining `GradientDescentLinearRegression` and generating the animation.
* `README.md` – This file.

---

## 🚀 Features

* **Custom Gradient Descent** implementation tracking `m`, `b`, and cost per epoch.
* **Feature scaling** via `StandardScaler` for faster convergence.
* **Matplotlib FuncAnimation** producing an animated GIF or inline display.
* Easy to extend to multiple features, optimizers, or different cost functions.

---

## 📋 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/animated-gradient-descent.git
   cd animated-gradient-descent
   ```
2. (Optional) Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

> **Note:** If `requirements.txt` is not provided, manually install:
>
> ```bash
> pip install numpy pandas matplotlib scikit-learn
> ```

---

## ⚙️ Usage

Run the main script to generate and display the animation:

```bash
python animated_linear_regression.py
```

This will:

* Load `study.csv`.
* Train a linear model via gradient descent for 250 epochs (default).
* Scale features automatically.
* Display a 2×2 animated plot showing fit, cost, slope, and intercept evolution.

To save the animation as a GIF instead of inline display, modify the bottom of the script:

```python
from matplotlib.animation import PillowWriter

anim = FuncAnimation(fig, update, frames=epochs, interval=50, blit=True)
writer = PillowWriter(fps=20)
anim.save("gradient_descent.gif", writer=writer)
```

---

## 🛠️ Configuration

* **Epochs** and **learning rate** can be adjusted in the `GradientDescentLinearRegression` constructor:

  ```python
  lr2 = LinearReg(epochs=500, learning_rate=0.005)
  ```
* **Data file** path and column names can be changed in `__main__` section.

---

## 🔧 Extending the Project

* Add support for **multiple input features** by vectorizing `X` and adjusting gradient formulas.
* Implement alternative **optimizers** (e.g., Adam, RMSProp) in place of vanilla gradient descent.
* Visualize other **cost functions** (MAE, Huber loss, cross-entropy for classification).
* Build a **Streamlit** or **Dash** app for interactive parameter tuning.

---

## 📜 License

This project is MIT‑licensed. See [LICENSE](LICENSE) for details.

---

## 🙏 Contributions

Feel free to open issues or submit pull requests to improve the visualization, add features, or refine documentation.

---

*Happy Learning & Visualizing!*
