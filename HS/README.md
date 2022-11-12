### Harmony Search

##### Reference: A new heuristic optimization algorithm: Harmony search[J]. Simulation, 2001, 2(2):60-68

| Variables | Meaning                                               |
| --------- | ----------------------------------------------------- |
| hms       | Harmony memory size                                   |
| iter      | The number of iterations                              |
| hmcr      | Harmony memory consideration rate                     |
| par       | Pitch adjustment rate                                 |
| bw        | Bandwidth                                             |
| nnew      | The number of new harmonies created in each iteration |
| lb        | The lower bound (list)                                |
| ub        | The upper bound (list)                                |
| pos       | The set of harmonies (list)                           |
| score     | The score of harmonies (list)                         |
| dim       | Dimension (list)                                      |
| new_pos   | The set of newly created harmonies (list)             |
| new_score | The score of newly created harmonies (list)           |
| gbest     | The score of the global best harmony                  |
| gbest_pos | The position of the global best harmony (list)        |
| iter_best | The global best score of each iteration (list)        |
| con_iter  | The last iteration number when "gbest" is updated     |

#### Test problem: Pressure vessel design

![](https://github.com/Xavier-MaYiMing/Harmony-Search/blob/main/Pressure%20vessel%20design.png)

$$
\begin{align}
&\text{min}\ f(x)=0.6224x_1x_3x_4+1.7781x_2x_3^2+3.1661x_1^2x_4+19.84x_1^2x_3,\\
&\text{s.t.} \\
&-x_1+0.0193x_3\leq0,\\
&-x_3+0.0095x_3\leq0,\\
&-\pi x_3^2x_4-\frac{4}{3}\pi x_3^3+1296000\leq0,\\
&x_4-240\leq0,\\
&0\leq x_1\leq99,\\
&0\leq x_2 \leq99,\\
&10\leq x_3 \leq 200,\\
&10\leq x_4 \leq 200.
\end{align}
$$


#### Example

```python
if __name__ == '__main__':
    # Parameter settings
    hms = 30
    iter = 1000
    hmcr = 0.9
    par = 0.1
    bw = 0.02
    nnew = 20
    lb = [0, 0, 10, 10]
    ub = [99, 99, 200, 200]
    print(main(hms, iter, hmcr, par, bw, nnew, lb, ub))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/Harmony-Search/blob/main/convergence%20iteration.png)

The HS converges at its 674-th iteration, and the global best value is 8114.787624542521. 

```python
{
    'best score': 8114.787624542521, 
    'best solution': [1.3078025478995294, 0.6468035853866221, 67.40186804119926, 10], 
    'convergence iteration': 674
}
```

