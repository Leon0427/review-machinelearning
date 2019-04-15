## Logistic Regression

### 1. cost function
cost(h,y) = -log(h), y=1; -log(1-h),y=0;
          = -ylog(h) - (1 - y)log(1-h)
>we choose this cost function coz it could generate a convex Objective function
J = (1/m)*Σ(cost(h,y)), if we use square cost function, the Objective J would be non-convex

### 2. gradient descent
### 3. optimize gradient descent
### 4. lr for multi-class classification