## XGBOOST

1. ensemble : boosting
2. obj = loss + regularization
3. addictive training
4. Taylor expansion of the loss function up to the second order:  l(y,y(t) + f(x)) = l(y,y(t)) + g·f(x) + 1/2 · h·f^2(x)
5. because of 4, xgboost could use customized base classifier
6. model complexity is modeled by num of leaf node and square sum of every leaf score
7. if a split's decrease on loss can't offset the amount complexity it increase, then no split will be conducted