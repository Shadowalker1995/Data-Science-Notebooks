### Scikit-Learn Framework

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_family import ModelAlgo
mymodel = ModelAlgo(param1, param2)
mymodel.fit(X_train, y_train)
predictions = mymodel.predict(X_test)

frrom sklearn.metrics import error_metric
performance = error_metric(y_test, predictions)
```

### Regularization

- Regularization seeks to solve a few common model issues by:

    - Minimizing model complexity
    - Penalizing the loss function
    - Reducing model overfitting (add more bias to reduce model variance)

- In general, we can think of regularization as a way to reduce model overfitting and variance.

    - Requires some additional bias
    - Requires a search for optimal penalty hyperparameters.

- Three main types of Regularization:

    - L1 Regularization (LASSO Regression - Least Absolute Shrinkage and Selection Operator)
        $$
        \sum_{i=1}^n{(y_i - \beta_0 - \sum_{j=1}^p(\beta_j x_{ij}))^2} + \lambda \sum_{j=1}^p{|\beta_j|} = RSS + \lambda \sum_{j=1}^p{|\beta_j|}
        $$
        

        L1 Regularization adds a penalty equal to the **absolute value** of the magnitude of coefficients.

        - Limits the size of the coefficients.
        - Can yield sparse models where some coefficients can become zero.

        ```python
        # LASSO with Cross Validation
        from sklearn.linear_model import LassoCV
        lasso_cv_model = LassoCV(eps=0.1, n_alpha=100, cv=5, max_iter=1000000)
        lasso_cv_model.fit(X_train, y_train)
        lasso_cv_model.alpha_
        test_predictions = lasso_cv_model.predict(X_test)
        lasso_cv_model.coef_
        ```

    - L2 Regularization (Ridge Regression)
        $$
        \sum_{i=1}^n{(y_i - \beta_0 - \sum_{j=1}^p(\beta_j x_{ij}))^2} + \lambda \sum_{j=1}^p{\beta_j^2} = RSS + \lambda \sum_{j=1}^p{\beta_j^2}
        $$
        L2 Regularization adds a penalty equal to the **square** of the magnitude of coefficients.

        - All coefficients are shrunk by the same factor.
        - Does not necessary eliminate coefficients.

        ```python
        from sklearn.linear_model import Ridge
        ridge_model = Ridge(alpha=10)
        ridge_model.fit(X_train, y_train)
        test_predictions = ridge_model.predict(X_test)
        
        # Ridge with Cross Validation
        from sklearn.linear_model import RidgeCV
        ridge_cv_model = RidgeCV(alpha=(0.1, 1.0, 10.0), scoring='neg_mean_absolute_error')
        ridge_cv_model.fit(X_train, y_train)
        ridge_cv_model.alpha_
        test_predictions = ridge_model.predict(X_test)
        ridge_cv_model.best_score_
        ```

    - Combining L1 and L2 (Elastic Net)
        $$
        \frac{\sum_{i=1}^n{(y_i - x_i^J \hat\beta)^2}}{2n} + \lambda (\frac{1-\alpha}{2} \sum_{j=1}^m{\hat\beta_j^2} + \alpha \sum_{j=1}^m{|\hat\beta_j|})
        $$
        Elastics Net combines L1 and L2 with the addition of an alpha parameter deciding the ratio between them.

        ```python
        from sklearn.linear_model import ElasticNetCV
        elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                                    eps=0.001, n_alphas=100, max_iter=1000000)
        elastic_model.fit(X_train, y_train)
        elastic_model.l1_ratio
        elastic_model.alpha_
        test_predictions = elastic_model.predict(X_test)
        ridge_cv_model.best_score_
        ```

        

- These regularization methods do have a cost:
    - Introduce an additional hyperparameter that need to be tuned.
    - A multiplier to the penalty to decide the "strength" of the penalty.

### Feature Scalling

- Feature scaling provides many benefits to our machine learning process!

- Some machine learning models that rely on distance metrics (e.g. KNN) **require** scaling to perform well.

- Feature scaling improves the convergence of steepest descent algorithms, which do not possess the property of scale invariance.

- If features are on different scales, certain weights may update faster than others since the feature values $x_j$ play a role in the weight updates.

- Critical benefit of feature scaling related to gradient descent.

- There are some ML Algos where scaling won't have an effect (e.g. CART based methods).

- Scaling the features so that their respective ranges are uniform is important in comparing measurements that have different unit.

- Allows us directly compare model coefficients to each other.

- Feature scaling caveats:

    - Must always scale new unseen data before feeding to model.
    - Effects direct interpretability of feature coefficients
        - Easier to compare coefficients to one another, harder to related back to original unscaled feature.

- Feature scaling benefits:

    - Can lead to great increases in performance.
    - Absolutely necessary for some models.
    - Virtually no "real" downside to scaling features.

- Two main ways to scale features:

    - Standardization:

        - Rescales data to have a mean ($\mu$) of 0 and standard deviation ($\sigma$) of 1 (unit variance).
            $$
            X_{changed} = \frac{X - \mu}{\sigma}
            $$

        - Namesake can be confusing since this is also referred to as "Z-score normalization".

        

    - Normalization:

        - Rescales all data values to be between 0-1.
            $$
            X_{changed} = \frac{X - X_{min}}{X_{max} - X{min}}
            $$
            

        - Simple and easy to understand.

- There are many more methods of scaling features and Scikit-Learn provides easy to use classes that "**fit**" and "**transform**" feature data for scaling
    - A `.fit()` method call simply calculates the necessary statistics (min, max, mean, standard deviation).
    - A `.transform()` call actually scales data and returns the new scaled version of data.
    - Very important consideration for **fit** and **transform**:
        - We only **fit** to training data.
        - Calculating statistical information should only come from training data.
        - Don't want to assume prior knowledge of the test set!
        - Using the full data set would cause **data leakage**:
            - Calculating statistics from full data leads to some information of the test set leaking into the training process upon transform() conversion.

- Feature scaling process:
    - Perform train test split
    - Fit to training feature data
    - Transform training feature data
    - Transform test feature data

- Do we need to scale the label?
    - In general it is not necessary nor advised.
    - Normalizing the output distribution is altering the definition of the target.
    - Predicting a distribution that doesn't mirror your real-world target.
    - Can negatively impact stochastic gradient descent.
    - [Reference: Is it necessary to scale the target value in addition to scaling features for regression analysis?](stats.stackexchange.com/questions/111467)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)
```

