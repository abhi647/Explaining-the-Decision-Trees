# Decision Trees
## Non-parametric supervised* learning method used to classification and regression, goal is to create a model that predicts the value of target variable by learning simple decision rules.
## <u>_*Trivia: What's the difference between supervised and un-supervised learning?_</u>
#### __Supervised__: _When you guide your machine and algorithm to reach a desired result with the help of lablled data, for e.g. in the case of classification and Regression you provide a target variable to train the model._
#### __Un-Supervised__: _When you don't know the target variable, you ask machine to infer the pattern by itself_.
<img src="https://miro.medium.com/max/640/1*WudJvOE0eJ87EKofpHCf0Q.png" style="width: 400px;"/>
### A simple example of decision tree
<img src="https://i.vas3k.ru/7w3.jpg" alt="Drawing" style="width: 400px;"/>

### Few Advantages of Decision tree
- __Easy to understand__, interpret and visualise
- Requires __little or no data preperation__
- Able to handle multiple-output problems
- Uses __Whitebox model__ which is __easily explained by boolean logic__, either yes or no. In contrast to black box model in CNN and ANN
- Possible and __easy to validate__ using statistical Tests

Trivia: They form the backbone of most of the best performing models in the industry like XGboost and Lightgbm

## What are all the various decision tree algorithms and how do they differ from each other?:
- __ID3 (Iterative Decotomizer)__:ID3 is an algorithm invented by Ross Quinlan used to generate a decision tree from a dataset. ID3 is the precursor to the C4.5 algorithm

- __C4.5__: C4.5 builds decision trees from a set of training data in the same way as ID3, using the concept of information entropy. The training data is a set S= S1,S2,S3...of already classified samples. Each sample Si consists of a p-dimensional vector (x1,x2,x3,...xpi) where the  represent attribute values or features of the sample, as well as the class in which Si falls.
___Improvement in C4.5 from ID3: Handling both continuous and discrete attributes - In order to handle continuous attributes, C4.5 creates a threshold and then splits the list into those whose attribute value is above the threshold and those that are less than or equal to it___

- __C5.0__: The C5.0 algorithm has become the industry standatd for producing decision trees, because it does well fo rmost types of problems directly out of the box. Compared to more advanced and sophisticated machine learning mnodels (e.g. Neural Networks and Support Vector Machines), the decision trees under the C5.0 algorithm generally perform nearly as well but are much easier to understan and deploy.

- __CART__: (Classification and Regression Trees) is very similar to C4.5, but it differs in that it supports numerical target variables (regression) and does not compute rule sets. CART constructs binary trees using the feature and threshold that yield the largest information gain at each node.

___scikit-learn uses an optimised version of the CART algorithm; however, scikit-learn implementation does not support categorical variables for now.___

### Mathematical Formulation: 
#### Given training vectors $ x_i \in R^n $ i=1,…, l and a label vector $y \in R^l$ a decision tree recursively partitions the space such that the samples with the same labels are grouped together.
#### Let the data at node $m$ be represented by $Q$. For each candidate split $\theta = (j, t_m)$ consisting of a feature $j$ and threshold $ t_m$, partition the data into $ Q_{left}(\theta) $ & $Q_{right}(\theta)$

$$  \begin{align}\begin{aligned}Q_{left}(\theta) = {(x, y) | x_j <= t_m}\\Q_{right}(\theta) = Q \setminus Q_{left}(\theta)\end{aligned}\end{align} $$
 The impurity at $m$ is computed using an impurity function $H()$ , the choice of which depends on the task being solved (classification or regression)

$$ G(Q, \theta) = \frac{n_{left}}{N_m} H(Q_{left}(\theta))
+ \frac{n_{right}}{N_m} H(Q_{right}(\theta)) $$

Select the parameters that minimises the impurity
$$ \theta^* = \operatorname{argmin}_\theta  G(Q, \theta) $$

Recurse for subsets $Q_{left}(\theta^*)$ and $Q_{right}(\theta^*)$ until the maximum allowable depth is reached, $N_m < \min_{samples}$ or $N_m = 1$


# There are three main splitting criteria used in decision trees
## 1. Gini Impurity
### _Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset._
## Ig(n) = 1-  \sum_{i=1}^{J} ( p_i)^2 

## 2. Entropy: It is the measure of randomness in the system
## Entropy =  \sum_{i=1}^{J} -p  \times  \log_{2} (pi)

## 3. Variance: 
### _While doing Classification we can use Gini Impurities and Entropy but as far as Regression is concerned we need to use Variance. In Regression most common split is measured with the weighted variance of the node, because we want minimum variation in the nodes after the split_
## Variance =  (\sum (x- \overline{x})^2) /x 

