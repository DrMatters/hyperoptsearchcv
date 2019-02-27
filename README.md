# hyperoptsearchcv

> Wrapper for hyperopt to use it in sklearn pipelines

### TODO
- [ ] Make interface similar to sklearn.model_selection.RandomizedSearchCV
- - [ ] Remove 'maximize' parameter
- - [ ] Implement usage of user-selected validation. Add `scoring` and `cv` parameters.
- - [ ] Modify text log behavior (use `verbose` parameter)
- [ ] Modify best_score calculation (remove magic numbers)
