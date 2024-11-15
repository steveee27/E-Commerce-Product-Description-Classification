# E-commerce Product Classification

This project focuses on classifying product descriptions from an e-commerce website into four categories: Household, Books, Electronics, and Clothing & Accessories. The classification model uses two machine learning algorithms, SVM and Random Forest, and two text representation methods, TF-IDF and Word2Vec.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Project Overview

- **Data Source**: Product descriptions from an e-commerce website with labels for four categories.
- **Text Representation Techniques**: TF-IDF and Word2Vec.
- **Machine Learning Algorithms**: Support Vector Machine (SVM) and Random Forest.
- **Performance Metrics**: Accuracy, Precision, Recall, and F1 Score.

## Project Workflow

1. **Data Preprocessing**: 
   - Clean the data by removing special characters, converting text to lowercase, and performing lemmatization.
   - Remove stopwords to focus on significant words.
2. **Text Representation**:
   - **TF-IDF**: Term Frequency-Inverse Document Frequency to represent text as numerical vectors.
   - **Word2Vec (CBOW)**: Continuous Bag of Words to capture the semantic similarity between words.
3. **Modeling and Hyperparameter Tuning**:
   - **Algorithms**: Apply SVM and Random Forest with hyperparameter tuning for each text representation method.
   - Tune at least two hyperparameters for each algorithm to improve performance.
4. **Evaluation and Comparison**:
   - Evaluate model performance on test data using Accuracy, Precision, Recall, and F1 Score.
   - Compare results across different models and text representation techniques to determine the most effective approach.

## Results

| Text Representation   | Algorithm     | Hyperparameters                              | Accuracy | Precision | Recall | F1 Score |
|-----------------------|---------------|----------------------------------------------|----------|-----------|--------|----------|
| **TF-IDF (Default)**  | **SVM**       | c = 1.0, kernel = rbf, gamma = scale         | 0.9547   | 0.9553    | 0.9547 | 0.9547   |
|                       | SVM (Tuning 1)| c = 0.5, kernel = linear, gamma = auto       | 0.9544   | 0.9546    | 0.9544 | 0.9543   |
|                       | SVM (Tuning 2)| c = 1.5, kernel = poly, gamma = scale        | 0.8346   | 0.8783    | 0.8346 | 0.8357   |
|                       | **Random Forest** | n_estimators = 100, max_depth = None, min_samples_split = 2 | 0.9345 | 0.9338 | 0.9345 | 0.9344 |
|                       | Random Forest (Tuning 1) | n_estimators = 150, max_depth = 20, min_samples_split = 5 | 0.8140 | 0.8630 | 0.8140 | 0.8140 |
|                       | Random Forest (Tuning 2) | n_estimators = 50, max_depth = 50, min_samples_split = 3 | 0.9218 | 0.9260 | 0.9218 | 0.9216 |
| **TF-IDF (Tuned)** <br> min_df = 5, max_df = 0.95 | **SVM** | c = 1.0, kernel = rbf, gamma = scale | **0.9551** | **0.9556** | **0.9551** | **0.9551** |
|                       | Random Forest | n_estimators = 100, max_depth = None, min_samples_split = 2 | 0.9361 | 0.9376 | 0.9361 | 0.9360 |
| **Word2Vec (CBOW)**   | **SVM**       | c = 1.0, kernel = rbf, gamma = scale         | 0.9290   | 0.9292    | 0.9290 | 0.9289   |
|                       | SVM (Tuning 1)| c = 1.5, kernel = linear, gamma = scale      | 0.9298   | 0.9298    | 0.9298 | 0.9297   |
|                       | SVM (Tuning 2)| c = 0.5, kernel = poly, gamma = auto         | 0.8544   | 0.8567    | 0.8544 | 0.8526   |
|                       | **Random Forest** | n_estimators = 100, max_depth = None, min_samples_split = 2 | 0.9405 | 0.9406 | 0.9405 | 0.9405 |
|                       | Random Forest (Tuning 1) | n_estimators = 75, max_depth = 45, min_samples_split = 4 | **0.9425** | **0.9426** | **0.9425** | **0.9424** |
|                       | Random Forest (Tuning 2) | n_estimators = 200, max_depth = 25, min_samples_split = 5 | 0.9406 | 0.9406 | 0.9406 | 0.9406 |

**Notes**:
- Hyperparameter tuning was performed to optimize the models for higher accuracy and F1 scores.
- The **SVM model with TF-IDF (Tuned)** and **Random Forest with Word2Vec (CBOW)** achieved the best performance overall.

## Conclusion

The results clearly demonstrate that the combination of TF-IDF with a tuned SVM model achieves the best performance across all metrics, including accuracy, precision, recall, and F1 score. Other combinations, such as Random Forest with Word2Vec or even TF-IDF, consistently lagged behind, with their highest scores reaching only around 0.92 compared to 0.95 achieved by the tuned SVM model. These findings highlight the importance of fine-tuning hyperparameters and selecting robust text representation techniques like TF-IDF for achieving superior results in e-commerce product classification. This approach proves particularly effective for tasks requiring both accuracy and consistency in categorization.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
