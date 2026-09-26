Project summary
Finlora is a digital financial technology company that processes a large volume of transactions across multiple customers, payment channels, devices, currencies, and geographic locations. As transaction activity continues to grow, identifying fraudulent transactions using traditional rule-based monitoring becomes increasingly challenging.

This project focuses on developing a machine learning-based fraud detection system that can analyse historical transaction data and learn patterns associated with fraudulent and legitimate transactions. The system will use transaction, customer, behavioural, temporal, device, and geographic features to classify transactions and identify those that may require further investigation.

The project will involve data analysis, feature engineering, machine learning model development, model evaluation, explainability, experiment tracking, and deployment. The final solution will provide a prototype fraud detection system supported by a Streamlit dashboard for fraud analysis and a FastAPI service for transaction-level predictions. The objective is to demonstrate how machine learning can complement Finlora's existing fraud monitoring process and provide a more scalable, data-driven approach to fraud detection.

Business context

Finlora processes a large volume of digital financial transactions across different customers, payment channels, devices, and geographic locations. As transaction activity increases, the company faces greater difficulty in identifying fraudulent transactions using traditional rule-based monitoring alone.

The current approach may generate false alerts while also failing to detect new or complex fraud patterns. Finlora therefore needs a data-driven solution that can analyse historical transaction behaviour, identify patterns associated with fraud, and support analysts in prioritising suspicious transactions.

This project will explore the use of machine learning to complement Finlora's existing fraud detection processes and provide a scalable approach to identifying potentially fraudulent transactions.

Project purpose

The purpose of this project is to develop a machine learning-based fraud detection system that can identify potentially fraudulent transactions using historical Finlora transaction data. The system will analyse transaction and customer behaviour, compare different classification models, and provide interpretable fraud predictions to support fraud analysts.

The project will also deliver a prototype dashboard and prediction API to demonstrate how the model can be integrated into Finlora's existing fraud monitoring process.

Expected outcomes

The project is expected to deliver a functional machine learning-based fraud detection prototype that demonstrates how Finlora can use its transaction data to improve fraud monitoring and support data-driven decision-making. The key outcomes include:

A cleaned and prepared fraud detection dataset.

Identification of key patterns and features associated with fraudulent transactions.

A trained and evaluated machine learning model for fraud classification.

A comparison of multiple fraud detection models using precision, recall, F1-score, and confusion matrix.

An explainable fraud detection model using feature importance

A Streamlit dashboard for monitoring fraud patterns and flagged transactions.

An MLflow experiment-tracking setup for managing model experiments and versions.

A FastAPI prediction service for transaction-level fraud detection.

A Dockerized deployment prototype demonstrating how the solution can be operationalized.
Company overview
FinLora

Finlora is a digital financial technology company focused on providing accessible and convenient money transfer and payment services through digital platforms. The company was established to simplify how individuals and businesses move money across different locations and payment channels while maintaining a strong emphasis on transaction speed, accessibility, and customer experience.

From its early operations, Finlora positioned itself as a technology-driven financial services provider, using digital platforms to reduce the friction traditionally associated with financial transactions. Its services are primarily delivered through digital channels, allowing customers to initiate and manage transactions without relying heavily on physical banking infrastructure.

Growth & Scale

As digital financial services continue to gain adoption, Finlora has experienced steady growth in its customer base and transaction activity. The company now processes a large volume of transactions across multiple customers, payment channels, currencies, devices, and geographic locations.

This growth has increased the amount of transaction data available to the organization but has also introduced greater operational complexity. Transactions now occur at a scale where manually reviewing suspicious activity is no longer practical, creating a growing need for automated and data-driven transaction monitoring.

Capabilities & Operations

Finlora's digital platform supports activities such as:

Digital money transfers

Customer-to-customer payments

Cross-border transactions

Digital wallet operations

Multiple payment channels

Customer account management

The company's transaction infrastructure generates data relating to transaction amounts, timestamps, customer behaviour, payment channels, devices, locations, and transaction history.

Differentiators

Finlora's operations are built around:

Digital-first financial services

Data-driven transaction processing

Cross-border payment capabilities

Automated transaction monitoring

Technology-enabled customer experience

Competitive Advantage

Finlora's ability to combine digital financial services with data-driven decision-making provides an opportunity to build more intelligent transaction monitoring capabilities as the company scales.

Business challenge
Finlora currently relies primarily on rule-based transaction monitoring to identify potentially fraudulent transactions. These rules use predefined thresholds and conditions to flag suspicious activities for further investigation.

While this approach can effectively identify known fraud patterns, it becomes increasingly limited as transaction volume grows and fraudulent behaviour becomes more complex. Fraudsters can modify their transaction patterns to avoid triggering predefined rules, while legitimate customers may occasionally behave in ways that cause unnecessary alerts.

Finlora therefore needs to determine whether a machine learning approach can provide a more effective first line of defence against fraudulent transactions.

Context Behind the Issue

Finlora processes transactions across multiple digital channels and customer segments. This creates considerable variation in:

Transaction amounts

Transaction frequency

Transaction timing

Customer behaviour

Payment channels

Devices

Geographic locations

Account history

These variations make it difficult to define a fixed set of rules that can accurately represent every legitimate and fraudulent transaction pattern.

At the same time, historical transaction data contains potentially valuable information about the characteristics of previously identified fraudulent transactions.

Key Obstacles and Pain Points

1. Static Detection Rules

Existing rules are based on predefined conditions and may not adapt effectively to new fraud patterns.

2. Increasing Transaction Volume

The growing number of transactions creates a larger pool of transactions that need to be monitored and potentially investigated.

3. False Positive Alerts

Legitimate transactions may be flagged because they resemble predefined suspicious patterns, increasing the workload for fraud analysts.

4. Complex Fraud Patterns

Fraudulent transactions may involve combinations of behavioural, transactional, temporal, and geographic characteristics that are difficult to capture using individual rules.

5. Class Imbalance

Fraudulent transactions represent only a small proportion of total transactions, creating a highly imbalanced classification problem.

Business Impact

If fraudulent transactions are not identified effectively, Finlora may experience:

Direct financial losses

Increased fraud investigation costs

Higher customer support workload

Customer dissatisfaction

Increased operational pressure on fraud analysts

Reduced confidence in transaction security

Conversely, excessive false-positive alerts can unnecessarily disrupt legitimate customers and increase the workload associated with manual investigations.



Rationale for this project
The project will investigate the use of supervised machine learning classification to identify potentially fraudulent transactions.

Instead of relying exclusively on manually defined rules, the proposed approach will learn relationships between transaction characteristics and historical fraud labels.

The model will assign a fraud prediction to each transaction based on the patterns learned from historical data.

Industry Relevance

Fraud detection is an important challenge across digital financial services because transaction systems operate at high volumes while fraudulent behaviour continuously evolves.

A machine learning approach can analyse multiple transaction characteristics simultaneously and identify combinations of features that may be difficult to capture using traditional rule-based systems.

For Finlora, this provides an opportunity to complement existing rules with a data-driven fraud detection capability.

Real-World Examples

Major digital financial service providers and payment companies increasingly use machine learning, behavioural analytics, and automated risk scoring to detect suspicious transaction activity.

This demonstrates the broader industry relevance of combining traditional fraud controls with data-driven detection methods.

Strategic Drivers

The project is driven by the need to:

Improve fraud detection by identifying suspicious transactions more effectively.

Support decision-making by providing fraud analysts with model-based risk signals.

Reduce operational workload by prioritizing potentially fraudulent transactions.

Reduce unnecessary alerts by improving the distinction between legitimate and suspicious transactions.

Improve adaptability by allowing the detection approach to learn from historical transaction patterns.

Establish a scalable foundation for future automated fraud detection capabilities.

Strategic Outcome

The desired outcome is a working machine learning prototype that demonstrates how Finlora can use its transaction data to supplement existing rules-based monitoring with a more intelligent and data-driven fraud detection approach.
Project objectives
The project aims to achieve the following objectives:

Objective 1: Transaction Risk Analysis
Analyse historical Finlora transaction data to identify characteristics and behavioural patterns associated with fraudulent transactions.

Objective 2: Fraud Pattern Identification
Identify significant transaction, customer, temporal, device, geographic, and behavioural factors that distinguish fraudulent transactions from legitimate transactions.

Objective 3: Fraud Classification Model
Develop and compare machine learning classification models capable of predicting whether a transaction is potentially fraudulent or legitimate.

Objective 4: Class Imbalance Handling
Apply appropriate techniques, such as class weighting and resampling, to address the imbalance between fraudulent and legitimate transactions.

Objective 5: Model Evaluation
Evaluate model performance using precision, recall, F1-score, and confusion matrix, with particular emphasis on the model's ability to correctly identify fraudulent transactions while limiting false-positive alerts.

Objective 6: Model Explainability
Apply SHAP and feature importance techniques to identify the factors contributing to fraud predictions and improve the interpretability of model outputs.

Objective 7: Fraud Intelligence Dashboard
Develop a Streamlit dashboard that enables fraud analysts to explore transaction patterns, fraud trends, model performance, predictions, and flagged transactions.

Objective 8: Model Experiment Tracking
Use MLflow to track model experiments, parameters, evaluation metrics, artifacts, and model versions throughout the development process.

Objective 9: Model Deployment Prototype
Develop a FastAPI-based prediction service that exposes the selected fraud detection model for transaction-level fraud predictions.

Objective 10: Containerization
Containerize the fraud detection API and its supporting components using Docker to provide a portable and reproducible deployment prototype.

In scope
The project will focus on developing an end-to-end machine learning prototype for fraudulent transaction detection at Finlora. The following activities are within the scope of the project:

Collection and preparation of transaction, customer, and fraud investigation data.

Data quality assessment, including missing values, duplicates, data types, outliers, and class distribution.

Exploratory data analysis to identify transaction and behavioural patterns associated with fraud.

Feature engineering using transaction, customer, temporal, device, geographic, and behavioural information.

Development of a baseline Logistic Regression model.

Training and comparison of Random Forest, XGBoost, and LightGBM classification models.

Handling class imbalance using techniques such as class weighting and SMOTE.

Model evaluation using precision, recall, F1-score, and confusion matrix.

Model explainability using SHAP and feature importance analysis.

MLflow implementation for experiment tracking.

Development of a Streamlit dashboard for fraud analysis and visualization.

Development of a FastAPI service for transaction-level fraud predictions.

Docker containerization of the prediction service.

Documentation of the data pipeline, modelling process, evaluation results, and deployment prototype.

Out of scope
The following activities are outside the scope of this project:

Real-time integration with Finlora's live transaction processing systems.

Direct connection to production banking, payment, or financial infrastructure.

Automated blocking, rejection, or approval of customer transactions.

Fully automated fraud investigation or case resolution.

Deployment of the system into a production environment.

Development of a mobile or web application for customers.

Development of new payment processing or transaction management systems.

Integration with external financial institutions or payment providers.

Development of customer-facing fraud alerts or notification systems.

Use of personally identifiable customer information beyond what is required for the modelling prototype.

Replacement of Finlora's existing rule-based fraud detection system.

Guaranteeing that all fraudulent transactions will be detected by the machine learning model.

Continuous model retraining and production-grade model monitoring.

Financial loss recovery, fraud investigation, or legal enforcement activities.
