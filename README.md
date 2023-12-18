# Fake-News-Kaggle
 
**Project Overview: Fake News Detection using BERT**

### Step 1: Data Collection and Understanding
- Gathered datasets containing news articles labeled as real or fake news.
- Explored the dataset structure, understanding columns like `text` and `label`.

### Step 2: Preprocessing the Text Data
- Used the BERT tokenizer to convert textual data into tokenized sequences.
- Defined a maximum sequence length to ensure uniformity in tokenization.
- Split the data into train and test sets.

### Step 3: Data Preparation and Loading
- Prepared the tokenized data for training and testing.
- Created DataLoaders to handle batches during model training.

### Step 4: Model Selection and Initialization
- Utilized the BERT model (`bert-base-uncased`) designed for sequence classification.
- Initialized the model for binary classification (real vs. fake news).

### Step 5: Model Training
- Trained the BERT model on the training data.
- Utilized the AdamW optimizer for updating model weights.
- Iterated through epochs to train the model.

### Step 6: Evaluation and Validation
- Evaluated the trained model using the test dataset.
- Calculated accuracy and other metrics like precision, recall, and F1-score.
- Assessed the model's performance in distinguishing between real and fake news.

### Step 7: Model Deployment
- Deployed the trained model for inference.
- Created an endpoint or function to accept new text inputs.
- Used the model to predict whether the input news is real or fake.

### Step 8: Results and Future Improvements
- Analyzed the accuracy and effectiveness of the model.
- Explored possibilities for further improvements:
  - Hyperparameter tuning (epoch size, learning rate, batch size).
  - Experimentation with different BERT model variations.
  - Fine-tuning the model on specific domains or datasets.

### Step 9: Conclusion
- Summarized the key findings from the project.
- Presented insights about fake news detection using BERT.
- Discussed the potential impact and real-world applications.

### Step 10: Presentation
- Utilized visuals (charts, graphs) to illustrate model performance.
- Showcased snippets of real and fake news articles for comparison.
- Explained the project workflow and technical aspects for the audience.

Training result:

Epoch 1/3, Average Training Loss: 0.0936
Epoch 2/3, Average Training Loss: 0.0197
Epoch 3/3, Average Training Loss: 0.0101


Testing result:

## Results

### Model Performance

- **Accuracy:** 99.16%
- **Classification Report:**

|           | Precision | Recall  | F1-Score | Support |
|-----------|-----------|---------|----------|---------|
| Class 0   | 0.99      | 0.99    | 0.99     | 2079    |
| Class 1   | 0.99      | 0.99    | 0.99     | 2074    |
| **Overall** | **0.99** | **0.99**| **0.99**| **4153**|

- **Macro Average (Precision, Recall, F1-Score):** 0.99
- **Weighted Average (Precision, Recall, F1-Score):** 0.99


### Sections of the Code:
1. **Import Libraries:** Import necessary libraries such as pandas, numpy, torch, transformers from Hugging Face, and sklearn.
2. **Load Data:** Load the fake news dataset using pandas.
3. **Preprocessing:** Tokenize and prepare the text data using BERT tokenizer, including encoding text sequences and creating attention masks.
4. **Split Data:** Split the encoded data into train and test sets.
5. **Create DataLoaders:** Convert data into PyTorch `DataLoader` objects for efficient model training.
6. **Load BERT Model:** Load the pre-trained BERT model for sequence classification and set up the optimizer.
7. **Training Loop:** Train the BERT model for a specified number of epochs on the training data.
8. **Evaluation:** Evaluate the trained model on the test set to calculate accuracy and generate a classification report.
9. **Save Test Results:** Save the test set predictions along with the 'id' column in a CSV file.

### Discussion of Results:
- **Epoch Training Loss:** The training loss decreased with each epoch, indicating that the model learned progressively better representations of the data.
- **Accuracy:** The accuracy of the model on the test set is high, around 99.15%. This means the model accurately predicted fake or real news categories for most of the test samples.
- **Classification Report:** Precision, recall, and F1-scores are all high for both classes ('fake' and 'real' news). The high F1-scores imply a good balance between precision and recall.
- **Overall Performance:** The model seems to perform exceptionally well, showing high accuracy and robustness in classifying fake news from real news based on the provided features. The model demonstrated exceptional performance on the test set, achieving a high accuracy of 99.16%. Both classes (0 and 1) showcase balanced precision, recall, and F1-score, indicating robust predictive capabilities and effectiveness in distinguishing between the classes.


### Conclusion:
- This model, leveraging BERT for sequence classification, demonstrates impressive accuracy and robustness in identifying fake news.
- It's important to note that the results could vary depending on the dataset, preprocessing steps, and model fine-tuning. Therefore, further analysis and fine-tuning may be required for different datasets or specific use cases.


