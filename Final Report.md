# Chest X Ray Image Classification

By Michael McCarty, Ziyi Yan, Lyndon Lee

## Scientific Problems

In this project we looked at biomedical image analysis using convolutional neural networks. In particular, we automated patient diagnosis for diseases from chest X-rays. This is not a new problem in the field. Both medical researchers and Kaggle enthusiasts have generated CNN models to this end. The specific point of interest that we explored is determining whether we could improve the predictive accuracy of a machine learning model by subsetting data.

Based on our interview with a resident Duke radiologist, we determined some interesting properties of chest x rays, which knowledge could be useful in training models to perform optimally.

Chest X ray images can be taken in one of two modes: Posterior-to-Anterior (PA) or Anterior-to-Posterior (AP). The difference is the position of the x ray source. The PA view position is considered to be the gold standard, as it contributes less distortion to the image with up to 15% of accurate diagnoses (Raoof et al., 2012). One example of this is when determining if a heart is enlarged (a condition known as cardiomegaly). Since the heart is located at the front of the chest, when the x ray source is in the front, it can make the heart in the image appear larger than in reality. This can cause difficulty in diagnosing the disease. 

Another difficulty when taking x ray images can be when taking images of female patients. As females tend to have more breast tissue, this can appear as differences in the acquired image, which can complicate the analysis.

For our project, we trained a model with the full dataset, and then we trained the model after subsetting our data based on sex and X-ray orientation. This was in an attempt to maximize the effectiveness of our models. For the reasons explained above, the view position and the sex of the patient can alter the appearance of an image. Since these parameters are known both for the training set, and for potential images which will be run through the completed model, we hypothesize that it would be beneficial to analyze images using a model that was trained on the same type of data.

By creating a model that can diagnose conditions based on x rays, this could assist radiologists to determine difficult to diagnose conditions. Additionally, even in cases where the condition is relatively easy for a radiologist to determine, an AI system could screen images and determine which should be given the highest priority for a radiologist to review, based on the severity of the condition.

## Data Provenance

The data for our study mainly comes from a NIH chest x-ray dataset on kaggle, which was obtained from >100,000 anonymized chest x-ray images and their corresponding data released by the NIH Clinical Center to the scientific community. The NIH Chest X-ray Dataset comprised of 112,120 X-ray images with disease labels from 30,805 unique patients from [Kaggle](https://www.kaggle.com/nih-chest-xrays/data). 

Classes and labels in the dataset are generated based on Natural Language Processing to text-mine disease classifications from the associated radiological reports, and the labels are expected to have an accuracy of over 90%. There are 15 disease classes within the dataset including healthy individuals. The labels in the dataset include age, gender, x-ray orientation (AP or PA), which are valuable for us to further explore how different features would affect CNN.

## Data Preprocessing Pipeline

For the initial model we trained, we used a small subset of the data provided by the kaggle challenge, which was obtained by random selection. Using this smaller dataset, we were able to train each of the diseases (and healthy condition) separately. More specifically, we used 5606 sample cases for training, and all 14 conditions were included in the sample cases. Working with a small set of data allowed us to work directly on google co-lab without a local GPU’s support, and the preprocessing sample test allows us to focus on building up the model and make sure it is functioning correctly. In order to separate the different conditions, we splitted the “Finding Labels” variable by “|” and created a list of labels for all cases. By doing so, each chest image can be trained under multiple conditions based on its labels.

In order to run the model on the larger dataset, while working within the parameters of the system we were running on, it became necessary to create datasets for each of the subsetted categories, as the full dataset could not be downloaded at once on the computer that is training the model (It is a 42GB dataset). On a local computer, which could hold all of the images, we created a script that read the csv file with all of the image metadata, and copied the images from the full dataset into one of four folders for each combination of Male/Female and AP/PA (FAP, FPA, MAP, MPA). New csv files were created as well for each of the categories. 

With the help of specifying patient gender and image orientation, we believe that we can enhance the accuracy of diagnosis by eliminating the confounding influences. The final subset data each contained around 6~7 GB of image files. 

## Machine Learning Methodology

### Loss Function

The loss function used in this model is the binary cross-entropy function.

This is the formula for the binary cross-entropy:

![binary_entropy.png](/images/image7.png)

Note that P(y <sub>i</sub>) here refers to the probability of being the true class. Taking the negative log of P(y<sub>i</sub>) means that the close the value of P(y<sub>i</sub>) to 1 (i.e. the more likely an image is a particular disease in question), the smaller the value of negative log(P(y<sub>i</sub>)) (since the log of a value between 0 and 1 is negative and the negative of negative is positive), and thus resulting in a smaller loss (this loss being denoted by H in the equation). This corresponds to the graph below. This should be intuitive because the loss function is penalizing less (by holding a smaller value) when the image is more likely to be of the disease in question.

![log_loss.png](/images/image5.png)

We did not use categorical cross-entropy because each image can be labelled with more than one disease. Categoricla cross-entropy is for mutually exclusives classes.

### Activation Function

Because the loss function calculates the probability of whether a given lung image has a specific disease or not, it becomes analogous to the case of logistic regression where the model predicts a binary value (yes/ no) for the response variable. Mathematically, it allows the output value to range between 0 and 1 which is the valid range for probabilities. This is why the sigmoid function is used as the activation function. 

We are, however, also mindful of the vanishing gradient problem for activation functions like logistic, tanh. This could slow down learning time for early layer parameters and generates inaccurate parameters. Thus, we tried other methods like ReLu activation function, but we do not find the vanishing gradient problem to be especially huge in this case.

### Optimizer

The optimizer that we used is the Adam optimizer. This is an algorithm that can be used to iteratively update the weights of the training model, in place of stochastic gradient descent. Some of its advantages include computational efficiency, intuitive hyperparameters, and that it does not require substantial fine tuning. Unlike stochastic gradient descent which maintains the same learning rate throughout the training process and for all parameters, the Adam optimizer had individual learning rates for different parameters. This allows us to solve deep learning problems faster than most other algorithms.

### Metrics

The metrics used in assessing the model’s performance was binary accuracy and mean absolute error. Refer to their equations below.

![accuracy.png](/images/image3.gif)

![mae.png](/images/image2.gif)

Binary accuracy refers to the ratio of correct predictions to the total number of predictions. The main drawback of using accuracy as a metric is that it is misleading at times. For instance, if a lung disease model that selects hernia 100% of the time is applied to predict a sample of say 90% Hernia patients and 10% of all other diseases, it will have a 90% accuracy even though we know the model can only predict 1 kind of disease and is highly limited. Thus, we also employed the Mean absolute error. MAE refers to the average magnitude of the difference between the original value and fitted value.

# Results

## ROC Curves for Various Subsets
![final_fap.png](/images/image10.png)
![final_fpa.png](/images/image9.png)
![final_map.png](/images/image6.png)
![final_mpa.png](/images/image4.png)

## Visualization of Results
![diff_gender.png](/images/image1.png)
![diff_position.png](/images/image8.png)

## Results & Discussion 

Above are the results obtained from the CNN models. The 4 AUC-ROC curves are each based on 13960 training images and 4654 validation images, with epochs set to 5. The latter 2 comparative graphs were generated based on the ROC differences between the 2 subset basis: view position (AP - PA) and Sex (Male - Female). If the comparing values are positive (blue), it shows that the ROC difference is larger for the former condition (AP/Male), suggesting that the ROC accuracy is higher in those conditions.

From the AUC-ROC curves, we can see that by dividing our data into subsets, the models trained by the male subsets are more accurate. The gender difference not only affected the overall accuracy, but also influenced the training accuracy for specific diseases such as Effusion and Pneumothorax. This result matches with our hypothesis that some disease might be harder to diagnose for females since there’s the beast tissue. In order to better visualize the difference, we created two comparative plots based on the gender and view position differences. In general, the ROC difference between male and female are positive, suggesting that models trained on male data are usually more accurate for both AP and PA view positions. The AP-PA comparative graph did not show a clear preference for either AP or PA, suggesting that the change in view position did not lead to a significant change in the training accuracy. 

However, there are a few exceptions in the graph, and the main one is the curve for Hernia. Hernia had extremely high false-positive rates in all four subset conditions, which also led to a large difference in the comparative graphs. As we went back to check our dataset, Hernia was the least frequent disease that appeared in each  group, with a percentage of only around ~0.10%. The lack of available images for the disease can explain why the prediction accuracy for certain diseases are low since there is insufficient information to train the models for them. One limitation of our study is that the image labels are NLP extracted so there could be some erroneous labels, and there was controversy on the data’s reliability. Though the labeling accuracy was estimated to be >90%, it was still possible for the erroneous labels to reduce our model accuracy. The inaccuracy might be caused due to the lack of patient history in the provided data. One study suggested that prior illness (e.g. fever, coughing) can indicate certain diseases over the others (e.g. pneumonia would be appropriate rather than infiltration or consolidation in the case of fever or cough) (Rajpurkar, P. et al., 2017). For future improvements, we believe that adding more related information would be a useful approach to enhance accuracy. We will search for datasets that contain patient history to take the factor into account. Now, we are working with 14 disease conditions at the same time, which might have brought chaos to the models when identifying diseases with similar phenotypes. To enhance accuracy, we can further classify these diseases into groups that are highly comorbid but with different traits, and by doing so we can eliminate the effect of similar traits on our models.

## References

Baltruschat, I. M., Nickisch, H., Grass, M., Knopp, T., & Saalbach, A. (2019). Comparison of Deep Learning Approaches for Multi-Label Chest X-Ray Classification. Scientific Reports, 9(1), 6381. doi:10.1038/s41598-019-42294-8

Brownlee, J. (2019, November 13). Gentle Introduction to the Adam Optimization Algorithm for Deep Learning. Retrieved May 02, 2020, from https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

Bullock, Joseph, Carolina Cuesta-Lazaro, and Arnau Quera-Bofarull. “XNet: a Convolutional Neural Network (CNN) Implementation for Medical x-Ray Image Segmentation Suitable for Small Datasets.” Ed. Barjor Gimi and Andrzej Krol. Medical Imaging 2019: Biomedical Applications in Molecular, Structural, and Functional Imaging (2019): n. pag. Crossref. Web.

Godoy, D.  (2018, November 21). Understanding binary cross-entropy / log loss: A visual explanation. Retrieved May 2, 2020, from https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

Mishra, A. (2018, November 01). Metrics to Evaluate your Machine Learning Algorithm. Retrieved May 02, 2020, from https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234

Que, Q., Tang, Z., Wang, R., Zeng, Z., Wang, J., Chua, M., . . . Veeravalli, B. (2018). CardioXNet: Automated Detection for Cardiomegaly Based on Deep Learning. Conf Proc IEEE Eng Med Biol Soc, 2018, 612-615. doi:10.1109/embc.2018.8512374

Rajpurkar, P. et al. Chexnet: Radiologist-level pneumonia detection on chest x-rays with deep learning. arXiv preprint arXiv:1711.05225 (2017).

Taylor, A. G., Mielke, C., & Mongan, J. (2018). Automated detection of moderate and large pneumothorax on frontal chest X-rays using deep convolutional neural networks: A retrospective study. PLoS medicine, 15(11), e1002697. https://doi.org/10.1371/journal.pmed.1002697

Yadav, S. S., & Jadhav, S. M. (2019). Deep convolutional neural network based medical image classification for disease diagnosis. Journal of Big Data, 6(1), 1-18. doi:http://dx.doi.org/10.1186/s40537-019-0276-2

Yao, L. et al. Learning to diagnose from scratch by exploiting dependencies among labels. arXiv preprint arXiv:1710.10501 (2017). 
