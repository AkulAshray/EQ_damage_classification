# EQ_damage_classification

Data Source: https://eq2015.npc.gov.np/
data dictionary is found in appendix
Project Structure :

* **Clean Data.zip**: Contains clean train and test datasets.
* **model_selection_final.ipynb** : Ipython notebook used for exploratoring the data and for building our models to classify data.
* **feature_selection.ipynb** : notebook used for cutting down on our features and includes analysis on how to reduce dimension and complexity of model.
* **utilities.py**: contains functions used in building and analysing functions.

In April 2015 Nepal was hit by a massive earthquake with a magnitude of 7.8Mw or 8.1Ms and a maximum Mercalli Intensity of VIII (Severe).According to the Nepal open data portal it affected 3,677,173 individuals and 762,106 properties.Post disaster it took years for Nepal to collect and asses the damage which in terms results in one of the world’s largest damage assessment data.After a large scale disaster like earthquake the recovery is generally modeled over two phases.

Collection of demographic,architectural and legal data
Damage assessment by domain experts using this large scale and noisy data
Based on aspects of building location and construction, our goal is to predict the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal.

# Performance Metrics:

We are predicting the level of damage from 1 to 3(Low,Medium,High). The level of damage is an ordinal variable meaning that ordering is important. This can be viewed as a classification or an ordinal regression problem. (Ordinal regression is sometimes described as an problem somewhere in between classification and regression.)
To measure the performance of our algorithms, we’ have used the F1 score which balances the precision and recall of a classifier. Traditionally, the F1 score is used to evaluate performance on a binary classifier, but since we have three possible labels we used a variant called the weighted averaged F1 score.

# Steps taken in the project
1. Structural,ownership and damage information has been cleaned and concatenated in order to prepare train and test data set.
2. Data has been sampled manualy in order to deal with the imbalanced data set.From the initial cleaned 700k data 100k data has been sampled of each class and a training set of 300k datapoints has been prepared for training.Stratified sampling has been maintained throughout the process.                                              3. Numerical variables have been normalized,categorical variables have been vectorized using Label Encoder.Data matrix has been prepared as required for various machine learning and deep learning architectures.
4. Used a unsmote logistic regression as baseline,tried machine learning models like with SMOTE logistic regression,Random Forest,and LGBM
5. Simple models has been choosen as baseline,model complexity has been incresed gradualy.
6. GridSearchCV,simple crossvalidation etc. has been used for cross validation at various step.                        7. In practice lightGBM gave the best F1 score and good distinction between high and low damage areas.
8. SHAP analysis was carried out to check the importance of different features and plot their effect on prediction
9. Final simplified model selected based on feature analysis consisting of only 9 features.

# conclusion

This model is thought to be useful for:

Government agencies : government bodies can get a closer and faster approximated view on the damage caused by earthquake without manual intervention which can catalyze damage recovery.

Insurers : After a large scale disaster claims systems of Insurers are overwhelmed with large number of new claims.It become harder for claim handlers to go through all of the damage data and decide the damage severity.Integrating claims system with AI based damage assessment services will help claim handlers to look over a single index(damage grade) and decide the severity of damage.Which in terms can results in faster claims processing.

SHAP values can be used to succesfully reduce features in the model to reduce the complexity. It can also be used to help find out which features among others are most important.


# Appendix:
<h2>Description</h2>
<ul>
<li><code>geo1</code>, <code>geo2</code>, <code>geo3</code> (type: int): geographic region in which building exists, from largest (level 1) to most specific sub-region (level 3). Possible values: level 1: 0-30, level 2: 0-1427, level 3: 0-12567.</li>
<li><code>count_floors_pre_eq</code> (type: int): number of floors in the building before the earthquake.</li>
<li><code>age</code> (type: int): age of the building in years.</li>
<li><code>plinth_area_sq_ft</code> (type: int):area of the building footprint.</li>
<li><code>land_surface_condition</code> (type: categorical): surface condition of the land where the building was built.</li>
<li><code>foundation_type</code> (type: categorical): type of foundation used while building.</li>
<li><code>roof_type</code> (type: categorical): type of roof used while building. </li>
<li><code>ground_floor_type</code> (type: categorical): type of the ground floor. </li>
<li><code>other_floor_type</code> (type: categorical): type of constructions used in higher than the ground floors (except of roof).</li>
<li><code>position</code> (type: categorical): position of the building. </li>
<li><code>plan_configuration</code> (type: categorical): building plan configuration. </li>
<li><code>has_superstructure_adobe_mud</code> (type: binary): flag variable that indicates if the superstructure was made of Adobe/Mud.</li>
<li><code>has_superstructure_mud_mortar_stone</code> (type: binary): flag variable that indicates if the superstructure was made of Mud Mortar - Stone.</li>
<li><code>has_superstructure_stone_flag</code> (type: binary): flag variable that indicates if the superstructure was made of Stone.</li>
<li><code>has_superstructure_cement_mortar_stone</code> (type: binary): flag variable that indicates if the superstructure was made of Cement Mortar - Stone.</li>
<li><code>has_superstructure_mud_mortar_brick</code> (type: binary): flag variable that indicates if the superstructure was made of Mud Mortar - Brick.</li>
<li><code>has_superstructure_cement_mortar_brick</code> (type: binary): flag variable that indicates if the superstructure was made of Cement Mortar - Brick.</li>
<li><code>has_superstructure_timber</code> (type: binary): flag variable that indicates if the superstructure was made of Timber.</li>
<li><code>has_superstructure_bamboo</code> (type: binary): flag variable that indicates if the superstructure was made of Bamboo.</li>
<li><code>has_superstructure_rc_non_engineered</code> (type: binary): flag variable that indicates if the superstructure was made of non-engineered reinforced concrete.</li>
<li><code>has_superstructure_rc_engineered</code> (type: binary): flag variable that indicates if the superstructure was made of engineered reinforced concrete.</li>
<li><code>has_superstructure_other</code> (type: binary): flag variable that indicates if the superstructure was made of any other material.</li>
<li><code>legal_ownership_status</code> (type: categorical): legal ownership status of the land where building was built. Possible values: a, r, v, w.</li>
<li><code>count_families</code>  (type: int): number of families that live in the building.</li>
<li><code>has_secondary_use</code> (type: binary): flag variable that indicates if the building was used for any secondary purpose.</li>
<li><code>has_secondary_use_agriculture</code> (type: binary): flag variable that indicates if the building was used for agricultural purposes.</li>
<li><code>has_secondary_use_hotel</code> (type: binary): flag variable that indicates if the building was used as a hotel.</li>
<li><code>has_secondary_use_rental</code> (type: binary): flag variable that indicates if the building was used for rental purposes.</li>
<li><code>has_secondary_use_institution</code> (type: binary): flag variable that indicates if the building was used as a location of any institution.</li>
<li><code>has_secondary_use_school</code> (type: binary): flag variable that indicates if the building was used as a school.</li>
<li><code>has_secondary_use_industry</code> (type: binary): flag variable that indicates if the building was used for industrial purposes.</li>
<li><code>has_secondary_use_health_post</code> (type: binary): flag variable that indicates if the building was used as a health post.</li>
<li><code>has_secondary_use_gov_office</code> (type: binary): flag variable that indicates if the building was used fas a government office.</li>
<li><code>has_secondary_use_use_police</code> (type: binary): flag variable that indicates if the building was used as a police station.</li>
<li><code>has_secondary_use_other</code> (type: binary): flag variable that indicates if the building was secondarily used for other purposes.</li>
</ul>
