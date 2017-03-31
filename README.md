# CCF Research Project
This is the repository for code related to my undergraduate research thesis at John Carroll University.  It represents several months of work, mostly during the summer of 2016.  As my first brush with data science, machine learning, and computer science in general, the code is near and dear, a bit naive, and rough around the edges.  However, this work has improved the state of the art in automated blood test classification, making automated detection feasible for large patient populations.  It's also a testament to the openness and accessibility of the data science and machine learning communities.

### The project
After meeting with Dr. Edmunds Reineks at the [Pathology and Laboratory Medicine Institute at the Cleveland Clinic](http://my.clevelandclinic.org/services/pathology-laboratory-medicine), we decided to apply machine learning to their blood testing data.  Classification of contaminated blood tests is done rather archaically, with rule-based methods the norm in major laboratories worldwide.  We received a small dataset to begin work with, and while the data progressively got larger, the core ideas behind the project remained simple: 
* Limit ourselves to the basic metabolic panel (Blood Urea Nitrogen, Calcium, Carbon Dioxide, Chloride, Creatinine, Glucose, Potassium, Sodium),
* Survey a variety of classifiers (both linear and nonlinear), and
* Use lightweight, agile tools that can be learned and automated quickly by a variety of professionals.

### The tools
* R for initial data cleaning and algorithm prototyping
* Hadoop when we decided to go Big Data
* Python's scikit-learn for Gaussian Mixture Model for data augmentation
* Keras for neural networks
* Spark ML when we decided to go with the all-in-one library for user simplicity

### The results
We are currently writing two papers associated with this research project; one for a clinical audience and one for data scientists.  I'll update with major findings and a link to the papers themselves upon completion and acceptance.
