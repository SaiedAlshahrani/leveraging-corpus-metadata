# Leveraging Corpus Metadata to Detect Template-based Translation: An Exploratory Case Study of the Egyptian Arabic Wikipedia Edition

### **Table of Content**

* **Exploratory Analysis:**
	1. Shallow Content
	2. Poor Quality Content
	3. Misleading Human Involvement
	
* **Experimental Setups:**
	* Dataset Filtering, Labeling, and Cleaning.
	* Dataset Encoding Using Spark-NLP & CAMeLBERT:
		- Encoding with Spark-NLP (Egyptian Word2Vec-CBOW 300D).
		- Encoding with CAMeLBERT (CAMeLBERT-Mix POS-EGY Model).

* **Template Translation Detection:**
	* Supervised Classification Algorithms:
		1. XGBoost
		2. Naive Bayes 
		3. Random Forest
		4. Logistic Regression 
		5. Support Vector Machine 
	* Unsupervised Clustering Algorithms:
		1. DBSCAN.
		2. K-Means.
		3. Hierarchical.
	* Web-based Detection System/Application:
		1. Best-performing Classifier, XGBoost.
		2. Egyptian Arabic Wikipedia Scanner:
			* Streamlit Community Cloud.
			* Hugging Face Spaces.
			

Saied Alshahrani, Hesham Haroon, Ali Elfilali, Mariama Njie, and Jeanna Matthews. 2024. [Leveraging Corpus Metadata to Detect Template-based Translation: An Exploratory Case Study of the Egyptian Arabic Wikipedia Edition](https://arxiv.org/abs/2404.00565). *arXiv preprint arXiv:2404.00565*.

```
@article{alshahrani2024leveraging,
      title={Leveraging Corpus Metadata to Detect Template-based Translation: An Exploratory Case Study of the Egyptian Arabic Wikipedia Edition}, 
      author={Saied Alshahrani and Hesham Haroon and Ali Elfilali and Mariama Njie and Jeanna Matthews},
      year={2024},
      eprint={2404.00565},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
      journal={arXiv preprint arXiv:2404.00565},
      url={https://arxiv.org/abs/2404.00565}
}
```
