# Leveraging Corpus Metadata to Detect Template-based Translation: An Exploratory Case Study of the Egyptian Arabic Wikipedia Edition

We, in this repository, share our labeled datasets, extracted corpora, code and scripts of the exploratory analysis, the multivariate machine learning classifiers and clusters, and the implementation and deployment of the best-performing classifier as a web-based detection system called "*Egyptian Arabic Wikipedia Scanner*", which all are introduced in our accepted paper, [**Leveraging Corpus Metadata to Detect Template-based Translation: An Exploratory Case Study of the Egyptian Arabic Wikipedia Edition**](https://arxiv.org/abs/2404.00565), at [*The 6th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT6)*](https://osact-lrec.github.io/), co-located with [LREC-COLING 2024](https://lrec-coling-2024.org/), 20-25 May 2024. 


* [**Exploratory Analysis:**](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/tree/main/Exploratory-Analysis)
	1. [Shallow Content](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Exploratory-Analysis/Shallow_Content.ipynb)
	2. [Poor Quality Content](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Exploratory-Analysis/Poor_Content_Quality.ipynb)
	3. [Misleading Human Involvement](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Exploratory-Analysis/Misleading_Human_Involvement.ipynb)
	
* [**Experimental Setups:**](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/tree/main/Experimental-Setups)
	* [Dataset Filtering, Labeling, and Cleaning](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Experimental-Setups/Dataset_Filtering_Labeling_Cleaning.ipynb)
	* [Dataset Encoding Using Spark-NLP & CAMeLBERT:](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/tree/main/Experimental-Setups)
		- [Dataset Encoding with Spark-NLP (Egyptian Word2Vec-CBOW 300D)](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Experimental-Setups/Dataset_Encoding_Spark-NLP.ipynb)
		- [Dataset Encoding with CAMeLBERT (CAMeLBERT-Mix POS-EGY Model)](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Experimental-Setups/Dataset_Encoding_CAMeLBERT.ipynb)
		
* [**Template Translation Detection:**](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/tree/main/Template-Translation-Detection)

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
* **Corpora and Datasets:**
	* Arabic Wikipedia Corpora:
		1. Arabic Wikipedia Articles.
		2. Egyptian Wikipedia Articles.
		3. Moroccan Wikipedia Articles.
	* Egyptian Arabic Dataset:
		1. Hugging Face Datasets.
		2. Raw CSV Datasets Files.

* **Paper Citations:**

>Saied Alshahrani, Hesham Haroon, Ali Elfilali, Mariama Njie, and Jeanna Matthews. 2024. [Leveraging Corpus Metadata to Detect Template-based Translation: An Exploratory Case Study of the Egyptian Arabic Wikipedia Edition](https://arxiv.org/abs/2404.00565). *arXiv preprint arXiv:2404.00565*.

<details><summary>BibTeX:</summary> <p align="left"></p>

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
